"""Concurrency tests that need a REAL PostgreSQL (row locks are ignored by SQLite).

NOT RUN by default: they are skipped unless TEST_POSTGRES_URL points at a disposable database, e.g.

    set TEST_POSTGRES_URL=postgresql+psycopg://supportx:pw@localhost:5432/supportx_test
    pytest tests/integration/test_postgres_concurrency.py

Each test creates its own uniquely named customer and order and removes them afterwards, but use a
throwaway database anyway: it also runs `create_all`.
"""

import os
import threading
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, delete, select

from supportx.db import (
    Customer,
    Order,
    OrderItem,
    OrderStatus,
    RefundRequest,
    init_db,
    session_scope,
)
from supportx.db.enums import RefundReason
from supportx.services import CancellationWorkflow, RefundWorkflow

URL = os.environ.get("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_POSTGRES_URL is not set (PostgreSQL needed)")


@pytest.fixture
def engine() -> Iterator[Engine]:
    assert URL
    eng = create_engine(URL, pool_size=5)
    init_db(eng)
    yield eng
    eng.dispose()


def _seed(engine: Engine, status: OrderStatus) -> tuple[str, str]:
    """A customer and one order with a 25.00 item; returns (order_number, email)."""
    tag = uuid.uuid4().hex[:10]
    number, email = f"ORD-{int(tag, 16) % 10**9}", f"{tag}@example.com"
    with session_scope(engine) as session:
        session.add(
            Order(
                order_number=number,
                customer=Customer(email=email, name="Concurrency Test"),
                status=status,
                items=[OrderItem(sku="M", name="Mug", unit_price_cents=2500, quantity=1)],
            )
        )
    return number, email


def _cleanup(engine: Engine, number: str) -> None:
    with session_scope(engine) as session:
        order = session.scalars(select(Order).where(Order.order_number == number)).one()
        session.execute(delete(RefundRequest).where(RefundRequest.order_id == order.id))
        session.execute(delete(OrderItem).where(OrderItem.order_id == order.id))
        customer_id = order.customer_id
        session.delete(order)
        session.flush()
        session.execute(delete(Customer).where(Customer.id == customer_id))


def _race(workers: list[Callable[[], Any]]) -> list[Any]:
    """Start every worker at the same moment; return each result or the exception it raised."""
    barrier = threading.Barrier(len(workers))
    results: list[Any] = [None] * len(workers)

    def run(i: int) -> None:
        barrier.wait()
        try:
            results[i] = workers[i]()
        except Exception as exc:  # reported to the test, which decides what is acceptable
            results[i] = exc

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    return results


def test_two_simultaneous_cancellations_cancel_once(engine: Engine) -> None:
    number, email = _seed(engine, OrderStatus.PLACED)
    try:

        def cancel(approver: str) -> Callable[[], str]:
            def run() -> str:
                with session_scope(engine) as session:
                    CancellationWorkflow(session).cancel(number, email, approver)
                return approver

            return run

        results = _race([cancel("jane"), cancel("bob")])
        assert not any(isinstance(r, Exception) for r in results), (
            results
        )  # both succeed (idempotent)
        with session_scope(engine) as session:
            order = session.scalars(select(Order).where(Order.order_number == number)).one()
            assert order.status is OrderStatus.CANCELLED
            assert order.cancelled_by in {"jane", "bob"}  # exactly one approver was recorded
    finally:
        _cleanup(engine, number)


def test_two_simultaneous_refund_requests_create_exactly_one(engine: Engine) -> None:
    number, email = _seed(engine, OrderStatus.DELIVERED)
    try:
        with session_scope(engine) as session:  # make the order refundable: delivered just now
            from datetime import UTC, datetime

            from supportx.db import Shipment

            order = session.scalars(select(Order).where(Order.order_number == number)).one()
            now = datetime.now(UTC)
            order.shipment = Shipment(
                carrier="DemoPost", tracking_number="T", shipped_at=now, delivered_at=now
            )

        def request() -> bool:
            with session_scope(engine) as session:
                _, created = RefundWorkflow(session).request_refund(
                    number, email, RefundReason.DAMAGED
                )
                return created is not None

        results = _race([request, request, request])
        assert not any(isinstance(r, Exception) for r in results), results
        assert sum(results) == 1  # the row lock makes the "one pending request" rule hold
        with session_scope(engine) as session:
            order = session.scalars(select(Order).where(Order.order_number == number)).one()
            assert len(order.refund_requests) == 1
    finally:
        _cleanup(engine, number)
