from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine

from supportx.core import NotFoundError, WorkflowError
from supportx.db import Order, OrderStatus, Shipment, session_scope
from supportx.db.repositories import OrderRepository
from supportx.services import (
    CancellationWorkflow,
    evaluate_cancellation,
    evaluate_cancellation_for_state,
)
from tests.unit.helpers import EMAIL, make_engine


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = make_engine()
    yield eng
    eng.dispose()


def _order(engine: Engine, number: str) -> Order:
    with session_scope(engine) as session:
        order = OrderRepository(session).get_by_number(number)
        assert order is not None
        return order


# ---------------------------------------------------------------- the pure rule
def test_only_an_unshipped_placed_order_is_eligible() -> None:
    assert evaluate_cancellation("placed", shipped=False).eligible
    assert not evaluate_cancellation("placed", shipped=True).eligible  # inconsistent data: be safe
    assert not evaluate_cancellation("shipped", shipped=True).eligible
    assert not evaluate_cancellation("delivered", shipped=True).eligible
    assert "already cancelled" in evaluate_cancellation("cancelled", shipped=False).explanation


def test_the_rule_reads_the_order_summary_kept_in_ticket_state() -> None:
    assert evaluate_cancellation_for_state({"status": "placed", "shipment": None}).eligible
    shipped = {"status": "placed", "shipment": {"shipped_at": "2026-11-01T00:00:00Z"}}
    assert not evaluate_cancellation_for_state(shipped).eligible
    assert not evaluate_cancellation_for_state({"status": "delivered", "shipment": {}}).eligible


# ---------------------------------------------------------------- the action
def test_cancel_changes_status_and_records_who_and_when(engine: Engine) -> None:
    moment = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)
    with session_scope(engine) as session:
        CancellationWorkflow(session).cancel("ORD-3003", EMAIL, "agent.jane", now=moment)
    order = _order(engine, "ORD-3003")
    assert order.status is OrderStatus.CANCELLED
    assert order.cancelled_by == "agent.jane"
    assert order.cancelled_at is not None and order.cancelled_at.replace(tzinfo=UTC) == moment


def test_cancel_is_repeatable_and_keeps_the_first_approver(engine: Engine) -> None:
    with session_scope(engine) as session:
        CancellationWorkflow(session).cancel("ORD-3003", EMAIL, "agent.jane")
    with session_scope(engine) as session:
        CancellationWorkflow(session).cancel("ORD-3003", EMAIL, "agent.bob")  # a retry
    assert _order(engine, "ORD-3003").cancelled_by == "agent.jane"


def test_cancel_refuses_a_shipped_order_and_changes_nothing(engine: Engine) -> None:
    with pytest.raises(WorkflowError, match="can no longer be cancelled"):
        with session_scope(engine) as session:
            CancellationWorkflow(session).cancel("ORD-1001", EMAIL, "agent.jane")
    assert _order(engine, "ORD-1001").status is OrderStatus.DELIVERED


def test_the_rule_is_rechecked_when_the_order_ships_while_waiting(engine: Engine) -> None:
    with session_scope(engine) as session:  # shipped after the customer asked
        order = OrderRepository(session).get_by_number("ORD-4004")
        assert order is not None
        order.shipment = Shipment(
            carrier="DemoPost",
            tracking_number="T",
            shipped_at=datetime.now(UTC) - timedelta(hours=1),
        )
    with pytest.raises(WorkflowError):
        with session_scope(engine) as session:
            CancellationWorkflow(session).cancel("ORD-4004", EMAIL, "agent.jane")
    assert _order(engine, "ORD-4004").status is OrderStatus.PLACED


def test_a_wrong_email_or_number_gives_the_same_error(engine: Engine) -> None:
    for number, email in (("ORD-3003", "thief@example.com"), ("ORD-9999", EMAIL)):
        with pytest.raises(NotFoundError) as caught:
            with session_scope(engine) as session:
                CancellationWorkflow(session).cancel(number, email, "agent.jane")
        assert str(caught.value) == "No order with that number was found for this customer."
    assert _order(engine, "ORD-3003").status is OrderStatus.PLACED
