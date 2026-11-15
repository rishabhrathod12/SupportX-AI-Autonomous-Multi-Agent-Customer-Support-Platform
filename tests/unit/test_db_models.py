from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.exc import IntegrityError

from supportx.core import ConfigurationError, Settings
from supportx.db import (
    Customer,
    Order,
    OrderItem,
    OrderStatus,
    RefundReason,
    RefundRequest,
    RefundStatus,
    Shipment,
    init_db,
    make_engine,
    session_scope,
)


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = create_engine("sqlite://")  # in-memory SQLite: tests need no PostgreSQL
    init_db(eng)
    yield eng
    eng.dispose()


def make_order(number: str = "ORD-1001") -> Order:
    customer = Customer(email=f"{number.lower()}@example.com", name="Test Customer")
    return Order(
        order_number=number,
        customer=customer,
        items=[OrderItem(sku="SKU-1", name="Mug", unit_price_cents=1250, quantity=2)],
        shipment=Shipment(carrier="DemoPost", tracking_number="TRK123"),
    )


def test_round_trip_and_defaults(engine: Engine) -> None:
    with session_scope(engine) as session:
        session.add(make_order())

    with session_scope(engine) as session:
        order = session.scalars(select(Order).where(Order.order_number == "ORD-1001")).one()
        assert order.status is OrderStatus.PLACED  # default
        assert order.customer.name == "Test Customer"
        assert [(i.sku, i.unit_price_cents, i.quantity, i.final_sale) for i in order.items] == [
            ("SKU-1", 1250, 2, False)
        ]
        assert order.shipment is not None and order.shipment.carrier == "DemoPost"
        assert order.placed_at is not None


def test_enums_are_stored_as_readable_strings(engine: Engine) -> None:
    with session_scope(engine) as session:
        order = make_order()
        order.status = OrderStatus.DELIVERED
        order.refund_requests.append(RefundRequest(amount_cents=500, reason=RefundReason.DAMAGED))
        session.add(order)

    with engine.connect() as conn:
        assert conn.execute(text("select status from orders")).scalar_one() == "delivered"
        row = conn.execute(text("select reason, status from refund_requests")).one()
        assert tuple(row) == ("damaged", "pending_approval")


def test_refund_request_waits_for_a_human_by_default(engine: Engine) -> None:
    with session_scope(engine) as session:
        order = make_order()
        order.refund_requests.append(
            RefundRequest(amount_cents=700, reason=RefundReason.CHANGED_MIND)
        )
        session.add(order)

    with session_scope(engine) as session:
        refund = session.scalars(select(RefundRequest)).one()
        assert refund.status is RefundStatus.PENDING_APPROVAL
        assert refund.decided_at is None and refund.decided_by is None
        refund.status = RefundStatus.APPROVED
        refund.decided_by = "agent.jane"
        refund.decided_at = datetime.now(UTC)

    with session_scope(engine) as session:
        refund = session.scalars(select(RefundRequest)).one()
        assert refund.status is RefundStatus.APPROVED and refund.decided_by == "agent.jane"


def test_order_number_must_be_unique(engine: Engine) -> None:
    with session_scope(engine) as session:
        session.add(make_order("ORD-7"))
    with pytest.raises(IntegrityError):
        with session_scope(engine) as session:
            session.add(make_order("ORD-7"))


@pytest.mark.parametrize(
    "bad_item",
    [
        {"unit_price_cents": 100, "quantity": 0},
        {"unit_price_cents": -5, "quantity": 1},
    ],
)
def test_check_constraints_reject_bad_items(engine: Engine, bad_item: dict[str, int]) -> None:
    order = make_order()
    order.items = [OrderItem(sku="X", name="Bad", **bad_item)]
    with pytest.raises(IntegrityError):
        with session_scope(engine) as session:
            session.add(order)


def test_refund_amount_must_be_positive(engine: Engine) -> None:
    order = make_order()
    order.refund_requests.append(RefundRequest(amount_cents=0, reason=RefundReason.DAMAGED))
    with pytest.raises(IntegrityError):
        with session_scope(engine) as session:
            session.add(order)


def test_session_scope_rolls_back_on_error(engine: Engine) -> None:
    with pytest.raises(RuntimeError):
        with session_scope(engine) as session:
            session.add(make_order("ORD-ROLLBACK"))
            session.flush()
            raise RuntimeError("boom")
    with session_scope(engine) as session:
        assert session.scalars(select(Order)).all() == []


def test_deleting_an_order_deletes_its_items_and_shipment(engine: Engine) -> None:
    with session_scope(engine) as session:
        session.add(make_order())
    with session_scope(engine) as session:
        session.delete(session.scalars(select(Order)).one())
    with engine.connect() as conn:
        assert conn.execute(text("select count(*) from order_items")).scalar_one() == 0
        assert conn.execute(text("select count(*) from shipments")).scalar_one() == 0


def test_make_engine_needs_a_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "supportx.db.session.get_settings",
        lambda: Settings(_env_file=None),  # type: ignore[call-arg]
    )
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        make_engine()
    assert make_engine("sqlite://").dialect.name == "sqlite"
