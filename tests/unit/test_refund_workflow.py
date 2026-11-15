from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.orm import Session

from supportx.core import NotFoundError, WorkflowError
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
    session_scope,
)
from supportx.db.repositories import OrderRepository, RefundRepository, to_utc
from supportx.services import DenialCode, OrderService, RefundPolicy, RefundWorkflow

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
EMAIL = "Test@Example.com"


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = create_engine("sqlite://")
    init_db(eng)
    with session_scope(eng) as session:
        seed_order(session, "ORD-1001", delivered_days_ago=5)
    yield eng
    eng.dispose()


def seed_order(
    session: Session,
    number: str,
    delivered_days_ago: int,
    email: str = EMAIL,
) -> Order:
    customer = session.scalars(select(Customer).where(Customer.email == email)).first()
    order = Order(
        order_number=number,
        customer=customer or Customer(email=email, name="Test Customer"),
        status=OrderStatus.DELIVERED,
        items=[
            OrderItem(sku="A", name="Mug", unit_price_cents=2000, quantity=2),
            OrderItem(sku="B", name="Plate", unit_price_cents=1500, quantity=1),
        ],  # refundable total 55.00
        shipment=Shipment(
            carrier="DemoPost",
            tracking_number="TRK-" + number,
            shipped_at=NOW - timedelta(days=delivered_days_ago + 3),
            delivered_at=NOW - timedelta(days=delivered_days_ago),
        ),
    )
    session.add(order)
    return order


def refund_count(engine: Engine) -> int:
    with session_scope(engine) as session:
        return session.scalar(select(func.count()).select_from(RefundRequest)) or 0


# ---------------------------------------------------------------- repositories / helpers
def test_to_utc_handles_naive_aware_and_none() -> None:
    assert to_utc(None) is None
    naive = to_utc(datetime(2026, 1, 1, 10, 0))
    assert naive == datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    plus_two = datetime(2026, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    assert to_utc(plus_two) == datetime(2026, 1, 1, 10, 0, tzinfo=UTC)


def test_order_number_lookup_is_normalized(engine: Engine) -> None:
    with session_scope(engine) as session:
        repo = OrderRepository(session)
        assert repo.get_by_number("  ord-1001 ") is not None
        assert repo.get_by_number("ORD-9999") is None


def test_list_pending_is_oldest_first(engine: Engine) -> None:
    with session_scope(engine) as session:
        order = OrderRepository(session).get_by_number("ORD-1001")
        assert order is not None
        repo = RefundRepository(session)
        first = repo.add(order, 100, RefundReason.DAMAGED)
        second = repo.add(order, 200, RefundReason.DAMAGED)
        second.status = RefundStatus.REJECTED
        assert [r.id for r in repo.list_pending()] == [first.id]


# ---------------------------------------------------------------- order lookup
def test_order_summary_for_the_owner(engine: Engine) -> None:
    with session_scope(engine) as session:
        summary = OrderService(session).get_order_summary(" ord-1001", "test@EXAMPLE.com")
    assert summary.order_number == "ORD-1001"
    assert summary.status is OrderStatus.DELIVERED
    assert summary.total_cents == 5500
    assert [i.name for i in summary.items] == ["Mug", "Plate"]
    assert summary.shipment is not None and summary.shipment.tracking_number == "TRK-ORD-1001"
    assert summary.shipment.delivered_at == NOW - timedelta(days=5)  # UTC-aware even on SQLite
    assert summary.has_pending_refund is False


def test_wrong_email_and_missing_order_look_identical(engine: Engine) -> None:
    with session_scope(engine) as session:
        service = OrderService(session)
        with pytest.raises(NotFoundError) as wrong_email:
            service.get_order_summary("ORD-1001", "someone.else@example.com")
        with pytest.raises(NotFoundError) as missing:
            service.get_order_summary("ORD-4040", EMAIL)
    assert str(wrong_email.value) == str(missing.value)  # no way to tell which order numbers exist


# ---------------------------------------------------------------- requesting refunds
def test_eligible_request_creates_a_pending_refund(engine: Engine) -> None:
    with session_scope(engine) as session:
        decision, request = RefundWorkflow(session).request_refund(
            "ORD-1001", EMAIL, RefundReason.DAMAGED, now=NOW
        )
        assert decision.eligible and request is not None
        assert request.status is RefundStatus.PENDING_APPROVAL
        assert request.amount_cents == 5500
    assert refund_count(engine) == 1
    with session_scope(engine) as session:
        assert OrderService(session).get_order_summary("ORD-1001", EMAIL).has_pending_refund


def test_partial_amount_is_respected(engine: Engine) -> None:
    with session_scope(engine) as session:
        _, request = RefundWorkflow(session).request_refund(
            "ORD-1001", EMAIL, RefundReason.CHANGED_MIND, requested_cents=1500, now=NOW
        )
        assert request is not None and request.amount_cents == 1500


def test_second_request_is_blocked_while_one_is_pending(engine: Engine) -> None:
    with session_scope(engine) as session:
        RefundWorkflow(session).request_refund("ORD-1001", EMAIL, RefundReason.DAMAGED, now=NOW)
    with session_scope(engine) as session:
        decision, request = RefundWorkflow(session).request_refund(
            "ORD-1001", EMAIL, RefundReason.DAMAGED, now=NOW
        )
    assert request is None and decision.reason_codes == (DenialCode.PENDING_REQUEST_EXISTS,)
    assert refund_count(engine) == 1


def test_ineligible_request_creates_nothing(engine: Engine) -> None:
    with session_scope(engine) as session:
        decision, request = RefundWorkflow(session).request_refund(
            "ORD-1001", EMAIL, RefundReason.DAMAGED, now=NOW + timedelta(days=60)
        )
    assert request is None and DenialCode.OUTSIDE_RETURN_WINDOW in decision.reason_codes
    assert refund_count(engine) == 0


def test_check_eligibility_creates_nothing_and_uses_the_policy(engine: Engine) -> None:
    with session_scope(engine) as session:
        strict = RefundWorkflow(session, RefundPolicy(version="strict", return_window_days=1))
        decision = strict.check_eligibility("ORD-1001", EMAIL, RefundReason.DAMAGED, now=NOW)
    assert not decision.eligible and decision.policy_version == "strict"
    assert refund_count(engine) == 0


def test_only_the_owner_can_request_a_refund(engine: Engine) -> None:
    with pytest.raises(NotFoundError):
        with session_scope(engine) as session:
            RefundWorkflow(session).request_refund(
                "ORD-1001", "intruder@example.com", RefundReason.DAMAGED, now=NOW
            )
    assert refund_count(engine) == 0


# ---------------------------------------------------------------- human decisions
def pending_request(engine: Engine) -> int:
    with session_scope(engine) as session:
        _, request = RefundWorkflow(session).request_refund(
            "ORD-1001", EMAIL, RefundReason.DAMAGED, now=NOW
        )
        assert request is not None
        return request.id


def test_approval_records_who_when_and_why(engine: Engine) -> None:
    refund_id = pending_request(engine)
    with session_scope(engine) as session:
        RefundWorkflow(session).approve(refund_id, " agent.jane ", "Photo confirms damage", now=NOW)
    with session_scope(engine) as session:
        saved = session.get(RefundRequest, refund_id)
        assert saved is not None
        assert saved.status is RefundStatus.APPROVED
        assert saved.decided_by == "agent.jane"
        assert saved.decision_note == "Photo confirms damage"
        assert to_utc(saved.decided_at) == NOW


def test_a_request_cannot_be_decided_twice(engine: Engine) -> None:
    refund_id = pending_request(engine)
    with session_scope(engine) as session:
        RefundWorkflow(session).approve(refund_id, "agent.jane")
    for action in ("approve", "reject"):
        with pytest.raises(WorkflowError, match="already approved"):
            with session_scope(engine) as session:
                workflow = RefundWorkflow(session)
                if action == "approve":
                    workflow.approve(refund_id, "agent.bob")
                else:
                    workflow.reject(refund_id, "agent.bob", "changed my mind")


def test_rejection_needs_a_reason_and_an_approver(engine: Engine) -> None:
    refund_id = pending_request(engine)
    with session_scope(engine) as session:
        workflow = RefundWorkflow(session)
        with pytest.raises(ValueError, match="reason"):
            workflow.reject(refund_id, "agent.jane", "   ")
        with pytest.raises(ValueError, match="approver"):
            workflow.approve(refund_id, "  ")
        with pytest.raises(ValueError, match="approver"):
            workflow.reject(refund_id, "", "no photo")
        session.rollback()


def test_rejected_refund_allows_a_new_request(engine: Engine) -> None:
    refund_id = pending_request(engine)
    with session_scope(engine) as session:
        RefundWorkflow(session).reject(refund_id, "agent.jane", "Damage not visible in photo")
    with session_scope(engine) as session:
        decision, request = RefundWorkflow(session).request_refund(
            "ORD-1001", EMAIL, RefundReason.DAMAGED, now=NOW
        )
    assert decision.eligible and request is not None


def test_after_full_approval_nothing_is_left_to_refund(engine: Engine) -> None:
    refund_id = pending_request(engine)
    with session_scope(engine) as session:
        RefundWorkflow(session).approve(refund_id, "agent.jane")
    with session_scope(engine) as session:
        decision, request = RefundWorkflow(session).request_refund(
            "ORD-1001", EMAIL, RefundReason.DAMAGED, now=NOW
        )
    assert request is None and DenialCode.ALREADY_FULLY_REFUNDED in decision.reason_codes


def test_unknown_refund_id(engine: Engine) -> None:
    with pytest.raises(NotFoundError):
        with session_scope(engine) as session:
            RefundWorkflow(session).approve(999, "agent.jane")


def test_approval_rechecks_the_refundable_amount(engine: Engine) -> None:
    refund_id = pending_request(engine)  # proposes the full 55.00
    with session_scope(engine) as session:  # meanwhile 20.00 gets approved some other way
        order = OrderRepository(session).get_by_number("ORD-1001")
        assert order is not None
        order.refund_requests.append(
            RefundRequest(
                amount_cents=2000, reason=RefundReason.DAMAGED, status=RefundStatus.APPROVED
            )
        )
    with pytest.raises(WorkflowError, match="only 35.00 is still refundable"):
        with session_scope(engine) as session:
            RefundWorkflow(session).approve(refund_id, "agent.jane")
    with session_scope(engine) as session:
        saved = session.get(RefundRequest, refund_id)
        assert saved is not None and saved.status is RefundStatus.PENDING_APPROVAL
