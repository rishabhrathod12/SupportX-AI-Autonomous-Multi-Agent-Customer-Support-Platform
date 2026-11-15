from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from supportx.db.enums import OrderStatus, RefundReason, RefundStatus
from supportx.services import (
    DenialCode,
    ExistingRefund,
    ItemFacts,
    OrderFacts,
    RefundPolicy,
    evaluate_refund,
)
from supportx.services.refund import format_money

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
ITEMS = (
    ItemFacts(unit_price_cents=2000, quantity=2),  # 40.00
    ItemFacts(unit_price_cents=1500, quantity=1),  # 15.00  -> 55.00 refundable
)


def delivered(days_ago: float, items: tuple[ItemFacts, ...] = ITEMS) -> OrderFacts:
    return OrderFacts(
        status=OrderStatus.DELIVERED,
        shipped_at=NOW - timedelta(days=days_ago + 3),
        delivered_at=NOW - timedelta(days=days_ago),
        items=items,
    )


def shipped(days_ago: float) -> OrderFacts:
    return OrderFacts(
        status=OrderStatus.SHIPPED, shipped_at=NOW - timedelta(days=days_ago), items=ITEMS
    )


def decide(order: OrderFacts, reason: RefundReason = RefundReason.DAMAGED, **kwargs):  # type: ignore[no-untyped-def]
    existing = kwargs.pop("existing", ())
    return evaluate_refund(order, existing, reason, NOW, **kwargs)


def test_recent_delivered_damaged_order_is_eligible_for_the_full_amount() -> None:
    d = decide(delivered(5))
    assert d.eligible and d.reason_codes == ()
    assert d.proposed_cents == d.max_refundable_cents == 5500
    assert d.requires_human_approval is True
    assert d.policy_version == "demo-1"


def test_return_window_boundary() -> None:
    assert decide(delivered(30)).eligible  # exactly 30 days: still inside
    late = decide(delivered(30.01))
    assert not late.eligible and late.reason_codes == (DenialCode.OUTSIDE_RETURN_WINDOW,)
    assert late.proposed_cents == 0


def test_custom_policy_window() -> None:
    policy = RefundPolicy(version="strict-7", return_window_days=7)
    assert not decide(delivered(10), policy=policy).eligible
    assert decide(delivered(10), policy=policy).policy_version == "strict-7"


def test_every_decision_requires_human_approval() -> None:
    for order in (delivered(1), delivered(90), shipped(1)):
        assert decide(order).requires_human_approval is True


def test_cancelled_order_stops_immediately() -> None:
    d = decide(OrderFacts(status=OrderStatus.CANCELLED, items=ITEMS))
    assert d.reason_codes == (DenialCode.ORDER_CANCELLED,)


def test_placed_and_shipped_orders_cannot_be_returned_yet() -> None:
    placed = decide(OrderFacts(status=OrderStatus.PLACED, items=ITEMS))
    assert placed.reason_codes == (DenialCode.ORDER_NOT_SHIPPED,)
    assert decide(shipped(2)).reason_codes == (DenialCode.ORDER_NOT_DELIVERED,)


def test_not_received_needs_enough_days_since_shipping() -> None:
    reason = RefundReason.NOT_RECEIVED
    assert decide(shipped(14), reason).eligible  # exactly 14 days
    early = decide(shipped(5), reason)
    assert early.reason_codes == (DenialCode.NOT_RECEIVED_TOO_EARLY,)


def test_not_received_but_carrier_says_delivered_goes_to_a_person() -> None:
    d = decide(delivered(2), RefundReason.NOT_RECEIVED)
    assert d.reason_codes == (DenialCode.MARKED_DELIVERED,)


def test_final_sale_items_are_excluded_from_the_amount() -> None:
    items = (*ITEMS, ItemFacts(unit_price_cents=9900, quantity=1, final_sale=True))
    assert decide(delivered(1, items)).max_refundable_cents == 5500


def test_all_final_sale() -> None:
    items = (ItemFacts(unit_price_cents=1000, quantity=1, final_sale=True),)
    assert decide(delivered(1, items)).reason_codes == (DenialCode.ALL_ITEMS_FINAL_SALE,)


def test_pending_request_blocks_a_second_one() -> None:
    pending = (ExistingRefund(amount_cents=1000, status=RefundStatus.PENDING_APPROVAL),)
    d = decide(delivered(1), existing=pending)
    assert d.reason_codes == (DenialCode.PENDING_REQUEST_EXISTS,)


def test_approved_refunds_reduce_what_is_left_and_rejected_ones_do_not() -> None:
    approved = ExistingRefund(amount_cents=2000, status=RefundStatus.APPROVED)
    rejected = ExistingRefund(amount_cents=9000, status=RefundStatus.REJECTED)
    d = decide(delivered(1), existing=(approved, rejected))
    assert d.eligible and d.max_refundable_cents == 3500


def test_fully_refunded_order() -> None:
    approved = ExistingRefund(amount_cents=5500, status=RefundStatus.APPROVED)
    d = decide(delivered(1), existing=(approved,))
    assert d.reason_codes == (DenialCode.ALREADY_FULLY_REFUNDED,)


def test_requested_amount_partial_and_too_large() -> None:
    partial = decide(delivered(1), requested_cents=1000)
    assert partial.eligible and partial.proposed_cents == 1000

    too_much = decide(delivered(1), requested_cents=5501)
    assert too_much.reason_codes == (DenialCode.AMOUNT_EXCEEDS_REFUNDABLE,)
    assert too_much.max_refundable_cents == 5500 and too_much.proposed_cents == 0
    assert "55.00" in too_much.explanations[0]


def test_several_problems_are_all_reported() -> None:
    d = decide(delivered(40), requested_cents=99999)
    assert set(d.reason_codes) == {
        DenialCode.OUTSIDE_RETURN_WINDOW,
        DenialCode.AMOUNT_EXCEEDS_REFUNDABLE,
    }


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="requested_cents"):
        decide(delivered(1), requested_cents=0)
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate_refund(delivered(1), (), RefundReason.DAMAGED, datetime(2026, 11, 15))
    with pytest.raises(ValidationError):
        OrderFacts(status=OrderStatus.DELIVERED, delivered_at=datetime(2026, 11, 1))
    with pytest.raises(ValidationError):
        ItemFacts(unit_price_cents=-1, quantity=1)


def test_decisions_and_policy_are_immutable() -> None:
    d = decide(delivered(1))
    with pytest.raises(ValidationError):
        d.eligible = False  # type: ignore[misc]
    with pytest.raises(ValidationError):
        RefundPolicy().return_window_days = 99  # type: ignore[misc]


def test_format_money() -> None:
    assert format_money(5500) == "55.00"
    assert format_money(5) == "0.05"
    assert format_money(123456) == "1234.56"
