"""Deterministic refund-eligibility rules.

Pure functions over plain data: no database, no LLM, no clock (the caller passes `now`).
That makes every rule unit-testable and every decision auditable. The result is a
RECOMMENDATION: `requires_human_approval` is always True, because refunds must be approved
by a person.
"""

from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from supportx.db.enums import OrderStatus, RefundReason, RefundStatus


class RefundPolicy(BaseModel):
    """Refund policy parameters.

    DEMO DEFAULTS: these numbers are placeholders, not a real company policy.
    Replace them (and the version label) with the real policy before any real use.
    """

    model_config = ConfigDict(frozen=True)

    version: str = "demo-1"
    return_window_days: int = Field(default=30, gt=0)  # days after delivery
    not_received_after_days: int = Field(default=14, gt=0)  # days after shipping


class DenialCode(StrEnum):
    """Machine-readable reasons a refund is not possible (agents use these to explain)."""

    ORDER_CANCELLED = "order_cancelled"
    PENDING_REQUEST_EXISTS = "pending_request_exists"
    ALL_ITEMS_FINAL_SALE = "all_items_final_sale"
    ALREADY_FULLY_REFUNDED = "already_fully_refunded"
    ORDER_NOT_SHIPPED = "order_not_shipped"
    ORDER_NOT_DELIVERED = "order_not_delivered"
    OUTSIDE_RETURN_WINDOW = "outside_return_window"
    MARKED_DELIVERED = "marked_delivered"
    NOT_RECEIVED_TOO_EARLY = "not_received_too_early"
    AMOUNT_EXCEEDS_REFUNDABLE = "amount_exceeds_refundable"


def _require_aware(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware (use UTC)")
    return value


class ItemFacts(BaseModel):
    model_config = ConfigDict(frozen=True)

    unit_price_cents: int = Field(ge=0)
    quantity: int = Field(gt=0)
    final_sale: bool = False


class ExistingRefund(BaseModel):
    """A refund request already on file for the same order."""

    model_config = ConfigDict(frozen=True)

    amount_cents: int = Field(gt=0)
    status: RefundStatus


class OrderFacts(BaseModel):
    """Everything the rules need to know about an order."""

    model_config = ConfigDict(frozen=True)

    status: OrderStatus
    shipped_at: datetime | None = None
    delivered_at: datetime | None = None
    items: tuple[ItemFacts, ...] = ()

    _check_shipped = field_validator("shipped_at")(_require_aware)
    _check_delivered = field_validator("delivered_at")(_require_aware)


class RefundDecision(BaseModel):
    """Outcome of the rules. Amounts are integer cents."""

    model_config = ConfigDict(frozen=True)

    eligible: bool
    proposed_cents: int  # amount to propose to the human approver (0 if not eligible)
    max_refundable_cents: int  # most that could still be refunded on this order
    reason_codes: tuple[DenialCode, ...] = ()
    explanations: tuple[str, ...] = ()
    requires_human_approval: bool = True
    policy_version: str


def format_money(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


def evaluate_refund(
    order: OrderFacts,
    existing_refunds: tuple[ExistingRefund, ...],
    reason: RefundReason,
    now: datetime,
    policy: RefundPolicy | None = None,
    requested_cents: int | None = None,
) -> RefundDecision:
    """Decide whether a refund can be proposed, and for how much."""
    policy = policy or RefundPolicy()
    if now.tzinfo is None:
        raise ValueError("`now` must be timezone-aware (use UTC)")
    if requested_cents is not None and requested_cents <= 0:
        raise ValueError("requested_cents must be positive")

    codes: list[DenialCode] = []
    notes: list[str] = []

    def result(max_refundable: int = 0) -> RefundDecision:
        eligible = not codes
        proposed = (requested_cents or max_refundable) if eligible else 0
        return RefundDecision(
            eligible=eligible,
            proposed_cents=proposed,
            max_refundable_cents=max_refundable,
            reason_codes=tuple(codes),
            explanations=tuple(notes),
            policy_version=policy.version,
        )

    def deny(code: DenialCode, message: str) -> None:
        codes.append(code)
        notes.append(message)

    # Terminal states: nothing else matters.
    if order.status is OrderStatus.CANCELLED:
        deny(DenialCode.ORDER_CANCELLED, "The order was cancelled, so there is nothing to refund.")
        return result()
    if any(r.status is RefundStatus.PENDING_APPROVAL for r in existing_refunds):
        deny(
            DenialCode.PENDING_REQUEST_EXISTS,
            "A refund request for this order is already waiting for approval.",
        )
        return result()

    # How much money could still be refunded?
    refundable_total = sum(i.unit_price_cents * i.quantity for i in order.items if not i.final_sale)
    already_refunded = sum(
        r.amount_cents for r in existing_refunds if r.status is RefundStatus.APPROVED
    )
    max_refundable = max(refundable_total - already_refunded, 0)
    if refundable_total == 0:
        deny(DenialCode.ALL_ITEMS_FINAL_SALE, "All items on this order are final sale.")
    elif max_refundable == 0:
        deny(DenialCode.ALREADY_FULLY_REFUNDED, "This order has already been fully refunded.")

    # Is the timing right for the stated reason?
    if order.status is OrderStatus.PLACED:
        deny(
            DenialCode.ORDER_NOT_SHIPPED,
            "The order has not shipped yet; it can be cancelled instead of refunded.",
        )
    elif reason is RefundReason.NOT_RECEIVED:
        if order.status is OrderStatus.DELIVERED:
            deny(
                DenialCode.MARKED_DELIVERED,
                "The carrier marked this order as delivered; a person must investigate.",
            )
        elif order.shipped_at is None or now - order.shipped_at < timedelta(
            days=policy.not_received_after_days
        ):
            deny(
                DenialCode.NOT_RECEIVED_TOO_EARLY,
                f"An order can be reported as not received only after "
                f"{policy.not_received_after_days} days from shipping.",
            )
    elif order.status is OrderStatus.SHIPPED or order.delivered_at is None:
        deny(DenialCode.ORDER_NOT_DELIVERED, "The order has not been delivered yet.")
    elif now - order.delivered_at > timedelta(days=policy.return_window_days):
        deny(
            DenialCode.OUTSIDE_RETURN_WINDOW,
            f"The {policy.return_window_days}-day return window after delivery has passed.",
        )

    # Is the requested amount allowed?
    if requested_cents is not None and max_refundable > 0 and requested_cents > max_refundable:
        deny(
            DenialCode.AMOUNT_EXCEEDS_REFUNDABLE,
            f"The requested amount {format_money(requested_cents)} is more than the "
            f"refundable amount {format_money(max_refundable)}.",
        )

    return result(max_refundable)
