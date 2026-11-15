"""Refund workflow: check the rules, create a request, let a HUMAN approve or reject it.

The caller owns the transaction (use `session_scope`); this class only flushes.
"""

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from supportx.core import NotFoundError, WorkflowError, get_logger
from supportx.db.enums import RefundReason, RefundStatus
from supportx.db.models import Order, RefundRequest
from supportx.db.repositories import OrderRepository, RefundRepository, to_utc
from supportx.services.orders import ensure_owner, find_order_for_customer
from supportx.services.refund import (
    ExistingRefund,
    ItemFacts,
    OrderFacts,
    RefundDecision,
    RefundPolicy,
    evaluate_refund,
    format_money,
    refundable_cents,
)

logger = get_logger(__name__)


def order_to_facts(
    order: Order, exclude_refund_id: int | None = None
) -> tuple[OrderFacts, tuple[ExistingRefund, ...]]:
    """Translate a database order into the plain data the rule engine understands.

    `exclude_refund_id` leaves one refund request out of the "existing refunds" list
    (used when re-checking a request that is about to be approved).
    """
    shipment = order.shipment
    facts = OrderFacts(
        status=order.status,
        shipped_at=to_utc(shipment.shipped_at) if shipment else None,
        delivered_at=to_utc(shipment.delivered_at) if shipment else None,
        items=tuple(
            ItemFacts(
                unit_price_cents=i.unit_price_cents,
                quantity=i.quantity,
                final_sale=i.final_sale,
            )
            for i in order.items
        ),
    )
    existing = tuple(
        ExistingRefund(amount_cents=r.amount_cents, status=r.status)
        for r in order.refund_requests
        if r.id != exclude_refund_id
    )
    return facts, existing


class RefundWorkflow:
    def __init__(self, session: Session, policy: RefundPolicy | None = None):
        self._session = session
        self._policy = policy or RefundPolicy()
        self._orders = OrderRepository(session)
        self._refunds = RefundRepository(session)

    def check_eligibility(
        self,
        order_number: str,
        customer_email: str,
        reason: RefundReason,
        requested_cents: int | None = None,
        now: datetime | None = None,
    ) -> RefundDecision:
        """Run the rules without creating anything."""
        order = find_order_for_customer(self._session, order_number, customer_email)
        facts, existing = order_to_facts(order)
        return evaluate_refund(
            facts, existing, reason, now or datetime.now(UTC), self._policy, requested_cents
        )

    def request_refund(
        self,
        order_number: str,
        customer_email: str,
        reason: RefundReason,
        requested_cents: int | None = None,
        now: datetime | None = None,
    ) -> tuple[RefundDecision, RefundRequest | None]:
        """Create a PENDING refund request if the rules allow one; money never moves here."""
        # One locked, freshly-read order: the owner check and the rules see the same data.
        order = ensure_owner(
            self._orders.get_by_number(order_number, for_update=True), customer_email
        )
        facts, existing = order_to_facts(order)
        decision = evaluate_refund(
            facts, existing, reason, now or datetime.now(UTC), self._policy, requested_cents
        )
        if not decision.eligible:
            logger.info(
                "Refund not allowed for %s: %s",
                order.order_number,
                [c.value for c in decision.reason_codes],
            )
            return decision, None
        request = self._refunds.add(order, decision.proposed_cents, reason)
        logger.info(
            "Refund request %d created for %s, awaiting approval", request.id, order.order_number
        )
        return decision, request

    def approve(
        self,
        refund_id: int,
        approver: str,
        note: str | None = None,
        now: datetime | None = None,
    ) -> RefundRequest:
        """A human approves. Re-checks that the amount is still refundable."""
        request = self._load_pending(refund_id)
        facts, others = order_to_facts(request.order, exclude_refund_id=request.id)
        still_refundable = refundable_cents(facts, others)
        if request.amount_cents > still_refundable:
            raise WorkflowError(
                f"Cannot approve {format_money(request.amount_cents)}: only "
                f"{format_money(still_refundable)} is still refundable on this order."
            )
        return self._decide(request, RefundStatus.APPROVED, approver, note, now)

    def reject(
        self, refund_id: int, approver: str, note: str, now: datetime | None = None
    ) -> RefundRequest:
        """A human rejects. A reason is mandatory."""
        if not note or not note.strip():
            raise ValueError("A reason (note) is required when rejecting a refund.")
        request = self._load_pending(refund_id)
        return self._decide(request, RefundStatus.REJECTED, approver, note, now)

    def _load_pending(self, refund_id: int) -> RefundRequest:
        request = self._refunds.get(refund_id, for_update=True)
        if request is None:
            raise NotFoundError(f"Refund request {refund_id} does not exist.")
        if request.status is not RefundStatus.PENDING_APPROVAL:
            raise WorkflowError(
                f"Refund request {refund_id} was already {request.status.value}; "
                "it cannot be decided again."
            )
        return request

    def _decide(
        self,
        request: RefundRequest,
        status: RefundStatus,
        approver: str,
        note: str | None,
        now: datetime | None,
    ) -> RefundRequest:
        if not approver or not approver.strip():
            raise ValueError("The approver's identity is required.")
        request.status = status
        request.decided_by = approver.strip()
        request.decided_at = now or datetime.now(UTC)
        request.decision_note = note.strip() if note else None
        self._session.flush()
        logger.info("Refund request %d %s by %s", request.id, status.value, request.decided_by)
        return request
