"""Order cancellation: the rule, and the action a person approves.

RULE (DEMO DEFAULT, matches data/sample/policies/cancellation_policy.md): an order can be cancelled
only while it has not shipped. Anything else is denied with a reason. Even an eligible cancellation
is only carried out after a human approves it.

The rule is pure (`evaluate_cancellation`) so the Supervisor can use it without a database; the
action (`CancellationWorkflow.cancel`) re-checks the rule at approval time, because the order may
have shipped while the ticket was waiting.
"""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from supportx.core import WorkflowError, get_logger
from supportx.db.enums import OrderStatus
from supportx.db.models import Order
from supportx.db.repositories import OrderRepository
from supportx.services.orders import ensure_owner

logger = get_logger(__name__)


class CancellationDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    eligible: bool
    explanation: str  # safe to show a customer


def evaluate_cancellation(status: str, shipped: bool) -> CancellationDecision:
    """Can an order with this status be cancelled? `shipped` = a shipment has a shipped date."""
    if status == OrderStatus.CANCELLED:
        return CancellationDecision(eligible=False, explanation="The order is already cancelled.")
    if status != OrderStatus.PLACED or shipped:
        return CancellationDecision(
            eligible=False,
            explanation=(
                "The order has already shipped, so it can no longer be cancelled. "
                "Once it arrives you can return it under our refund policy."
            ),
        )
    return CancellationDecision(eligible=True, explanation="The order has not shipped yet.")


def evaluate_cancellation_for_state(order: Mapping[str, Any]) -> CancellationDecision:
    """Same rule for the order summary dict kept in the ticket state."""
    shipment = order.get("shipment") or {}
    return evaluate_cancellation(str(order["status"]), bool(shipment.get("shipped_at")))


class CancellationWorkflow:
    def __init__(self, session: Session):
        self._session = session

    def cancel(
        self, order_number: str, customer_email: str, approver: str, now: datetime | None = None
    ) -> Order:
        """Cancel the order after a human approved it. The caller commits.

        Safe to repeat: an already-cancelled order is returned unchanged (a retry after a crash).
        Raises NotFoundError for a wrong number/email (same error for both) and WorkflowError if
        the order can no longer be cancelled.
        """
        order = ensure_owner(
            OrderRepository(self._session).get_by_number(order_number, for_update=True),
            customer_email,
        )
        if order.status is OrderStatus.CANCELLED:
            return order
        shipped = bool(order.shipment and order.shipment.shipped_at)
        decision = evaluate_cancellation(order.status.value, shipped)
        if not decision.eligible:
            raise WorkflowError(
                f"Order {order.order_number} can no longer be cancelled: {decision.explanation}"
            )
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = now or datetime.now(UTC)
        order.cancelled_by = approver
        self._session.flush()
        logger.info("Order %s cancelled, approved by %s", order.order_number, approver)
        return order
