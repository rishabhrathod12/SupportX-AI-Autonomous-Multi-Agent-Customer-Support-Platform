"""Customer-facing endpoints. POST with a JSON body so emails never appear in URLs (and logs)."""

from fastapi import APIRouter, Response

from supportx.api.deps import SessionDep, WorkflowDep
from supportx.api.schemas import (
    OrderLookupIn,
    RefundRequestIn,
    RefundRequestOut,
    RefundRequestResult,
)
from supportx.services import OrderService, OrderSummary
from supportx.services.refund import RefundDecision

router = APIRouter(tags=["customer"])


@router.post("/orders/lookup", response_model=OrderSummary)
def lookup_order(body: OrderLookupIn, session: SessionDep) -> OrderSummary:
    """Order status for the order's owner. A wrong email looks exactly like a missing order."""
    return OrderService(session).get_order_summary(body.order_number, body.customer_email)


@router.post("/refunds/eligibility", response_model=RefundDecision)
def check_eligibility(body: RefundRequestIn, workflow: WorkflowDep) -> RefundDecision:
    """Would a refund be allowed? Creates nothing."""
    return workflow.check_eligibility(
        body.order_number, body.customer_email, body.reason, body.requested_cents
    )


@router.post("/refunds/requests", response_model=RefundRequestResult)
def request_refund(
    body: RefundRequestIn,
    response: Response,
    session: SessionDep,
    workflow: WorkflowDep,
) -> RefundRequestResult:
    """Ask for a refund. If the rules allow it, a PENDING request awaits a human decision."""
    decision, request = workflow.request_refund(
        body.order_number, body.customer_email, body.reason, body.requested_cents
    )
    if request is None:
        return RefundRequestResult(created=False, decision=decision)
    session.commit()  # saved before the client sees success
    response.status_code = 201
    return RefundRequestResult(
        created=True, decision=decision, request=RefundRequestOut.from_model(request)
    )
