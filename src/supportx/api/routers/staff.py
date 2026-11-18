"""Staff endpoints: the human-approval step. Every route needs the X-API-Key header."""

from fastapi import APIRouter, Request

from supportx.agents.state import HumanDecision
from supportx.agents.tickets import TicketService
from supportx.api.deps import SessionDep, StaffDep, TicketsDep, WorkflowDep
from supportx.api.schemas import (
    ApproveIn,
    RefundRequestOut,
    RejectIn,
    StaffTicketOut,
    TicketDecisionIn,
)
from supportx.core import NotFoundError
from supportx.db.repositories import RefundRepository

router = APIRouter(prefix="/staff", tags=["staff"], dependencies=[StaffDep])


def _ticket_service(request: Request) -> TicketService | None:
    return getattr(request.app.state, "tickets", None)


@router.get("/refunds/pending", response_model=list[RefundRequestOut])
def list_pending(session: SessionDep) -> list[RefundRequestOut]:
    return [RefundRequestOut.from_model(r) for r in RefundRepository(session).list_pending()]


def _decide_refund(
    request: Request,
    session: SessionDep,
    workflow: WorkflowDep,
    refund_id: int,
    decision: HumanDecision,
) -> RefundRequestOut:
    """Decide a refund. When the ticket workflow is on, a paused ticket waiting for this refund is
    resumed too, so one staff action can never leave the database and the ticket out of step."""
    service = _ticket_service(request)
    if service is not None:
        service.decide_refund(refund_id, decision)
        session.expire_all()  # the service used its own sessions; read the committed result
        saved = RefundRepository(session).get(refund_id)
        if saved is None:
            raise NotFoundError(f"Refund request {refund_id} does not exist.")
        return RefundRequestOut.from_model(saved)
    if decision.approved:
        updated = workflow.approve(refund_id, decision.approver, decision.note)
    else:
        updated = workflow.reject(refund_id, decision.approver, decision.note or "")
    session.commit()
    return RefundRequestOut.from_model(updated)


@router.post("/refunds/{refund_id}/approve", response_model=RefundRequestOut)
def approve(
    refund_id: int, body: ApproveIn, request: Request, session: SessionDep, workflow: WorkflowDep
) -> RefundRequestOut:
    decision = HumanDecision(approved=True, approver=body.approver, note=body.note)
    return _decide_refund(request, session, workflow, refund_id, decision)


@router.post("/refunds/{refund_id}/reject", response_model=RefundRequestOut)
def reject(
    refund_id: int, body: RejectIn, request: Request, session: SessionDep, workflow: WorkflowDep
) -> RefundRequestOut:
    decision = HumanDecision(approved=False, approver=body.approver, note=body.note)
    return _decide_refund(request, session, workflow, refund_id, decision)


@router.get("/tickets/awaiting", response_model=list[StaffTicketOut])
def list_awaiting_tickets(tickets: TicketsDep) -> list[StaffTicketOut]:
    return [StaffTicketOut.from_view(v) for v in tickets.list_awaiting()]


@router.post("/tickets/{ticket_id}/decision", response_model=StaffTicketOut)
def decide_ticket(ticket_id: str, body: TicketDecisionIn, tickets: TicketsDep) -> StaffTicketOut:
    """Approve or reject a paused ticket (refunds are also recorded on the refund request)."""
    decision = HumanDecision(approved=body.approved, approver=body.approver, note=body.note)
    return StaffTicketOut.from_view(tickets.decide_ticket(ticket_id, decision))
