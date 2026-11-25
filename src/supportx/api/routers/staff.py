"""Staff endpoints: the human-approval step. Every route needs the X-API-Key header."""

import json
from typing import Any

from fastapi import APIRouter, Request

from supportx.agents.state import HumanDecision
from supportx.agents.tickets import TicketService
from supportx.api.deps import (
    SessionDep,
    StaffDep,
    StaffIdentityDep,
    TicketsDep,
    WorkflowDep,
    resolve_approver,
)
from supportx.api.schemas import (
    ApproveIn,
    AuditOut,
    OutboxOut,
    RefundRequestOut,
    RejectIn,
    StaffTicketOut,
    TicketDecisionIn,
)
from supportx.core import NotFoundError
from supportx.db import OutboxEvent, OutboxStatus
from supportx.db.repositories import RefundRepository, to_utc
from supportx.integrations.outbox import list_events, outbox_stats, requeue
from supportx.services import list_audit

router = APIRouter(prefix="/staff", tags=["staff"], dependencies=[StaffDep])


def _ticket_service(request: Request) -> TicketService | None:
    return getattr(request.app.state, "tickets", None)


@router.get("/model-info")
def model_info(request: Request) -> dict[str, object]:
    """Which intent model file and settings are serving (for audits and incident debugging)."""
    service = _ticket_service(request)
    settings = request.app.state.settings
    return {
        "app_version": request.app.version,
        "app_env": settings.app_env,
        "intent_min_confidence": settings.intent_min_confidence,
        "integrations_mode": settings.integrations_mode,
        "business_rules": {
            "version": settings.business_rules_version,
            "refund_return_window_days": settings.refund_return_window_days,
            "refund_not_received_after_days": settings.refund_not_received_after_days,
            "confirmed_as_real_policy": settings.business_rules_confirmed,
        },
        "intent_model": service.model_info if service is not None else None,
    }


def _outbox_out(row: OutboxEvent) -> OutboxOut:
    return OutboxOut(
        id=row.id,
        event_type=row.event_type,
        idempotency_key=row.idempotency_key,
        status=row.status.value,
        attempts=row.attempts,
        created_at=to_utc(row.created_at) or row.created_at,
        next_attempt_at=to_utc(row.next_attempt_at) or row.next_attempt_at,
        processed_at=to_utc(row.processed_at),
        last_error=row.last_error,
        provider_reference=row.provider_reference,
        is_mock=row.is_mock,
    )


@router.get("/outbox/stats")
def outbox_status(session: SessionDep) -> dict[str, Any]:
    """Counts per status and the oldest waiting event: for monitoring and alerts."""
    return outbox_stats(session)


@router.get("/outbox", response_model=list[OutboxOut])
def outbox_events(
    session: SessionDep, status: OutboxStatus | None = None, limit: int = 50
) -> list[OutboxOut]:
    """Follow-up work: refund payments, stock releases, customer messages. `dead` needs a person."""
    return [_outbox_out(r) for r in list_events(session, status, limit)]


@router.post("/outbox/{event_id}/retry", response_model=OutboxOut)
def retry_outbox_event(event_id: int, session: SessionDep, staff: StaffIdentityDep) -> OutboxOut:
    """Requeue a DEAD event after the cause was fixed."""
    row = requeue(session, event_id, resolve_approver(staff, "shared-key"))
    session.commit()
    return _outbox_out(row)


@router.get("/audit", response_model=list[AuditOut])
def audit_trail(
    session: SessionDep, limit: int = 50, action: str | None = None, target_id: str | None = None
) -> list[AuditOut]:
    """Newest first. Filter by `action` (e.g. order.cancelled) or `target_id`."""
    return [
        AuditOut(
            id=r.id,
            occurred_at=to_utc(r.occurred_at) or r.occurred_at,
            actor=r.actor,
            action=r.action,
            target_type=r.target_type,
            target_id=r.target_id,
            detail=json.loads(r.detail) if r.detail else None,
        )
        for r in list_audit(session, limit, action, target_id)
    ]


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
    refund_id: int,
    body: ApproveIn,
    request: Request,
    session: SessionDep,
    workflow: WorkflowDep,
    staff: StaffIdentityDep,
) -> RefundRequestOut:
    approver = resolve_approver(staff, body.approver)
    decision = HumanDecision(approved=True, approver=approver, note=body.note)
    return _decide_refund(request, session, workflow, refund_id, decision)


@router.post("/refunds/{refund_id}/reject", response_model=RefundRequestOut)
def reject(
    refund_id: int,
    body: RejectIn,
    request: Request,
    session: SessionDep,
    workflow: WorkflowDep,
    staff: StaffIdentityDep,
) -> RefundRequestOut:
    approver = resolve_approver(staff, body.approver)
    decision = HumanDecision(approved=False, approver=approver, note=body.note)
    return _decide_refund(request, session, workflow, refund_id, decision)


@router.get("/tickets/awaiting", response_model=list[StaffTicketOut])
def list_awaiting_tickets(tickets: TicketsDep) -> list[StaffTicketOut]:
    return [StaffTicketOut.from_view(v) for v in tickets.list_awaiting()]


@router.post("/tickets/{ticket_id}/decision", response_model=StaffTicketOut)
def decide_ticket(
    ticket_id: str, body: TicketDecisionIn, tickets: TicketsDep, staff: StaffIdentityDep
) -> StaffTicketOut:
    """Approve or reject a paused ticket (refunds are also recorded on the refund request)."""
    approver = resolve_approver(staff, body.approver)
    decision = HumanDecision(approved=body.approved, approver=approver, note=body.note)
    return StaffTicketOut.from_view(tickets.decide_ticket(ticket_id, decision))
