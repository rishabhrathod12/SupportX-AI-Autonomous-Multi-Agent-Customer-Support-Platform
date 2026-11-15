"""Staff endpoints: the human-approval step. Every route needs the X-API-Key header."""

from fastapi import APIRouter

from supportx.api.deps import SessionDep, StaffDep, WorkflowDep
from supportx.api.schemas import ApproveIn, RefundRequestOut, RejectIn
from supportx.db.repositories import RefundRepository

router = APIRouter(prefix="/staff", tags=["staff"], dependencies=[StaffDep])


@router.get("/refunds/pending", response_model=list[RefundRequestOut])
def list_pending(session: SessionDep) -> list[RefundRequestOut]:
    return [RefundRequestOut.from_model(r) for r in RefundRepository(session).list_pending()]


@router.post("/refunds/{refund_id}/approve", response_model=RefundRequestOut)
def approve(
    refund_id: int, body: ApproveIn, session: SessionDep, workflow: WorkflowDep
) -> RefundRequestOut:
    request = workflow.approve(refund_id, body.approver, body.note)
    session.commit()
    return RefundRequestOut.from_model(request)


@router.post("/refunds/{refund_id}/reject", response_model=RefundRequestOut)
def reject(
    refund_id: int, body: RejectIn, session: SessionDep, workflow: WorkflowDep
) -> RefundRequestOut:
    request = workflow.reject(refund_id, body.approver, body.note)
    session.commit()
    return RefundRequestOut.from_model(request)
