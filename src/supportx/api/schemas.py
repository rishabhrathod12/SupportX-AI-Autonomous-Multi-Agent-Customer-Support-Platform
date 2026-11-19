"""Request and response bodies. Pydantic validates every input before our code sees it."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from supportx.agents.tickets import TicketView
from supportx.db.enums import RefundReason, RefundStatus, TicketStatus
from supportx.db.models import RefundRequest
from supportx.db.repositories import to_utc
from supportx.services.refund import RefundDecision


class _Strict(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class OrderLookupIn(_Strict):
    order_number: str = Field(min_length=1, max_length=40)
    customer_email: str = Field(min_length=3, max_length=255)

    @field_validator("customer_email")
    @classmethod
    def _looks_like_an_email(cls, value: str) -> str:
        if "@" not in value:
            raise ValueError("customer_email must contain '@'")
        return value


class RefundRequestIn(OrderLookupIn):
    reason: RefundReason
    requested_cents: int | None = Field(default=None, gt=0)  # omit to ask for the maximum


class TicketCreateIn(_Strict):
    message: str = Field(min_length=1, max_length=2000)
    customer_email: str | None = Field(default=None, min_length=3, max_length=255)
    order_number: str | None = Field(default=None, min_length=1, max_length=40)

    @field_validator("customer_email")
    @classmethod
    def _looks_like_an_email(cls, value: str | None) -> str | None:
        if value is not None and "@" not in value:
            raise ValueError("customer_email must contain '@'")
        return value


class TicketStatusIn(_Strict):
    ticket_id: str = Field(min_length=1, max_length=36)
    customer_email: str = Field(min_length=3, max_length=255)


class CustomerTicketOut(BaseModel):
    """What the customer may see: no internal ids, reasons or approval details."""

    ticket_id: str
    status: TicketStatus
    message: str  # the reply, or a short note while the ticket waits for a person

    @classmethod
    def from_view(cls, view: TicketView) -> "CustomerTicketOut":
        if view.status is TicketStatus.AWAITING_APPROVAL:
            text = "Your request is with our support team for review. We'll update you here."
        elif view.status is TicketStatus.OPEN:
            text = "We're still working on your request."
        else:
            text = view.response or "A member of our team will follow up."
        return cls(ticket_id=view.ticket_id, status=view.status, message=text)


class StaffTicketOut(BaseModel):
    ticket_id: str
    status: TicketStatus
    approval_kind: str | None
    refund_request_id: int | None
    amount_cents: int | None
    escalated: bool
    escalation_reason: str | None
    response: str | None
    created_at: datetime

    @classmethod
    def from_view(cls, view: TicketView) -> "StaffTicketOut":
        return cls(**view.model_dump())


class TicketDecisionIn(_Strict):
    approved: bool
    approver: str | None = Field(default=None, min_length=1, max_length=120)
    note: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _rejection_needs_a_reason(self) -> "TicketDecisionIn":
        if not self.approved and not self.note:
            raise ValueError("A reason (note) is required when rejecting.")
        return self


class ApproveIn(_Strict):
    approver: str | None = Field(default=None, min_length=1, max_length=120)
    note: str | None = Field(default=None, max_length=2000)


class RejectIn(_Strict):
    approver: str | None = Field(default=None, min_length=1, max_length=120)
    note: str = Field(min_length=1, max_length=2000)  # a reason is mandatory


class RefundRequestOut(BaseModel):
    id: int
    order_number: str
    amount_cents: int
    reason: RefundReason
    status: RefundStatus
    created_at: datetime
    decided_at: datetime | None
    decided_by: str | None
    decision_note: str | None

    @classmethod
    def from_model(cls, request: RefundRequest) -> "RefundRequestOut":
        return cls(
            id=request.id,
            order_number=request.order.order_number,
            amount_cents=request.amount_cents,
            reason=request.reason,
            status=request.status,
            created_at=to_utc(request.created_at) or request.created_at,
            decided_at=to_utc(request.decided_at),
            decided_by=request.decided_by,
            decision_note=request.decision_note,
        )


class RefundRequestResult(BaseModel):
    created: bool
    decision: RefundDecision
    request: RefundRequestOut | None = None


class HealthOut(BaseModel):
    status: str


class AuditOut(BaseModel):
    id: int
    occurred_at: datetime
    actor: str
    action: str
    target_type: str
    target_id: str
    detail: dict[str, Any] | None
