"""Request and response bodies. Pydantic validates every input before our code sees it."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from supportx.db.enums import RefundReason, RefundStatus
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


class ApproveIn(_Strict):
    approver: str = Field(min_length=1, max_length=120)
    note: str | None = Field(default=None, max_length=2000)


class RejectIn(_Strict):
    approver: str = Field(min_length=1, max_length=120)
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
