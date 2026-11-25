"""Shared state for the ticket workflow.

Everything in the state is plain JSON-friendly data (dicts, lists, strings, numbers), because a
checkpointer must be able to save a paused workflow to storage and bring it back.
"""

import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TicketState(TypedDict, total=False):
    # --- input ---
    ticket_id: str
    message: str
    customer_email: str | None
    order_number: str | None

    # --- filled in by the agents ---
    intent: str
    intent_confidence: float
    sentiment: str  # negative | neutral | positive
    urgency: str  # low | high
    passages: list[dict[str, Any]]  # policy passages (may be empty = nothing relevant found)
    order: dict[str, Any]
    order_not_found: bool
    refund: dict[str, Any]  # the rule engine's recommendation (+ request id if one was created)
    approval: dict[str, Any]  # the human's decision
    missing_info: list[str]
    knowledge_fallback: bool  # answered from policy text although the intent was uncertain

    # --- outcome ---
    response: str
    escalated: bool
    escalation_reason: str

    # --- bookkeeping ---
    next: str
    steps: int
    errors: Annotated[list[str], operator.add]  # appended to, never overwritten
    trace: Annotated[list[str], operator.add]  # audit trail of what ran, in order


class HumanDecision(BaseModel):
    """What a person tells the paused workflow when they resume it."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    approved: bool
    approver: str = Field(min_length=1, max_length=120)
    note: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _rejection_needs_a_reason(self) -> "HumanDecision":
        if not self.approved and not self.note:
            raise ValueError("A reason (note) is required when rejecting.")
        return self
