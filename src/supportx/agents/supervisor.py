"""The Supervisor's brain: a pure function that looks at the state and picks the next step.

Rules, not an LLM, so every routing decision is deterministic and unit-testable.
(An LLM fallback for ambiguous cases can be added later without changing this contract.)
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

MIN_CONFIDENCE = 0.4  # below this we do not trust the intent and hand over to a human
MAX_STEPS = 12  # loop guard: a ticket that needs more supervisor turns is escalated

END = "__end__"
UNDERSTAND, SENTIMENT, KNOWLEDGE, ORDER, REFUND = (
    "query_understanding",
    "sentiment",
    "knowledge",
    "order",
    "refund",
)
HUMAN_APPROVAL, RESPOND = "human_approval", "respond"


@dataclass(frozen=True)
class Route:
    """What an intent needs before we can answer."""

    needs_order: bool = False
    needs_knowledge: bool = False
    needs_refund_check: bool = False
    approval_kind: str | None = None  # "refund" / "cancellation": a person must decide
    escalate: str | None = None  # hand straight to a human, with this reason


ROUTES: dict[str, Route] = {
    "track_order": Route(needs_order=True),
    "track_refund": Route(needs_order=True),
    "get_refund": Route(needs_order=True, needs_refund_check=True, approval_kind="refund"),
    "cancel_order": Route(needs_order=True, approval_kind="cancellation"),
    "check_refund_policy": Route(needs_knowledge=True),
    "check_cancellation_fee": Route(needs_knowledge=True),
    "delivery_options": Route(needs_knowledge=True),
    "delivery_period": Route(needs_knowledge=True),
    "complaint": Route(escalate="complaint"),
    "contact_human_agent": Route(escalate="customer_requested_human"),
    "contact_customer_service": Route(escalate="customer_requested_human"),
}
DEFAULT_ROUTE = Route(needs_knowledge=True)  # unknown intents: try the policy documents


@dataclass(frozen=True)
class Decision:
    next: str
    update: dict[str, Any] = field(default_factory=dict)  # extra state changes to record


def _escalate(reason: str) -> Decision:
    return Decision(RESPOND, {"escalated": True, "escalation_reason": reason})


def decide_next(state: Mapping[str, Any]) -> Decision:
    """Pick the next step for a ticket. Called after every agent finishes."""
    if state.get("response"):
        return Decision(END)
    if state.get("steps", 0) >= MAX_STEPS:
        return _escalate("step_limit")
    if state.get("escalated"):  # set by an earlier rule or by a failed agent
        return Decision(RESPOND)

    if "intent" not in state:
        return Decision(UNDERSTAND)
    if "sentiment" not in state:
        return Decision(SENTIMENT)

    if state.get("intent_confidence", 0.0) < MIN_CONFIDENCE:
        return _escalate("low_confidence")
    if state.get("sentiment") == "negative" and state.get("urgency") == "high":
        return _escalate("urgent_negative")

    route = ROUTES.get(state["intent"], DEFAULT_ROUTE)
    if route.escalate:
        return _escalate(route.escalate)

    if route.needs_order:
        if "order" not in state and not state.get("order_not_found"):
            missing = [
                name
                for name, key in (("order number", "order_number"), ("email", "customer_email"))
                if not state.get(key)
            ]
            if missing:
                return Decision(RESPOND, {"missing_info": missing})
            return Decision(ORDER)
        if state.get("order_not_found"):
            return Decision(RESPOND)

    if route.needs_knowledge:
        if "passages" not in state:
            return Decision(KNOWLEDGE)
        if not state["passages"]:
            return _escalate("no_policy_match")  # never answer from nothing

    if route.needs_refund_check:
        if "refund" not in state:
            return Decision(REFUND)
        if not state["refund"].get("eligible"):
            return Decision(RESPOND)  # explain why; no human decision needed

    if route.approval_kind and "approval" not in state:
        return Decision(HUMAN_APPROVAL)

    return Decision(RESPOND)
