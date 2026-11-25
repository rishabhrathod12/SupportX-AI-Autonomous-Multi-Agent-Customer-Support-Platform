"""Ticket triage: priority, responsible team and expected resolution time.

Always available: transparent rules. Optionally: trained models (`supportx.ml.triage_models`)
behind the same `Predictor` interface. A model answer is used only when it is confident; an
uncertain or failing model falls back to the rules and the ticket is flagged `needs_review` so a
person looks at it.

The rule tables are DEMO defaults chosen by the developer, not confirmed business policy.
Resolution time has NO rule (inventing hours would be a fabricated number): it is None until a
model trained on real resolution data supplies it.

Only information known when the ticket arrives is used (message, intent, sentiment, urgency,
the approval kind). Nothing that is only known afterwards (outcome, closing time, status).
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from supportx.core import get_logger

logger = get_logger(__name__)

PRIORITIES = ("low", "normal", "high", "urgent")  # lowest -> highest
TEAMS = ("refunds", "orders", "customer_care", "general")

_REFUND_INTENTS = {"get_refund", "track_refund", "check_refund_policy"}
_ORDER_INTENTS = {
    "track_order",
    "cancel_order",
    "delivery_options",
    "delivery_period",
    "check_cancellation_fee",
    "place_order",
    "change_order",
}
_CARE_INTENTS = {"complaint", "contact_human_agent", "contact_customer_service"}


@dataclass(frozen=True)
class Triage:
    priority: str
    team: str
    expected_resolution_hours: float | None
    source: str  # "rules" | "model" | "rules+model"
    needs_review: bool  # a model was unsure or failed: a person should check this triage


# A trained classifier: ticket facts -> (label, confidence)
Predictor = Callable[[Mapping[str, Any]], tuple[str, float]]


def rule_priority(facts: Mapping[str, Any]) -> str:
    if facts.get("escalation_reason") == "urgent_negative":
        return "urgent"
    negative = facts.get("sentiment") == "negative"
    if negative and facts.get("urgency") == "high":
        return "urgent"
    if negative or facts.get("intent") in _CARE_INTENTS or facts.get("urgency") == "high":
        return "high"
    if facts.get("approval_kind") in ("refund", "cancellation"):
        return "normal"
    return "low" if facts.get("intent") else "normal"  # unknown intent: not low by default


def rule_team(facts: Mapping[str, Any]) -> str:
    intent = facts.get("intent")
    if facts.get("approval_kind") == "refund" or intent in _REFUND_INTENTS:
        return "refunds"
    if intent in _CARE_INTENTS:
        return "customer_care"
    if facts.get("approval_kind") == "cancellation" or intent in _ORDER_INTENTS:
        return "orders"
    return "general"


def ticket_facts(state: Mapping[str, Any], approval_kind: str | None = None) -> dict[str, Any]:
    """The arrival-time facts used for triage (no outcome fields)."""
    return {
        "message": str(state.get("message", "")),
        "intent": state.get("intent"),
        "sentiment": state.get("sentiment"),
        "urgency": state.get("urgency"),
        "escalation_reason": state.get("escalation_reason"),
        "approval_kind": approval_kind,
    }


def make_triager(
    priority_model: Predictor | None = None,
    team_model: Predictor | None = None,
    resolution_model: Callable[[Mapping[str, Any]], float] | None = None,
    min_confidence: float = 0.7,
) -> Callable[[Mapping[str, Any]], Triage]:
    """Build the triage function. With no models it is pure rules."""

    def triage(facts: Mapping[str, Any]) -> Triage:
        needs_review = False
        used_model = False

        def pick(model: Predictor | None, rule: str, allowed: tuple[str, ...]) -> str:
            nonlocal needs_review, used_model
            if model is None:
                return rule
            try:
                label, confidence = model(facts)
            except Exception:  # a broken model must never break ticket handling
                logger.exception("Triage model failed; using rules")
                needs_review = True
                return rule
            if label in allowed and confidence >= min_confidence:
                used_model = True
                return label
            needs_review = True
            return rule

        priority = pick(priority_model, rule_priority(facts), PRIORITIES)
        team = pick(team_model, rule_team(facts), TEAMS)
        hours: float | None = None
        if resolution_model is not None:
            try:
                value = float(resolution_model(facts))
                hours = round(value, 1) if value > 0 else None
            except Exception:
                logger.exception("Resolution-time model failed")
        source = ("rules+model" if needs_review else "model") if used_model else "rules"
        return Triage(priority, team, hours, source, needs_review)

    return triage


triage_with_rules = make_triager()


def priority_rank(priority: str | None) -> int:
    """Higher = more urgent (unknown sorts as normal)."""
    return PRIORITIES.index(priority) if priority in PRIORITIES else PRIORITIES.index("normal")
