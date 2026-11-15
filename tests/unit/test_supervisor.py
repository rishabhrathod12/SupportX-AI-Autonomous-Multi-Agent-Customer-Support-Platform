from typing import Any

import pytest

from supportx.agents.supervisor import (
    END,
    HUMAN_APPROVAL,
    KNOWLEDGE,
    MAX_STEPS,
    ORDER,
    REFUND,
    RESPOND,
    SENTIMENT,
    UNDERSTAND,
    decide_next,
)

PASSAGE = {
    "source": "refund_policy.md",
    "heading": "Refund Policy",
    "text": "30 days.",
    "score": 0.5,
}


def state(**extra: Any) -> dict[str, Any]:
    """A ticket that is already understood and calm; tests override what they care about."""
    base = {
        "intent": "track_order",
        "intent_confidence": 0.9,
        "sentiment": "neutral",
        "urgency": "low",
        "order_number": "ORD-1",
        "customer_email": "a@b.com",
    }
    return {**base, **extra}


def test_a_fresh_ticket_is_understood_then_sentiment_checked() -> None:
    assert decide_next({}).next == UNDERSTAND
    assert decide_next({"intent": "track_order"}).next == SENTIMENT


def test_a_finished_ticket_ends() -> None:
    assert decide_next(state(response="done")).next == END
    # even a runaway ticket that already has its answer just ends
    assert decide_next(state(response="done", steps=MAX_STEPS + 5)).next == END


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"intent_confidence": 0.1}, "low_confidence"),
        ({"sentiment": "negative", "urgency": "high"}, "urgent_negative"),
        ({"intent": "complaint"}, "complaint"),
        ({"intent": "contact_human_agent"}, "customer_requested_human"),
        ({"intent": "check_refund_policy", "passages": []}, "no_policy_match"),
        ({"steps": MAX_STEPS}, "step_limit"),
    ],
)
def test_escalation_triggers(overrides: dict[str, Any], reason: str) -> None:
    decision = decide_next(state(**overrides))
    assert decision.next == RESPOND
    assert decision.update == {"escalated": True, "escalation_reason": reason}


def test_the_confidence_threshold_defaults_to_0_6_and_can_be_changed() -> None:
    borderline = state(intent_confidence=0.5)
    assert decide_next(borderline).update["escalation_reason"] == "low_confidence"  # default 0.6
    assert decide_next(borderline, min_confidence=0.4).next == ORDER
    assert decide_next(state(intent_confidence=0.6)).next == ORDER  # exactly at the limit passes
    assert decide_next(state(intent_confidence=0.7), min_confidence=0.9).update == {
        "escalated": True,
        "escalation_reason": "low_confidence",
    }


def test_negative_but_not_urgent_is_still_handled_automatically() -> None:
    assert decide_next(state(sentiment="negative", urgency="low")).next == ORDER


def test_an_escalated_ticket_goes_straight_to_the_reply() -> None:
    assert decide_next(state(escalated=True, escalation_reason="order_unavailable")).next == RESPOND


def test_order_lookup_flow() -> None:
    assert decide_next(state()).next == ORDER
    assert decide_next(state(order={"order_number": "ORD-1"})).next == RESPOND
    assert decide_next(state(order_not_found=True)).next == RESPOND


@pytest.mark.parametrize(
    ("overrides", "missing"),
    [
        ({"order_number": None}, ["order number"]),
        ({"customer_email": None}, ["email"]),
        ({"order_number": "", "customer_email": ""}, ["order number", "email"]),
    ],
)
def test_missing_details_are_asked_for_not_guessed(
    overrides: dict[str, Any], missing: list[str]
) -> None:
    decision = decide_next(state(**overrides))
    assert decision.next == RESPOND and decision.update == {"missing_info": missing}


def test_policy_questions_use_the_knowledge_agent() -> None:
    assert decide_next(state(intent="check_refund_policy")).next == KNOWLEDGE
    assert decide_next(state(intent="check_refund_policy", passages=[PASSAGE])).next == RESPOND
    assert decide_next(state(intent="never_seen_before")).next == KNOWLEDGE  # unknown intent


def test_refund_flow_ends_with_a_human_decision() -> None:
    s = state(intent="get_refund")
    assert decide_next(s).next == ORDER
    s["order"] = {"order_number": "ORD-1"}
    assert decide_next(s).next == REFUND
    s["refund"] = {"eligible": True, "proposed_cents": 500}
    assert decide_next(s).next == HUMAN_APPROVAL
    s["approval"] = {"approved": True, "approver": "jane"}
    assert decide_next(s).next == RESPOND


def test_an_ineligible_refund_is_explained_without_bothering_a_human() -> None:
    s = state(intent="get_refund", order={"order_number": "ORD-1"}, refund={"eligible": False})
    assert decide_next(s).next == RESPOND


def test_cancellation_needs_a_human_but_no_refund_check() -> None:
    s = state(intent="cancel_order")
    assert decide_next(s).next == ORDER
    s["order"] = {"order_number": "ORD-1"}
    assert decide_next(s).next == HUMAN_APPROVAL


def test_decisions_do_not_modify_the_state() -> None:
    s = state(intent="complaint")
    before = dict(s)
    decide_next(s)
    assert s == before
