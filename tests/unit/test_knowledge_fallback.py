"""Knowledge fallback: a low-confidence INFORMATIONAL message is answered from the policy text
instead of escalating, but never for refunds, cancellations, orders, upset customers, other
intents or weak matches."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from supportx.agents import TicketService
from supportx.agents.adapters import build_real_agents
from supportx.agents.graph import build_graph
from supportx.agents.replies import compose_reply
from supportx.agents.supervisor import KNOWLEDGE, RESPOND, decide_next
from supportx.rag import Retriever
from tests.unit.helpers import EMAIL, POLICIES, make_engine

FALLBACK = 0.2
PASSAGE_OK = {"source": "s.md", "heading": "H", "text": "t", "score": 0.4}
PASSAGE_WEAK = {**PASSAGE_OK, "score": 0.18}


def low(**extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "intent": "check_refund_policy",
        "intent_confidence": 0.3,
        "sentiment": "neutral",
        "urgency": "low",
    }
    return {**base, **extra}


def test_fallback_is_off_by_default() -> None:
    d = decide_next(low())
    assert d.next == RESPOND and d.update["escalation_reason"] == "low_confidence"


def test_an_informational_guess_is_searched_then_answered_if_the_match_is_strong() -> None:
    first = decide_next(low(), fallback_min_score=FALLBACK)
    assert first.next == KNOWLEDGE and first.update == {"knowledge_fallback": True}
    answered = decide_next(low(passages=[PASSAGE_OK]), fallback_min_score=FALLBACK)
    assert answered.next == RESPOND and "escalated" not in answered.update


@pytest.mark.parametrize("passages", [[], [PASSAGE_WEAK]])
def test_no_or_weak_match_still_escalates(passages: list[dict[str, Any]]) -> None:
    d = decide_next(low(passages=passages), fallback_min_score=FALLBACK)
    assert d.update.get("escalation_reason") == "low_confidence"


@pytest.mark.parametrize(
    "overrides",
    [
        {"intent": "get_refund"},  # the guess is a refund request
        {"intent": "cancel_order"},
        {"intent": "track_order"},
        {"intent": "complaint"},
        {"intent": "payment_issue"},  # unknown to the routes: must not default to policy text
        {"intent": "create_account"},
        {"intent": "contact_human_agent"},
        {"order_number": "ORD-1001"},  # about a specific order
        {"sentiment": "negative"},
        {"urgency": "high"},
    ],
)
def test_sensitive_cases_still_escalate_without_searching(overrides: dict[str, Any]) -> None:
    d = decide_next(low(**overrides), fallback_min_score=FALLBACK)
    assert d.next == RESPOND and d.update["escalation_reason"] == "low_confidence"


def test_a_confident_message_is_unaffected() -> None:
    d = decide_next(low(intent_confidence=0.9), fallback_min_score=FALLBACK)
    assert d.next == KNOWLEDGE and not d.update  # normal knowledge route, not the fallback


def test_the_reply_says_it_was_unsure_and_offers_a_person() -> None:
    state = {"message": "m", "passages": [PASSAGE_OK], "knowledge_fallback": True}
    reply = compose_reply(state)  # type: ignore[arg-type]
    assert "wasn't completely sure" in reply and "member of our team" in reply
    assert "wasn't completely sure" not in compose_reply({"message": "m", "passages": [PASSAGE_OK]})


class UnsureModel:
    """Guesses an informational intent with low confidence; a refund guess when refunds come up."""

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        if "refund" in text.lower():
            return ("get_refund", 0.3)
        return ("check_refund_policy", 0.3)


def test_end_to_end_the_fallback_answers_policy_questions_and_not_refund_requests(
    tmp_path: Path,
) -> None:
    connection = sqlite3.connect(tmp_path / "c.db", check_same_thread=False)
    try:
        engine = make_engine()
        retriever = Retriever.from_documents(POLICIES, expand_queries=True)
        agents = build_real_agents(UnsureModel(), retriever, engine)
        graph = build_graph(agents, SqliteSaver(connection), knowledge_fallback_min_score=FALLBACK)
        service = TicketService(engine, graph)
        answered = service.submit("The mug came smashed. What do I do?", EMAIL)
        assert not answered.escalated and "photo" in (answered.response or "").lower()
        assert "wasn't completely sure" in (answered.response or "")
        refund = service.submit("I want a refund for my mug please", EMAIL)
        assert refund.escalated and refund.escalation_reason == "low_confidence"
        nonsense = service.submit("asdf qwerty zxcv", EMAIL)
        assert nonsense.escalated  # no strong match: a person takes over
    finally:
        connection.close()
