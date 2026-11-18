import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
from pydantic import ValidationError

from supportx.agents import AgentSet, HumanDecision, build_graph, resume_ticket, start_ticket
from supportx.agents.replies import HANDOFF
from supportx.agents.stubs import make_stub_agents
from supportx.core import ExternalServiceError
from supportx.rag import Retriever

POLICIES = Path(__file__).resolve().parents[2] / "data" / "sample" / "policies"
EMAIL = "sam@example.com"
ORDERS: dict[str, dict[str, Any]] = {
    "ORD-1001": {
        "email": EMAIL,
        "status": "delivered",
        "shipment": {"tracking_number": "T1"},
        "refund_eligible": True,
        "refundable_cents": 4999,
    },
    "ORD-2002": {
        "email": EMAIL,
        "status": "delivered",
        "shipment": {"tracking_number": "T2"},
        "refund_eligible": False,
    },
    "ORD-3003": {"email": EMAIL, "status": "placed", "shipment": None},
}
FAST_RETRY = RetryPolicy(max_attempts=3, initial_interval=0.01, retry_on=ExternalServiceError)


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever.from_documents(POLICIES)


@pytest.fixture
def agents(retriever: Retriever) -> AgentSet:
    return make_stub_agents(retriever, ORDERS)


def make_graph(agents: AgentSet) -> CompiledStateGraph[Any, Any, Any, Any]:
    return build_graph(agents, InMemorySaver(), retry=FAST_RETRY)


@pytest.fixture
def graph(agents: AgentSet) -> CompiledStateGraph[Any, Any, Any, Any]:
    return make_graph(agents)


def paused_at(graph: CompiledStateGraph[Any, Any, Any, Any], ticket_id: str) -> tuple[str, ...]:
    return tuple(graph.get_state({"configurable": {"thread_id": ticket_id}}).next)


# ---------------------------------------------------------------- straight-through tickets
def test_a_policy_question_is_answered_from_the_documents(
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    outcome = start_ticket(graph, "t1", "What is your return policy?")
    assert outcome.status == "completed" and not outcome.state.get("escalated")
    top = outcome.state["passages"][0]
    assert top["source"] == "refund_policy.md"
    assert top["source"] in outcome.state["response"] and top["text"] in outcome.state["response"]
    assert outcome.state["trace"] == [
        "supervisor->query_understanding", "query_understanding",
        "supervisor->sentiment", "sentiment",
        "supervisor->knowledge", "knowledge",
        "supervisor->respond", "respond",
        "supervisor->__end__",
    ]  # fmt: skip


def test_replies_only_quote_retrieved_passages_even_when_retrieval_is_ambiguous(
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    """For this phrasing the lexical retriever's top scores are nearly tied (0.143 / 0.128 / 0.126)
    and the best-ranked passage is not the most relevant one. That is a retrieval limitation, not a
    workflow bug; the workflow guarantees that the reply is built from retrieved text only."""
    question = "What is your return policy? How many days do I have?"
    outcome = start_ticket(graph, "t1b", question)
    assert outcome.status == "completed"
    passages = outcome.state["passages"]
    assert 1 <= len(passages) <= 2
    reply = outcome.state["response"]
    assert reply.startswith(
        f"According to our {passages[0]['source']} ({passages[0]['heading']}): "
        f"{passages[0]['text']}"
    )
    assert all(p["text"] in reply for p in passages)  # built from retrieved text only
    assert reply.count("Also relevant") == len(passages) - 1


def test_order_tracking_uses_only_order_facts(
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    outcome = start_ticket(graph, "t2", "Where is my order ORD-1001?", EMAIL, "ORD-1001")
    assert outcome.status == "completed"
    assert (
        outcome.state["response"] == "Order ORD-1001 is currently delivered. Tracking number: T1."
    )


def test_unknown_order_or_wrong_email_gets_a_generic_answer(
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    missing = start_ticket(graph, "t3", "Where is my order ORD-9999?", EMAIL, "ORD-9999")
    wrong = start_ticket(graph, "t4", "Where is my order ORD-1001?", "x@y.com", "ORD-1001")
    assert missing.state["response"] == wrong.state["response"]  # cannot tell which exists
    assert "couldn't find an order" in missing.state["response"]


def test_missing_details_are_requested(graph: CompiledStateGraph[Any, Any, Any, Any]) -> None:
    outcome = start_ticket(graph, "t5", "Where is my order?")
    assert outcome.state["response"] == (
        "To look into this I need your order number and email. Could you please send it?"
    )
    assert "order" not in outcome.state  # nothing was looked up


# ---------------------------------------------------------------- human approval
def test_refund_pauses_for_a_human_and_resumes_with_the_decision(
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    first = start_ticket(graph, "r1", "My item arrived damaged, I want a refund", EMAIL, "ORD-1001")
    assert first.status == "awaiting_approval"
    assert first.approval_request == {
        "kind": "refund",
        "ticket_id": "r1",
        "order_number": "ORD-1001",
        "amount_cents": 4999,
        "refund_request_id": None,
    }
    assert "response" not in first.state  # nothing is promised before a person decides
    assert paused_at(graph, "r1") == ("human_approval",)

    done = resume_ticket(
        graph, "r1", HumanDecision(approved=True, approver="agent.jane", note="Photo ok")
    )
    assert done.status == "completed"
    assert done.state["response"] == "Good news: our team approved your refund request of 49.99."
    assert done.state["approval"]["approver"] == "agent.jane"
    assert paused_at(graph, "r1") == ()
    assert done.state["trace"].count("human_approval(approved)") == 1


def test_a_rejected_refund_tells_the_customer_the_reason(
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    start_ticket(graph, "r2", "I want a refund please", EMAIL, "ORD-1001")
    done = resume_ticket(
        graph, "r2", {"approved": False, "approver": "agent.bob", "note": "No photo"}
    )
    assert done.state["response"] == (
        "Our team reviewed your refund request and could not approve it. Reason: No photo"
    )


def test_a_bad_decision_is_refused_and_the_ticket_stays_paused(
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    start_ticket(graph, "r3", "I want a refund please", EMAIL, "ORD-1001")
    for bad in (
        {"approved": False, "approver": "agent.bob"},  # rejection without a reason
        {"approved": True, "approver": "  "},  # nobody identified
        {"approved": True, "approver": "bob", "extra": 1},  # unknown field
    ):
        with pytest.raises(ValidationError):
            resume_ticket(graph, "r3", bad)
    assert paused_at(graph, "r3") == ("human_approval",)  # still waiting, unharmed
    assert (
        resume_ticket(graph, "r3", HumanDecision(approved=True, approver="bob")).status
        == "completed"
    )


def test_an_ineligible_refund_is_explained_with_no_human_step(
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    outcome = start_ticket(graph, "r4", "I want a refund for this", EMAIL, "ORD-2002")
    assert outcome.status == "completed"
    assert "isn't eligible for a refund" in outcome.state["response"]
    assert "The 30-day return window has passed." in outcome.state["response"]
    assert not any("human_approval" in step for step in outcome.state["trace"])


def test_a_cancellation_also_needs_a_human(graph: CompiledStateGraph[Any, Any, Any, Any]) -> None:
    first = start_ticket(graph, "c1", "Please cancel my order ORD-3003", EMAIL, "ORD-3003")
    assert first.status == "awaiting_approval"
    assert first.approval_request is not None
    assert first.approval_request["kind"] == "cancellation"
    assert first.approval_request["amount_cents"] is None
    done = resume_ticket(graph, "c1", HumanDecision(approved=True, approver="agent.jane"))
    assert done.state["response"] == "Good news: our team approved your cancellation request."


def test_a_shipped_order_cannot_be_cancelled_and_no_human_is_asked(
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    outcome = start_ticket(graph, "c2", "Please cancel my order ORD-1001", EMAIL, "ORD-1001")
    assert outcome.status == "completed"
    assert "can't be cancelled" in outcome.state["response"]
    assert "already shipped" in outcome.state["response"]
    assert not any("human_approval" in step for step in outcome.state["trace"])


def test_tickets_are_independent_and_resumable_in_any_order(
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    start_ticket(graph, "A", "I want a refund please", EMAIL, "ORD-1001")  # paused
    other = start_ticket(graph, "B", "What is your return policy?")
    assert other.status == "completed"
    assert paused_at(graph, "A") == ("human_approval",) and paused_at(graph, "B") == ()
    assert (
        resume_ticket(graph, "A", HumanDecision(approved=True, approver="jane")).status
        == "completed"
    )


# ---------------------------------------------------------------- escalation to a human
@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("This is a complaint about your service", "complaint"),
        ("This is unacceptable! I need my refund immediately", "urgent_negative"),
        ("blah blah blah", "low_confidence"),
        ("I want to speak to a real person", "customer_requested_human"),
    ],
)
def test_escalations_hand_over_to_a_person(
    graph: CompiledStateGraph[Any, Any, Any, Any], message: str, reason: str
) -> None:
    outcome = start_ticket(graph, f"e-{reason}", message, EMAIL, "ORD-1001")
    assert outcome.status == "completed"
    assert outcome.state["escalated"] is True and outcome.state["escalation_reason"] == reason
    assert outcome.state["response"] == HANDOFF
    assert (
        "order" not in outcome.state and "approval" not in outcome.state
    )  # nothing automated happened


def test_no_matching_policy_means_a_handoff_not_an_invented_answer(
    agents: AgentSet, retriever: Retriever
) -> None:
    graph = make_graph(replace(agents, search_policy=lambda q: []))
    outcome = start_ticket(graph, "n1", "What is your return policy?")
    assert outcome.state["escalation_reason"] == "no_policy_match"
    assert outcome.state["response"] == HANDOFF


def test_the_step_limit_stops_runaway_tickets(
    agents: AgentSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("supportx.agents.supervisor.MAX_STEPS", 2)
    outcome = start_ticket(make_graph(agents), "s1", "What is your return policy?")
    assert outcome.state["escalation_reason"] == "step_limit"
    assert outcome.state["response"] == HANDOFF


# ---------------------------------------------------------------- retries and failures
def test_a_flaky_agent_is_retried_until_it_works(agents: AgentSet) -> None:
    calls = {"n": 0}
    original = agents.search_policy

    def flaky(question: str) -> list[dict[str, Any]]:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ExternalServiceError("temporary outage")
        return original(question)

    outcome = start_ticket(
        make_graph(replace(agents, search_policy=flaky)), "f1", "What is your return policy?"
    )
    assert calls["n"] == 3
    assert outcome.state["response"].startswith("According to our refund_policy.md")
    assert not outcome.state.get("escalated") and outcome.state["errors"] == []


def test_an_agent_that_keeps_failing_degrades_to_a_human_handoff(
    agents: AgentSet, caplog: pytest.LogCaptureFixture
) -> None:
    calls = {"n": 0}

    def down(order_number: str, email: str) -> dict[str, Any]:
        calls["n"] += 1
        raise ExternalServiceError("database is down: password=hunter2")

    graph = make_graph(replace(agents, lookup_order=down))
    with caplog.at_level(logging.ERROR):
        outcome = start_ticket(graph, "f2", "Where is my order ORD-1001?", EMAIL, "ORD-1001")
    assert calls["n"] == 3  # tried the maximum number of times
    assert outcome.state["escalation_reason"] == "order_unavailable"
    assert outcome.state["response"] == HANDOFF
    assert outcome.state["errors"] == ["order failed: ExternalServiceError"]
    assert "hunter2" not in str(outcome.state)  # internal details never reach the state or customer
    assert any("Agent 'order' failed" in r.getMessage() for r in caplog.records)


def test_an_unexpected_bug_is_not_retried_and_does_not_leak(agents: AgentSet) -> None:
    calls = {"n": 0}

    def buggy(order_number: str, email: str) -> dict[str, Any]:
        calls["n"] += 1
        raise RuntimeError("secret internal detail")

    outcome = start_ticket(
        make_graph(replace(agents, lookup_order=buggy)),
        "f3",
        "Where is my order ORD-1001?",
        EMAIL,
        "ORD-1001",
    )
    assert calls["n"] == 1  # only transient errors are retried
    assert outcome.state["escalation_reason"] == "order_unavailable"
    assert "secret" not in outcome.state["response"] and "secret" not in str(
        outcome.state["errors"]
    )


def test_a_failing_response_agent_falls_back_to_the_handoff_text(agents: AgentSet) -> None:
    def broken(state: Any) -> str:
        raise RuntimeError("template crashed")

    outcome = start_ticket(
        make_graph(replace(agents, compose_reply=broken)), "f4", "What is your return policy?"
    )
    assert outcome.status == "completed"
    assert outcome.state["response"] == HANDOFF and outcome.state["escalated"] is True
