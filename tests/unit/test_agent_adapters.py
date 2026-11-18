from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.pool import StaticPool

from supportx.agents import (
    HumanDecision,
    build_graph,
    build_real_agents,
    infer_refund_reason,
    resume_ticket,
    start_ticket,
)
from supportx.agents.adapters import make_database_agents
from supportx.agents.replies import HANDOFF
from supportx.core import ExternalServiceError, Settings
from supportx.data.sample import generate_sample_tickets
from supportx.db import (
    Customer,
    Order,
    OrderItem,
    OrderStatus,
    RefundReason,
    RefundRequest,
    RefundStatus,
    Shipment,
    init_db,
    session_scope,
)
from supportx.ml import IntentClassifier
from supportx.rag import Retriever
from supportx.services import RefundWorkflow

POLICIES = Path(__file__).resolve().parents[2] / "data" / "sample" / "policies"
EMAIL = "sam@example.com"
FAST_RETRY = RetryPolicy(max_attempts=3, initial_interval=0.01, retry_on=ExternalServiceError)
Graph = CompiledStateGraph[Any, Any, Any, Any]


class FakeIntentModel:
    """A deterministic stand-in so graph routing does not depend on a trained model's quirks."""

    RULES = (
        ("refund", ("get_refund", 0.95)),
        ("cancel", ("cancel_order", 0.95)),
        ("where is my order", ("track_order", 0.9)),
        ("return policy", ("check_refund_policy", 0.9)),
        ("maybe", ("track_order", 0.5)),
    )

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        for keyword, answer in self.RULES:
            if keyword in text.lower():
                return answer
        return ("unknown", 0.1)


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever.from_documents(POLICIES)


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    init_db(eng)
    now = datetime.now(UTC)
    with session_scope(eng) as session:
        customer = Customer(email=EMAIL, name="Sam")
        for number, days in (("ORD-1001", 5), ("ORD-2002", 40)):
            session.add(
                Order(
                    order_number=number,
                    customer=customer,
                    status=OrderStatus.DELIVERED,
                    items=[OrderItem(sku="M", name="Mug", unit_price_cents=2500, quantity=2)],
                    shipment=Shipment(
                        carrier="DemoPost",
                        tracking_number=f"TRK-{number}",
                        shipped_at=now - timedelta(days=days + 3),
                        delivered_at=now - timedelta(days=days),
                    ),
                )
            )
        session.add(  # placed, not shipped: the only kind of order that can be cancelled
            Order(
                order_number="ORD-3003",
                customer=customer,
                status=OrderStatus.PLACED,
                items=[OrderItem(sku="M", name="Mug", unit_price_cents=2500, quantity=1)],
            )
        )
    yield eng
    eng.dispose()


def make_graph(engine: Engine, retriever: Retriever, **kwargs: Any) -> Graph:
    agents = build_real_agents(FakeIntentModel(), retriever, engine)
    return build_graph(agents, InMemorySaver(), retry=FAST_RETRY, **kwargs)


def refund_rows(engine: Engine) -> list[RefundRequest]:
    with session_scope(engine) as session:
        return list(session.scalars(select(RefundRequest)))


# ---------------------------------------------------------------- small pieces
@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ("My mug arrived damaged", RefundReason.DAMAGED),
        ("the handle is broken and cracked", RefundReason.DAMAGED),
        ("It's not as described on the page", RefundReason.NOT_AS_DESCRIBED),
        ("You sent the wrong size", RefundReason.NOT_AS_DESCRIBED),
        ("My parcel has not arrived", RefundReason.NOT_RECEIVED),
        ("it never arrived", RefundReason.NOT_RECEIVED),
        ("I want a refund", RefundReason.CHANGED_MIND),
        ("", RefundReason.CHANGED_MIND),
    ],
)
def test_refund_reason_is_inferred_from_the_words(message: str, reason: RefundReason) -> None:
    assert infer_refund_reason(message) is reason


def test_the_real_intent_model_plugs_into_the_interface(
    engine: Engine, retriever: Retriever
) -> None:
    tickets = generate_sample_tickets(200, seed=3)  # SYNTHETIC: proves wiring only
    model = IntentClassifier().fit(tickets["text"], tickets["intent"])
    agents = build_real_agents(model, retriever, engine)
    intent, confidence = agents.understand("Where is my order ORD-12345? Thanks.")
    assert intent in model.classes_ and 0.0 < confidence <= 1.0


def test_the_confidence_threshold_is_a_setting() -> None:
    assert Settings(_env_file=None).intent_min_confidence == 0.6  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        Settings(_env_file=None, intent_min_confidence=1.5)  # type: ignore[call-arg]


# ---------------------------------------------------------------- real order lookup
def test_tracking_reads_the_real_database(engine: Engine, retriever: Retriever) -> None:
    graph = make_graph(engine, retriever)
    outcome = start_ticket(graph, "t1", "Where is my order ORD-1001?", EMAIL, "ORD-1001")
    assert outcome.state["response"] == (
        "Order ORD-1001 is currently delivered. Tracking number: TRK-ORD-1001."
    )


def test_wrong_email_gets_the_generic_answer_and_creates_nothing(
    engine: Engine, retriever: Retriever
) -> None:
    graph = make_graph(engine, retriever)
    outcome = start_ticket(graph, "t2", "I want a refund", "intruder@example.com", "ORD-1001")
    assert "couldn't find an order" in outcome.state["response"]
    assert refund_rows(engine) == []


def test_policy_questions_use_the_real_retriever(engine: Engine, retriever: Retriever) -> None:
    outcome = start_ticket(make_graph(engine, retriever), "t3", "What is your return policy?")
    assert outcome.state["response"].startswith("According to our refund_policy.md")


# ---------------------------------------------------------------- refund: database + human + graph
def test_refund_creates_a_pending_request_then_a_human_decides(
    engine: Engine, retriever: Retriever
) -> None:
    graph = make_graph(engine, retriever)
    first = start_ticket(graph, "r1", "My mug arrived damaged, I want a refund", EMAIL, "ORD-1001")

    # the paused ticket points at a real pending request in the database
    assert first.status == "awaiting_approval"
    rows = refund_rows(engine)
    assert len(rows) == 1
    request = rows[0]
    assert request.status is RefundStatus.PENDING_APPROVAL
    assert request.reason is RefundReason.DAMAGED and request.amount_cents == 5000
    assert first.approval_request is not None
    assert first.approval_request["refund_request_id"] == request.id
    assert first.approval_request["amount_cents"] == 5000
    assert "response" not in first.state  # nothing promised yet

    # staff decide in the database, then the ticket is resumed with the same decision
    with session_scope(engine) as session:
        RefundWorkflow(session).approve(request.id, "agent.jane", "Photo ok")
    done = resume_ticket(
        graph, "r1", HumanDecision(approved=True, approver="agent.jane", note="Photo ok")
    )
    assert done.state["response"] == "Good news: our team approved your refund request of 50.00."
    assert refund_rows(engine)[0].status is RefundStatus.APPROVED


def test_a_second_ticket_for_the_same_order_creates_no_duplicate(
    engine: Engine, retriever: Retriever
) -> None:
    graph = make_graph(engine, retriever)
    start_ticket(graph, "a", "I want a refund", EMAIL, "ORD-1001")
    second = start_ticket(graph, "b", "I want a refund", EMAIL, "ORD-1001")
    assert second.status == "completed"
    assert "already waiting for approval" in second.state["response"]
    assert len(refund_rows(engine)) == 1


def test_an_order_outside_the_return_window_creates_nothing(
    engine: Engine, retriever: Retriever
) -> None:
    outcome = start_ticket(make_graph(engine, retriever), "w", "I want a refund", EMAIL, "ORD-2002")
    assert outcome.status == "completed"
    assert "return window" in outcome.state["response"]
    assert refund_rows(engine) == []


def test_cancellation_pauses_for_a_human_without_touching_refunds(
    engine: Engine, retriever: Retriever
) -> None:
    outcome = start_ticket(
        make_graph(engine, retriever), "c", "Please cancel my order", EMAIL, "ORD-3003"
    )
    assert outcome.status == "awaiting_approval"
    assert (
        outcome.approval_request is not None and outcome.approval_request["kind"] == "cancellation"
    )
    assert refund_rows(engine) == []


# ---------------------------------------------------------------- confidence threshold and failures
def test_a_borderline_intent_is_escalated_with_the_default_threshold(
    engine: Engine, retriever: Retriever
) -> None:
    graph = make_graph(engine, retriever)  # default 0.6; the fake model says 0.5 for "maybe"
    outcome = start_ticket(graph, "m1", "maybe where is it", EMAIL, "ORD-1001")
    assert outcome.state["escalation_reason"] == "low_confidence"
    assert outcome.state["response"] == HANDOFF


def test_a_lower_threshold_lets_the_same_ticket_through(
    engine: Engine, retriever: Retriever
) -> None:
    graph = make_graph(engine, retriever, min_confidence=0.4)
    outcome = start_ticket(graph, "m2", "maybe where is it", EMAIL, "ORD-1001")
    assert not outcome.state.get("escalated")
    assert "Order ORD-1001" in outcome.state["response"]


def test_an_unreachable_database_degrades_to_a_handoff_without_leaking(
    retriever: Retriever,
) -> None:
    broken = create_engine("sqlite:////nonexistent-folder/none.db")
    graph = make_graph(broken, retriever)
    outcome = start_ticket(graph, "d1", "Where is my order ORD-1001?", EMAIL, "ORD-1001")
    assert outcome.state["escalation_reason"] == "order_unavailable"
    assert outcome.state["response"] == HANDOFF
    assert "nonexistent" not in str(outcome.state)


def test_the_refund_agent_turns_database_trouble_into_a_retryable_error() -> None:
    broken = create_engine("sqlite:////nonexistent-folder/none.db")
    _, check_refund = make_database_agents(broken)
    with pytest.raises(ExternalServiceError, match="refund database is unavailable"):
        check_refund("ORD-1001", EMAIL, "I want a refund")


def test_the_order_agent_turns_database_trouble_into_a_retryable_error() -> None:
    broken = create_engine("sqlite:////nonexistent-folder/none.db")
    lookup_order, _ = make_database_agents(broken)
    with pytest.raises(ExternalServiceError, match="order database is unavailable"):
        lookup_order("ORD-1001", EMAIL)


def test_database_count_helper_sanity(engine: Engine) -> None:
    with session_scope(engine) as session:
        assert session.scalar(select(func.count()).select_from(Order)) == 3
