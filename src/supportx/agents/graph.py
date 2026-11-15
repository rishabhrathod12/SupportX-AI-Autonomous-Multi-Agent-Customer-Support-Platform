"""The ticket workflow as a LangGraph graph (hub and spoke around the Supervisor).

    START -> supervisor -> (agent) -> supervisor -> ... -> respond -> supervisor -> END

Every agent reports back to the Supervisor, which decides the next step from the state.
A refund or cancellation pauses at `human_approval` until a person resumes the ticket.
"""

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, RetryPolicy, interrupt

from supportx.agents.ports import AgentSet
from supportx.agents.replies import HANDOFF
from supportx.agents.state import HumanDecision, TicketState
from supportx.agents.supervisor import (
    HUMAN_APPROVAL,
    KNOWLEDGE,
    MIN_CONFIDENCE,
    ORDER,
    REFUND,
    RESPOND,
    ROUTES,
    SENTIMENT,
    UNDERSTAND,
    decide_next,
)
from supportx.core import ExternalServiceError, NotFoundError, get_logger

logger = get_logger(__name__)

DEFAULT_RETRY = RetryPolicy(max_attempts=3, initial_interval=0.5, retry_on=ExternalServiceError)


def _failure_handler(fallback_response: str | None = None):  # type: ignore[no-untyped-def]
    """Build an error handler: when a node still fails after its retries, hand over to a human.

    The customer only ever sees the generic handoff message; the real exception goes to the logs.
    """

    def handle(state: TicketState, error: NodeError) -> Command[Any]:
        logger.error(
            "Agent '%s' failed (ticket %s): %s",
            error.node,
            state.get("ticket_id"),
            type(error.error).__name__,
            exc_info=error.error,
        )
        update: dict[str, Any] = {
            "escalated": True,
            "escalation_reason": f"{error.node}_unavailable",
            "errors": [f"{error.node} failed: {type(error.error).__name__}"],
            "trace": [f"{error.node}(failed)"],
        }
        if fallback_response is not None:
            update["response"] = fallback_response
        # Without goto the run would simply END here, leaving the customer without a reply.
        return Command(update=update, goto="supervisor")

    return handle


def build_graph(
    agents: AgentSet,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    retry: RetryPolicy | None = None,
    min_confidence: float = MIN_CONFIDENCE,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Assemble the workflow. Pass a checkpointer to allow pausing for human approval."""
    retry = retry or DEFAULT_RETRY

    def supervisor(state: TicketState) -> dict[str, Any]:
        decision = decide_next(state, min_confidence)
        return {
            **decision.update,
            "next": decision.next,
            "steps": state.get("steps", 0) + 1,
            "trace": [f"supervisor->{decision.next}"],
        }

    def query_understanding(state: TicketState) -> dict[str, Any]:
        intent, confidence = agents.understand(state["message"])
        return {"intent": intent, "intent_confidence": confidence, "trace": [UNDERSTAND]}

    def sentiment(state: TicketState) -> dict[str, Any]:
        label, urgency = agents.analyze_sentiment(state["message"])
        return {"sentiment": label, "urgency": urgency, "trace": [SENTIMENT]}

    def knowledge(state: TicketState) -> dict[str, Any]:
        return {"passages": agents.search_policy(state["message"]), "trace": [KNOWLEDGE]}

    def order(state: TicketState) -> dict[str, Any]:
        try:
            found = agents.lookup_order(state["order_number"] or "", state["customer_email"] or "")
        except NotFoundError:
            return {"order_not_found": True, "trace": [f"{ORDER}(not_found)"]}
        return {"order": found, "trace": [ORDER]}

    def refund(state: TicketState) -> dict[str, Any]:
        try:
            result = agents.check_refund(
                state["order_number"] or "", state["customer_email"] or "", state["message"]
            )
        except NotFoundError:
            return {"order_not_found": True, "trace": [f"{REFUND}(not_found)"]}
        return {"refund": result, "trace": [REFUND]}

    def human_approval(state: TicketState) -> dict[str, Any]:
        # NOTE: when the ticket is resumed this whole function runs again from the top, so
        # nothing before interrupt() may have side effects.
        route = ROUTES.get(state.get("intent", ""))
        kind = (route.approval_kind if route else None) or "refund"
        refund_info = state.get("refund", {})
        raw = interrupt(
            {
                "kind": kind,
                "ticket_id": state.get("ticket_id"),
                "order_number": state.get("order_number"),
                "amount_cents": refund_info.get("proposed_cents") if kind == "refund" else None,
                "refund_request_id": refund_info.get("request_id"),
            }
        )
        decision = HumanDecision.model_validate(raw)
        outcome = "approved" if decision.approved else "rejected"
        return {"approval": decision.model_dump(), "trace": [f"{HUMAN_APPROVAL}({outcome})"]}

    def respond(state: TicketState) -> dict[str, Any]:
        return {"response": agents.compose_reply(state), "trace": [RESPOND]}

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(TicketState)
    graph.add_node("supervisor", supervisor)
    for name, func, handler in (
        (UNDERSTAND, query_understanding, _failure_handler()),
        (SENTIMENT, sentiment, _failure_handler()),
        (KNOWLEDGE, knowledge, _failure_handler()),
        (ORDER, order, _failure_handler()),
        (REFUND, refund, _failure_handler()),
        (RESPOND, respond, _failure_handler(fallback_response=HANDOFF)),
    ):
        graph.add_node(name, func, retry_policy=retry, error_handler=handler)
    graph.add_node(HUMAN_APPROVAL, human_approval)  # no retry: it pauses instead of failing

    graph.add_edge(START, "supervisor")
    destinations = [UNDERSTAND, SENTIMENT, KNOWLEDGE, ORDER, REFUND, HUMAN_APPROVAL, RESPOND]
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {**{d: d for d in destinations}, "__end__": END},
    )
    for name in destinations:
        graph.add_edge(name, "supervisor")
    return graph.compile(checkpointer=checkpointer)


@dataclass(frozen=True)
class TicketOutcome:
    status: Literal["completed", "awaiting_approval"]
    state: dict[str, Any]
    approval_request: dict[str, Any] | None = None  # what the human is being asked to decide


def _config(ticket_id: str) -> RunnableConfig:
    # thread_id = ticket id: each ticket has its own saved, resumable conversation
    return RunnableConfig(configurable={"thread_id": ticket_id}, recursion_limit=60)


def _outcome(
    graph: CompiledStateGraph[Any, Any, Any, Any], ticket_id: str, result: Any
) -> TicketOutcome:
    state = dict(graph.get_state(_config(ticket_id)).values)
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if interrupts:
        return TicketOutcome("awaiting_approval", state, dict(interrupts[0].value))
    return TicketOutcome("completed", state)


def start_ticket(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    ticket_id: str,
    message: str,
    customer_email: str | None = None,
    order_number: str | None = None,
) -> TicketOutcome:
    """Run a new ticket until it finishes or needs a human decision."""
    initial: TicketState = {
        "ticket_id": ticket_id,
        "message": message,
        "customer_email": customer_email,
        "order_number": order_number,
        "steps": 0,
        "errors": [],
        "trace": [],
    }
    return _outcome(graph, ticket_id, graph.invoke(initial, _config(ticket_id)))


def resume_ticket(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    ticket_id: str,
    decision: HumanDecision | dict[str, Any],
) -> TicketOutcome:
    """Give a paused ticket its human decision and run it to the end.

    The decision is validated BEFORE touching the graph, so a bad decision leaves the ticket paused.
    """
    validated = (
        decision if isinstance(decision, HumanDecision) else HumanDecision.model_validate(decision)
    )
    result = graph.invoke(Command(resume=validated.model_dump()), _config(ticket_id))
    return _outcome(graph, ticket_id, result)
