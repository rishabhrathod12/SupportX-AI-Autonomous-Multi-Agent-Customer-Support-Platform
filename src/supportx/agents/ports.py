"""The interface the graph needs from its agents.

The graph only knows these function signatures. Stub versions power the tests and the demo;
real ones (ML intent model, RAG retriever, database services, LLM) are plugged in later
without touching the graph.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from supportx.agents.state import TicketState


@dataclass(frozen=True)
class AgentSet:
    # Query Understanding Agent: message -> (intent, confidence 0..1)
    understand: Callable[[str], tuple[str, float]]
    # Sentiment Agent: message -> (sentiment, urgency)
    analyze_sentiment: Callable[[str], tuple[str, str]]
    # Knowledge Agent: question -> policy passages [{"source", "heading", "text", "score"}]
    search_policy: Callable[[str], list[dict[str, Any]]]
    # Order Management Agent: (order_number, customer_email) -> order facts; raises NotFoundError
    lookup_order: Callable[[str, str], dict[str, Any]]
    # Refund Agent: (order_number, customer_email, message) -> rule-engine recommendation
    check_refund: Callable[[str, str, str], dict[str, Any]]
    # Response Agent: final state -> reply text (must only use facts present in the state)
    compose_reply: Callable[[TicketState], str]
