"""Multi-agent workflow (LangGraph): state, Supervisor, agent interface, graph."""

from supportx.agents.adapters import build_real_agents, infer_refund_reason
from supportx.agents.graph import TicketOutcome, build_graph, resume_ticket, start_ticket
from supportx.agents.ports import AgentSet
from supportx.agents.state import HumanDecision, TicketState
from supportx.agents.supervisor import Decision, decide_next

__all__ = [
    "AgentSet",
    "Decision",
    "HumanDecision",
    "TicketOutcome",
    "TicketState",
    "build_graph",
    "build_real_agents",
    "decide_next",
    "infer_refund_reason",
    "resume_ticket",
    "start_ticket",
]
