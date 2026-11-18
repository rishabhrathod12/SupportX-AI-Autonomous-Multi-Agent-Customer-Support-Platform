"""Tickets: run the workflow for a customer message and apply human decisions to paused tickets.

One place owns the two-step "decision": (1) record it on the refund request in the database,
(2) resume the paused ticket. Step 1 is idempotent, so if the process dies between the steps a
retry is safe: the already-recorded decision is recognised and only step 2 runs.
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine

from supportx.agents.adapters import build_real_agents
from supportx.agents.graph import TicketOutcome, build_graph, resume_ticket, start_ticket
from supportx.agents.replies import HANDOFF
from supportx.agents.state import HumanDecision
from supportx.agents.stubs import extract_order_number
from supportx.core import (
    ConfigurationError,
    ExternalServiceError,
    NotFoundError,
    Settings,
    WorkflowError,
    get_logger,
)
from supportx.db import RefundStatus, Ticket, TicketStatus, session_scope
from supportx.db.repositories import RefundRepository, TicketRepository, to_utc
from supportx.ml import IntentClassifier
from supportx.rag import Retriever
from supportx.services import RefundPolicy, RefundWorkflow

logger = get_logger(__name__)
Graph = CompiledStateGraph[Any, Any, Any, Any]
NOT_FOUND = "No ticket was found with those details."


class TicketView(BaseModel):
    """A snapshot of a ticket (plain data, safe to use after the database session is closed)."""

    model_config = ConfigDict(frozen=True)

    ticket_id: str
    status: TicketStatus
    response: str | None
    escalated: bool
    escalation_reason: str | None
    approval_kind: str | None
    refund_request_id: int | None
    amount_cents: int | None
    created_at: datetime


def _view(ticket: Ticket) -> TicketView:
    refund = ticket.refund_request
    return TicketView(
        ticket_id=ticket.id,
        status=ticket.status,
        response=ticket.response,
        escalated=ticket.escalated,
        escalation_reason=ticket.escalation_reason,
        approval_kind=ticket.approval_kind,
        refund_request_id=ticket.refund_request_id,
        amount_cents=refund.amount_cents if refund else None,
        created_at=to_utc(ticket.created_at) or ticket.created_at,
    )


class TicketService:
    def __init__(self, engine: Engine, graph: Graph):
        self._engine = engine
        self._graph = graph

    # ------------------------------------------------------------------ customer side
    def submit(
        self, message: str, customer_email: str | None = None, order_number: str | None = None
    ) -> TicketView:
        """Create a ticket and run it until it has a reply or needs a human decision."""
        order_number = order_number or extract_order_number(message)
        ticket_id = str(uuid4())
        with session_scope(self._engine) as session:
            TicketRepository(session).add(
                Ticket(
                    id=ticket_id,
                    customer_email=customer_email,
                    order_number=order_number,
                    message=message,
                    status=TicketStatus.OPEN,
                )
            )
        try:
            outcome = start_ticket(self._graph, ticket_id, message, customer_email, order_number)
        except Exception:
            # Agent failures are already handled inside the graph; reaching here means the
            # workflow machinery itself failed (e.g. the checkpoint store). Never show the customer
            # an error page: hand over to a person and keep the details in the logs.
            logger.exception("Workflow failed for ticket %s", ticket_id)
            return self._record_failure(ticket_id)
        return self._record(ticket_id, outcome)

    def get_for_customer(self, ticket_id: str, customer_email: str) -> TicketView:
        """Status of a ticket, only for its owner (a wrong email looks like a missing ticket)."""
        with session_scope(self._engine) as session:
            ticket = TicketRepository(session).get(ticket_id)
            owner = ticket.customer_email if ticket else None
            if ticket is None or not owner or owner.casefold() != customer_email.strip().casefold():
                raise NotFoundError(NOT_FOUND)
            return _view(ticket)

    # ------------------------------------------------------------------ staff side
    def list_awaiting(self) -> list[TicketView]:
        with session_scope(self._engine) as session:
            return [_view(t) for t in TicketRepository(session).list_awaiting()]

    def decide_ticket(self, ticket_id: str, decision: HumanDecision) -> TicketView:
        """A person decides a paused ticket. Refund decisions are also recorded on the request."""
        with session_scope(self._engine) as session:
            ticket = TicketRepository(session).get(ticket_id)
            if ticket is None:
                raise NotFoundError(NOT_FOUND)
            if ticket.status is not TicketStatus.AWAITING_APPROVAL:
                raise WorkflowError(f"Ticket {ticket_id} is not waiting for a decision.")
            refund_id = ticket.refund_request_id if ticket.approval_kind == "refund" else None
        if refund_id is not None:
            self._record_refund_decision(refund_id, decision)  # step 1 (idempotent)
        return self._resume(ticket_id, decision)  # step 2

    def decide_refund(self, refund_id: int, decision: HumanDecision) -> TicketView | None:
        """Decide a refund request; if a paused ticket is waiting on it, resume that ticket too."""
        self._record_refund_decision(refund_id, decision)  # step 1 (idempotent)
        with session_scope(self._engine) as session:
            ticket = TicketRepository(session).find_awaiting_by_refund(refund_id)
            ticket_id = ticket.id if ticket else None
        return self._resume(ticket_id, decision) if ticket_id else None  # step 2

    # ------------------------------------------------------------------ internals
    def _record_refund_decision(self, refund_id: int, decision: HumanDecision) -> None:
        target = RefundStatus.APPROVED if decision.approved else RefundStatus.REJECTED
        with session_scope(self._engine) as session:
            request = RefundRepository(session).get(refund_id, for_update=True)
            if request is None:
                raise NotFoundError(f"Refund request {refund_id} does not exist.")
            if request.status is target:
                return  # already recorded (a retry after a crash): nothing to do
            workflow = RefundWorkflow(session)
            if decision.approved:
                workflow.approve(refund_id, decision.approver, decision.note)
            elif request.status is RefundStatus.PENDING_APPROVAL:
                workflow.reject(refund_id, decision.approver, decision.note or "")
            else:
                raise WorkflowError(
                    f"Refund request {refund_id} was already {request.status.value}; "
                    f"it cannot be {target.value}."
                )

    def _resume(self, ticket_id: str, decision: HumanDecision) -> TicketView:
        try:
            outcome = resume_ticket(self._graph, ticket_id, decision)
        except Exception as exc:
            logger.exception("Could not resume ticket %s", ticket_id)
            raise ExternalServiceError(
                "The decision was recorded, but the ticket could not be resumed. "
                "It is safe to submit the same decision again."
            ) from exc
        return self._record(ticket_id, outcome)

    def _record(self, ticket_id: str, outcome: TicketOutcome) -> TicketView:
        state = outcome.state
        with session_scope(self._engine) as session:
            ticket = TicketRepository(session).get(ticket_id, for_update=True)
            if ticket is None:  # cannot happen: submit() created it first
                raise NotFoundError(NOT_FOUND)
            request = outcome.approval_request or {}
            ticket.status = (
                TicketStatus.AWAITING_APPROVAL
                if outcome.status == "awaiting_approval"
                else TicketStatus.COMPLETED
            )
            ticket.response = state.get("response")
            ticket.escalated = bool(state.get("escalated", False))
            ticket.escalation_reason = state.get("escalation_reason")
            if request:
                ticket.approval_kind = request.get("kind")
                ticket.refund_request_id = request.get("refund_request_id")
            ticket.updated_at = datetime.now(UTC)
            session.flush()
            return _view(ticket)

    def _record_failure(self, ticket_id: str) -> TicketView:
        with session_scope(self._engine) as session:
            ticket = TicketRepository(session).get(ticket_id, for_update=True)
            if ticket is None:
                raise NotFoundError(NOT_FOUND)
            ticket.status = TicketStatus.COMPLETED
            ticket.escalated = True
            ticket.escalation_reason = "workflow_error"
            ticket.response = HANDOFF
            ticket.updated_at = datetime.now(UTC)
            session.flush()
            return _view(ticket)


def build_ticket_service(
    settings: Settings, engine: Engine, policy: RefundPolicy | None = None
) -> tuple[TicketService, sqlite3.Connection]:
    """Assemble the production ticket workflow from settings.

    Returns the service and the checkpoint connection (the caller closes it on shutdown).
    Raises ConfigurationError with a clear message when the intent model or documents are missing.
    """
    model_path = Path(settings.intent_model_path)
    if not model_path.exists():
        raise ConfigurationError(
            f"Intent model not found at {model_path}. Train it first: "
            "python -m supportx.ml.train_intent bitext"
        )
    docs_dir = Path(settings.policy_docs_dir)
    if not docs_dir.is_dir():
        raise ConfigurationError(f"Policy documents folder not found: {docs_dir}")

    intent_model = IntentClassifier.load(model_path)  # pickle: only load files you created
    retriever = Retriever.from_documents(docs_dir)
    connection = sqlite3.connect(settings.checkpoint_db_path, check_same_thread=False)
    saver = SqliteSaver(connection)
    agents = build_real_agents(intent_model, retriever, engine, policy)
    graph = build_graph(agents, saver, min_confidence=settings.intent_min_confidence)
    return TicketService(engine, graph), connection
