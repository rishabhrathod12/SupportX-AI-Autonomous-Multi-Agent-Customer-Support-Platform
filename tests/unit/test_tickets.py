from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from langgraph.types import Command
from sqlalchemy import Engine, select

from supportx.agents import HumanDecision, TicketService, build_ticket_service
from supportx.agents.replies import HANDOFF
from supportx.core import (
    ConfigurationError,
    ExternalServiceError,
    NotFoundError,
    Settings,
    WorkflowError,
)
from supportx.data.sample import generate_sample_tickets
from supportx.db import (
    Order,
    OrderStatus,
    RefundRequest,
    RefundStatus,
    Shipment,
    Ticket,
    TicketStatus,
    session_scope,
)
from supportx.ml import IntentClassifier
from supportx.rag import Retriever
from tests.unit.helpers import EMAIL, POLICIES, make_engine, make_service, sqlite_saver

APPROVE = HumanDecision(approved=True, approver="agent.jane", note="Photo ok")
REJECT = HumanDecision(approved=False, approver="agent.bob", note="No photo")


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever.from_documents(POLICIES)


@pytest.fixture
def engine() -> Engine:
    return make_engine()


@pytest.fixture
def service(engine: Engine, retriever: Retriever, tmp_path: Path) -> TicketService:
    saver, connection = sqlite_saver(tmp_path / "checkpoints.db")
    yield make_service(engine, retriever, saver)  # type: ignore[misc]
    connection.close()


def refunds(engine: Engine) -> list[RefundRequest]:
    with session_scope(engine) as session:
        return list(session.scalars(select(RefundRequest)))


def ticket_row(engine: Engine, ticket_id: str) -> Ticket:
    with session_scope(engine) as session:
        ticket = session.get(Ticket, ticket_id)
        assert ticket is not None
        return ticket


def refund_ticket(service: TicketService) -> str:
    view = service.submit("My mug arrived damaged, I want a refund", EMAIL, "ORD-1001")
    assert view.status is TicketStatus.AWAITING_APPROVAL
    return view.ticket_id


# ---------------------------------------------------------------- submitting
def test_a_simple_ticket_completes_and_is_recorded(service: TicketService, engine: Engine) -> None:
    view = service.submit("Where is my order ORD-1001?", EMAIL)  # number found in the text
    assert view.status is TicketStatus.COMPLETED and not view.escalated
    assert view.response == "Order ORD-1001 is currently delivered. Tracking number: TRK-ORD-1001."
    row = ticket_row(engine, view.ticket_id)
    assert row.order_number == "ORD-1001" and row.customer_email == EMAIL
    assert row.message == "Where is my order ORD-1001?" and row.response == view.response


def test_an_escalated_ticket_is_recorded_as_such(service: TicketService) -> None:
    view = service.submit("hello there", EMAIL)
    assert view.status is TicketStatus.COMPLETED
    assert view.escalated and view.escalation_reason == "low_confidence"
    assert view.response == HANDOFF


def test_a_refund_ticket_pauses_and_links_to_its_refund_request(
    service: TicketService, engine: Engine
) -> None:
    ticket_id = refund_ticket(service)
    (request,) = refunds(engine)
    awaiting = service.list_awaiting()
    assert [t.ticket_id for t in awaiting] == [ticket_id]
    assert awaiting[0].approval_kind == "refund"
    assert awaiting[0].refund_request_id == request.id and awaiting[0].amount_cents == 5000
    assert awaiting[0].response is None  # nothing promised yet


# ---------------------------------------------------------------- the decision, step by step
def test_approving_a_ticket_updates_the_database_and_replies(
    service: TicketService, engine: Engine
) -> None:
    ticket_id = refund_ticket(service)
    view = service.decide_ticket(ticket_id, APPROVE)
    assert view.status is TicketStatus.COMPLETED
    assert view.response == "Good news: our team approved your refund request of 50.00."
    (request,) = refunds(engine)
    assert request.status is RefundStatus.APPROVED and request.decided_by == "agent.jane"
    assert service.list_awaiting() == []


def test_rejecting_a_ticket_records_the_reason_everywhere(
    service: TicketService, engine: Engine
) -> None:
    ticket_id = refund_ticket(service)
    view = service.decide_ticket(ticket_id, REJECT)
    assert view.response == (
        "Our team reviewed your refund request and could not approve it. Reason: No photo"
    )
    (request,) = refunds(engine)
    assert request.status is RefundStatus.REJECTED and request.decision_note == "No photo"


def test_the_refund_endpoint_path_resumes_the_waiting_ticket_too(
    service: TicketService, engine: Engine
) -> None:
    ticket_id = refund_ticket(service)
    (request,) = refunds(engine)
    resumed = service.decide_refund(request.id, APPROVE)
    assert resumed is not None and resumed.ticket_id == ticket_id
    assert resumed.status is TicketStatus.COMPLETED
    assert ticket_row(engine, ticket_id).response == resumed.response


def test_a_refund_without_a_ticket_is_still_decided(service: TicketService, engine: Engine) -> None:
    from supportx.db import RefundReason
    from supportx.services import RefundWorkflow

    with session_scope(engine) as session:
        _, request = RefundWorkflow(session).request_refund("ORD-1001", EMAIL, RefundReason.DAMAGED)
        assert request is not None
        refund_id = request.id
    assert service.decide_refund(refund_id, APPROVE) is None  # no ticket was waiting
    assert refunds(engine)[0].status is RefundStatus.APPROVED


def test_a_crash_between_the_two_steps_is_safe_to_retry(
    service: TicketService, engine: Engine
) -> None:
    """Step 1 (database) is done but step 2 (resume) never ran: a retry must work."""
    ticket_id = refund_ticket(service)
    (request,) = refunds(engine)
    service._record_refund_decision(request.id, APPROVE)  # the process "dies" here
    assert ticket_row(engine, ticket_id).status is TicketStatus.AWAITING_APPROVAL

    view = service.decide_ticket(ticket_id, HumanDecision(approved=True, approver="agent.jane"))
    assert view.status is TicketStatus.COMPLETED
    (request,) = refunds(engine)
    assert request.status is RefundStatus.APPROVED
    assert request.decision_note == "Photo ok"  # the first, original record was kept


def test_a_conflicting_decision_is_refused_and_the_ticket_stays_paused(
    service: TicketService, engine: Engine
) -> None:
    ticket_id = refund_ticket(service)
    (request,) = refunds(engine)
    service._record_refund_decision(request.id, APPROVE)  # already approved
    with pytest.raises(WorkflowError, match="already approved"):
        service.decide_ticket(ticket_id, REJECT)
    assert ticket_row(engine, ticket_id).status is TicketStatus.AWAITING_APPROVAL


def test_a_failed_resume_says_the_decision_is_saved_and_a_retry_works(
    engine: Engine, retriever: Retriever, tmp_path: Path
) -> None:
    saver, connection = sqlite_saver(tmp_path / "flaky.db")
    real = make_service(engine, retriever, saver)
    ticket_id = refund_ticket(real)

    class FlakyGraph:
        """Wraps the real graph; the first resume fails, like a checkpoint store outage."""

        def __init__(self) -> None:
            self.failed = False

        def __getattr__(self, name: str) -> Any:
            return getattr(real._graph, name)

        def invoke(self, payload: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(payload, Command) and not self.failed:
                self.failed = True
                raise RuntimeError("checkpoint store unreachable")
            return real._graph.invoke(payload, *args, **kwargs)

    flaky = TicketService(engine, FlakyGraph())  # type: ignore[arg-type]
    with pytest.raises(ExternalServiceError, match="decision was recorded"):
        flaky.decide_ticket(ticket_id, APPROVE)
    assert refunds(engine)[0].status is RefundStatus.APPROVED  # step 1 stuck
    assert ticket_row(engine, ticket_id).status is TicketStatus.AWAITING_APPROVAL

    assert flaky.decide_ticket(ticket_id, APPROVE).status is TicketStatus.COMPLETED  # retry
    connection.close()


def test_cancellation_tickets_have_no_refund_to_update(
    service: TicketService, engine: Engine
) -> None:
    view = service.submit("Please cancel my order", EMAIL, "ORD-3003")
    assert view.status is TicketStatus.AWAITING_APPROVAL and view.approval_kind == "cancellation"
    done = service.decide_ticket(view.ticket_id, APPROVE)
    assert done.response == "Good news: our team approved your cancellation request."
    assert refunds(engine) == []


def test_only_paused_tickets_can_be_decided(service: TicketService) -> None:
    done = service.submit("Where is my order ORD-1001?", EMAIL)
    with pytest.raises(WorkflowError, match="not waiting"):
        service.decide_ticket(done.ticket_id, APPROVE)
    with pytest.raises(NotFoundError):
        service.decide_ticket("no-such-ticket", APPROVE)
    with pytest.raises(NotFoundError):
        service.decide_refund(999, APPROVE)


def test_deciding_twice_is_refused(service: TicketService) -> None:
    ticket_id = refund_ticket(service)
    service.decide_ticket(ticket_id, APPROVE)
    with pytest.raises(WorkflowError, match="not waiting"):
        service.decide_ticket(ticket_id, APPROVE)


# ---------------------------------------------------------------- customer view
def test_only_the_owner_can_see_a_ticket(service: TicketService) -> None:
    ticket_id = service.submit("Where is my order ORD-1001?", EMAIL).ticket_id
    assert service.get_for_customer(ticket_id, " SAM@example.com ").ticket_id == ticket_id
    errors = []
    for tid, email in ((ticket_id, "x@y.com"), ("unknown", EMAIL)):
        with pytest.raises(NotFoundError) as caught:
            service.get_for_customer(tid, email)
        errors.append(str(caught.value))
    assert errors[0] == errors[1]  # cannot tell a wrong email from a missing ticket


def test_a_ticket_without_an_email_cannot_be_fetched_by_anyone(service: TicketService) -> None:
    ticket_id = service.submit("What is your return policy?").ticket_id
    with pytest.raises(NotFoundError):
        service.get_for_customer(ticket_id, EMAIL)


# ---------------------------------------------------------------- failures and restarts
def test_a_broken_workflow_never_becomes_a_customer_error(engine: Engine) -> None:
    class BrokenGraph:
        def invoke(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("secret internal detail")

    view = TicketService(engine, BrokenGraph()).submit("Where is my order?", EMAIL)  # type: ignore[arg-type]
    assert view.status is TicketStatus.COMPLETED and view.escalated
    assert view.escalation_reason == "workflow_error" and view.response == HANDOFF
    assert "secret" not in str(view)


def test_a_paused_ticket_survives_a_full_restart(
    engine: Engine, retriever: Retriever, tmp_path: Path
) -> None:
    """The point of the persistent checkpointer: new graph, new connection, same files."""
    checkpoint_file = tmp_path / "checkpoints.db"

    saver, connection = sqlite_saver(checkpoint_file)
    before = make_service(engine, retriever, saver)
    ticket_id = refund_ticket(before)
    connection.close()  # "the server stops"
    del before

    saver2, connection2 = sqlite_saver(checkpoint_file)  # "the server starts again"
    after = make_service(engine, retriever, saver2)
    assert [t.ticket_id for t in after.list_awaiting()] == [ticket_id]
    view = after.decide_ticket(ticket_id, APPROVE)
    assert view.status is TicketStatus.COMPLETED and "approved" in (view.response or "")
    assert refunds(engine)[0].status is RefundStatus.APPROVED
    connection2.close()


# ---------------------------------------------------------------- production wiring
def test_the_production_service_builds_from_settings(engine: Engine, tmp_path: Path) -> None:
    tickets = generate_sample_tickets(100, seed=1)  # SYNTHETIC: wiring only
    model_path = (
        IntentClassifier().fit(tickets["text"], tickets["intent"]).save(tmp_path / "intent.joblib")
    )
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        intent_model_path=str(model_path),
        policy_docs_dir=str(POLICIES),
        checkpoint_db_path=str(tmp_path / "ckpt.db"),
    )
    service, connection = build_ticket_service(settings, engine)
    try:
        assert isinstance(service, TicketService)
        assert service.submit("What is your return policy?", EMAIL).status is TicketStatus.COMPLETED
    finally:
        connection.close()


def test_the_production_service_explains_what_is_missing(engine: Engine, tmp_path: Path) -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, intent_model_path=str(tmp_path / "missing.joblib")
    )
    with pytest.raises(ConfigurationError, match="Intent model not found"):
        build_ticket_service(settings, engine)
    toy = IntentClassifier().fit(["a b", "c d"] * 3, ["x", "y"] * 3).save(tmp_path / "m.joblib")
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, intent_model_path=str(toy), policy_docs_dir=str(tmp_path / "nodocs")
    )
    with pytest.raises(ConfigurationError, match="Policy documents folder not found"):
        build_ticket_service(settings, engine)


# ---------------------------------------------------------------- cancellations
def order_row(engine: Engine, number: str) -> Order:
    with session_scope(engine) as session:
        order = session.scalars(select(Order).where(Order.order_number == number)).one()
        return order


def test_an_approved_cancellation_really_cancels_the_order(
    service: TicketService, engine: Engine
) -> None:
    view = service.submit("Please cancel my order ORD-3003", EMAIL)
    assert view.status is TicketStatus.AWAITING_APPROVAL and view.approval_kind == "cancellation"
    assert (
        order_row(engine, "ORD-3003").status is OrderStatus.PLACED
    )  # nothing yet: a person decides
    done = service.decide_ticket(view.ticket_id, APPROVE)
    assert done.status is TicketStatus.COMPLETED
    assert done.response == "Good news: our team approved your cancellation request."
    order = order_row(engine, "ORD-3003")
    assert order.status is OrderStatus.CANCELLED and order.cancelled_by == "agent.jane"


def test_a_rejected_cancellation_leaves_the_order_alone(
    service: TicketService, engine: Engine
) -> None:
    view = service.submit("Please cancel my order ORD-4004", EMAIL)
    done = service.decide_ticket(view.ticket_id, REJECT)
    assert "could not approve" in (done.response or "")
    order = order_row(engine, "ORD-4004")
    assert order.status is OrderStatus.PLACED and order.cancelled_by is None


def test_a_shipped_order_is_refused_at_once_without_a_ticket_waiting(
    service: TicketService,
) -> None:
    view = service.submit("Please cancel my order ORD-1001", EMAIL)
    assert view.status is TicketStatus.COMPLETED and not view.escalated
    assert "can't be cancelled" in (view.response or "")
    assert service.list_awaiting() == []


def test_if_the_order_ships_while_waiting_the_approval_is_refused_and_the_ticket_stays_open(
    service: TicketService, engine: Engine
) -> None:
    view = service.submit("Please cancel my order ORD-5005", EMAIL)
    with session_scope(engine) as session:
        order = session.scalars(select(Order).where(Order.order_number == "ORD-5005")).one()
        order.shipment = Shipment(
            carrier="DemoPost",
            tracking_number="T",
            shipped_at=datetime.now(UTC) - timedelta(hours=2),
        )
    with pytest.raises(WorkflowError, match="can no longer be cancelled"):
        service.decide_ticket(view.ticket_id, APPROVE)
    assert order_row(engine, "ORD-5005").status is OrderStatus.PLACED
    assert [t.ticket_id for t in service.list_awaiting()] == [view.ticket_id]
    done = service.decide_ticket(view.ticket_id, REJECT)  # staff can still close it by rejecting
    assert done.status is TicketStatus.COMPLETED


def test_a_retried_approval_after_a_crash_does_not_fail(
    service: TicketService, engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = service.submit("Please cancel my order ORD-3003", EMAIL)
    monkeypatch.setattr(
        TicketService, "_resume", lambda *a, **k: (_ for _ in ()).throw(ExternalServiceError("x"))
    )
    with pytest.raises(ExternalServiceError):
        service.decide_ticket(view.ticket_id, APPROVE)  # order cancelled, ticket not resumed
    assert order_row(engine, "ORD-3003").status is OrderStatus.CANCELLED
    monkeypatch.undo()
    done = service.decide_ticket(view.ticket_id, APPROVE)  # same decision again: safe
    assert done.status is TicketStatus.COMPLETED
