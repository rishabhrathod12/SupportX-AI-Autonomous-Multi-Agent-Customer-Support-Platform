"""Two tickets asking to cancel the same order must not double-cancel or contradict each other."""

from pathlib import Path

import pytest
from sqlalchemy import Engine, select

from supportx.agents import HumanDecision, TicketService
from supportx.core import WorkflowError
from supportx.db import Order, OrderStatus, TicketStatus, session_scope
from supportx.rag import Retriever
from tests.unit.helpers import EMAIL, POLICIES, make_engine, make_service, sqlite_saver

JANE = HumanDecision(approved=True, approver="jane", note="ok")
BOB = HumanDecision(approved=True, approver="bob", note="ok")
REJECT = HumanDecision(approved=False, approver="bob", note="customer changed their mind")


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever.from_documents(POLICIES)


@pytest.fixture
def engine() -> Engine:
    return make_engine()


@pytest.fixture
def service(engine: Engine, retriever: Retriever, tmp_path: Path):  # type: ignore[no-untyped-def]
    saver, connection = sqlite_saver(tmp_path / "checkpoints.db")
    yield make_service(engine, retriever, saver)
    connection.close()


def _order(engine: Engine) -> Order:
    with session_scope(engine) as session:
        return session.scalars(select(Order).where(Order.order_number == "ORD-3003")).one()


def _submit_twice(service: TicketService) -> tuple[str, str]:
    first = service.submit("Please cancel my order ORD-3003", EMAIL)
    second = service.submit("cancel ORD-3003 please, I asked already", EMAIL)
    assert first.status is second.status is TicketStatus.AWAITING_APPROVAL
    return first.ticket_id, second.ticket_id


def test_approving_both_duplicate_tickets_cancels_once_and_keeps_the_first_approver(
    service: TicketService, engine: Engine
) -> None:
    first, second = _submit_twice(service)
    assert service.decide_ticket(first, JANE).status is TicketStatus.COMPLETED
    assert service.decide_ticket(second, BOB).status is TicketStatus.COMPLETED  # already cancelled
    order = _order(engine)
    assert order.status is OrderStatus.CANCELLED and order.cancelled_by == "jane"


def test_rejecting_one_duplicate_does_not_stop_the_other_from_being_approved(
    service: TicketService, engine: Engine
) -> None:
    first, second = _submit_twice(service)
    service.decide_ticket(first, REJECT)
    assert _order(engine).status is OrderStatus.PLACED
    service.decide_ticket(second, JANE)
    assert _order(engine).status is OrderStatus.CANCELLED


def test_a_new_request_after_cancellation_is_answered_without_a_human(
    service: TicketService,
) -> None:
    first, _ = _submit_twice(service)
    service.decide_ticket(first, JANE)
    again = service.submit("Please cancel my order ORD-3003", EMAIL)
    assert again.status is TicketStatus.COMPLETED
    assert "already cancelled" in (again.response or "")


def test_a_decided_ticket_cannot_be_decided_again(service: TicketService) -> None:
    first, _ = _submit_twice(service)
    service.decide_ticket(first, JANE)
    with pytest.raises(WorkflowError):
        service.decide_ticket(first, BOB)
