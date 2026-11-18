from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine, select

from supportx.api import create_app
from supportx.core import Settings
from supportx.db import RefundRequest, RefundStatus, session_scope
from supportx.rag import Retriever
from tests.unit.helpers import EMAIL, POLICIES, make_engine, make_service, sqlite_saver

STAFF = {"X-API-Key": "test-staff-key"}


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"staff_api_key": SecretStr("test-staff-key"), **overrides}
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever.from_documents(POLICIES)


@pytest.fixture
def engine() -> Engine:
    return make_engine()


@pytest.fixture
def client(engine: Engine, retriever: Retriever, tmp_path: Path) -> Iterator[TestClient]:
    saver, connection = sqlite_saver(tmp_path / "checkpoints.db")
    app = create_app(make_settings(), engine, tickets=make_service(engine, retriever, saver))
    yield TestClient(app)
    connection.close()


def send(client: TestClient, message: str, **extra: Any) -> Any:
    return client.post("/tickets", json={"message": message, "customer_email": EMAIL, **extra})


def refund_ticket(client: TestClient) -> str:
    response = send(client, "My mug arrived damaged, I want a refund", order_number="ORD-1001")
    assert response.status_code == 202
    return str(response.json()["ticket_id"])


# ---------------------------------------------------------------- customer endpoints
def test_a_question_is_answered_in_the_same_call(client: TestClient) -> None:
    response = send(client, "Where is my order ORD-1001?")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert (
        body["message"] == "Order ORD-1001 is currently delivered. Tracking number: TRK-ORD-1001."
    )


def test_a_refund_request_is_accepted_but_not_finished(client: TestClient) -> None:
    response = send(client, "I want a refund", order_number="ORD-1001")
    assert response.status_code == 202
    body = response.json()
    assert set(body) == {"ticket_id", "status", "message"}  # no internal ids or reasons leak
    assert body["status"] == "awaiting_approval" and "support team" in body["message"]


def test_the_customer_can_check_on_the_ticket(client: TestClient) -> None:
    ticket_id = refund_ticket(client)
    status = client.post("/tickets/status", json={"ticket_id": ticket_id, "customer_email": EMAIL})
    assert status.status_code == 200 and status.json()["status"] == "awaiting_approval"


def test_wrong_email_and_unknown_ticket_look_the_same(client: TestClient) -> None:
    ticket_id = refund_ticket(client)
    wrong = client.post(
        "/tickets/status", json={"ticket_id": ticket_id, "customer_email": "x@y.com"}
    )
    unknown = client.post("/tickets/status", json={"ticket_id": "nope", "customer_email": EMAIL})
    assert wrong.status_code == unknown.status_code == 404
    assert wrong.json() == unknown.json()


@pytest.mark.parametrize(
    "bad",
    [
        {"message": ""},
        {"message": "hi", "customer_email": "not-an-email"},
        {"message": "x" * 2001},
        {"message": "hi", "unexpected": 1},
        {},
    ],
)
def test_invalid_ticket_bodies_are_rejected(client: TestClient, bad: dict[str, Any]) -> None:
    assert client.post("/tickets", json=bad).status_code == 422


# ---------------------------------------------------------------- staff decisions
def test_staff_endpoints_need_the_key(client: TestClient) -> None:
    assert client.get("/staff/tickets/awaiting").status_code == 401
    assert client.post("/staff/tickets/x/decision", json={}).status_code == 401  # before the body


def test_full_flow_ticket_to_staff_approval_to_customer_reply(
    client: TestClient, engine: Engine
) -> None:
    ticket_id = refund_ticket(client)
    awaiting = client.get("/staff/tickets/awaiting", headers=STAFF).json()
    assert [t["ticket_id"] for t in awaiting] == [ticket_id]
    assert awaiting[0]["approval_kind"] == "refund" and awaiting[0]["amount_cents"] == 5000

    decided = client.post(
        f"/staff/tickets/{ticket_id}/decision",
        headers=STAFF,
        json={"approved": True, "approver": "agent.jane", "note": "Photo ok"},
    )
    assert decided.status_code == 200 and decided.json()["status"] == "completed"

    # the customer now sees the final answer; the database and the ticket agree
    status = client.post("/tickets/status", json={"ticket_id": ticket_id, "customer_email": EMAIL})
    assert status.json()["message"] == "Good news: our team approved your refund request of 50.00."
    with session_scope(engine) as session:
        assert session.scalars(select(RefundRequest)).one().status is RefundStatus.APPROVED
    assert client.get("/staff/tickets/awaiting", headers=STAFF).json() == []


def test_rejection_needs_a_reason_and_double_decisions_conflict(client: TestClient) -> None:
    ticket_id = refund_ticket(client)
    url = f"/staff/tickets/{ticket_id}/decision"
    assert (
        client.post(url, headers=STAFF, json={"approved": False, "approver": "bob"}).status_code
        == 422
    )
    assert (
        client.post(url, headers=STAFF, json={"approved": True, "approver": ""}).status_code == 422
    )

    rejected = client.post(
        url, headers=STAFF, json={"approved": False, "approver": "bob", "note": "No photo"}
    )
    assert rejected.status_code == 200
    again = client.post(url, headers=STAFF, json={"approved": True, "approver": "jane"})
    assert again.status_code == 409 and "not waiting" in again.json()["detail"]
    unknown = client.post(
        "/staff/tickets/nope/decision", headers=STAFF, json={"approved": True, "approver": "j"}
    )
    assert unknown.status_code == 404


def test_the_old_refund_endpoint_now_resumes_the_waiting_ticket(
    client: TestClient, engine: Engine
) -> None:
    """Deciding through /staff/refunds/... must not leave the ticket paused forever."""
    ticket_id = refund_ticket(client)
    (pending,) = client.get("/staff/refunds/pending", headers=STAFF).json()
    approved = client.post(
        f"/staff/refunds/{pending['id']}/approve", headers=STAFF, json={"approver": "agent.jane"}
    )
    assert approved.status_code == 200 and approved.json()["status"] == "approved"
    status = client.post("/tickets/status", json={"ticket_id": ticket_id, "customer_email": EMAIL})
    assert status.json()["status"] == "completed" and "approved" in status.json()["message"]


def test_the_old_refund_endpoint_reports_a_conflicting_decision(client: TestClient) -> None:
    refund_ticket(client)
    (pending,) = client.get("/staff/refunds/pending", headers=STAFF).json()
    base = f"/staff/refunds/{pending['id']}"
    assert (
        client.post(f"{base}/approve", headers=STAFF, json={"approver": "jane"}).status_code == 200
    )
    conflict = client.post(f"{base}/reject", headers=STAFF, json={"approver": "bob", "note": "no"})
    assert conflict.status_code == 409 and "already approved" in conflict.json()["detail"]


# ---------------------------------------------------------------- when the workflow is off
def test_ticket_endpoints_say_unavailable_when_the_workflow_is_not_built(engine: Engine) -> None:
    client = TestClient(create_app(make_settings(), engine))  # no ticket service
    response = client.post("/tickets", json={"message": "hi"})
    assert response.status_code == 503 and "not available" in response.json()["detail"]
    assert client.get("/staff/tickets/awaiting", headers=STAFF).status_code == 503
    assert client.get("/health").status_code == 200  # the rest of the API keeps working
