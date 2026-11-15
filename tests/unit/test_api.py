from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine, create_engine, func, select, text
from sqlalchemy.pool import StaticPool

from supportx.api import create_app
from supportx.core import ConfigurationError, Settings
from supportx.db import (
    Customer,
    Order,
    OrderItem,
    OrderStatus,
    RefundRequest,
    RefundStatus,
    Shipment,
    init_db,
    session_scope,
)

EMAIL = "test@example.com"
STAFF = {"X-API-Key": "test-staff-key"}


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"staff_api_key": SecretStr("test-staff-key"), **overrides}
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def engine() -> Iterator[Engine]:
    # StaticPool + check_same_thread=False: the API serves requests from worker threads,
    # and an in-memory SQLite database must be shared between them.
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    init_db(eng)
    now = datetime.now(UTC)
    with session_scope(eng) as session:
        session.add(
            Order(
                order_number="ORD-1001",
                customer=Customer(email=EMAIL, name="Test Customer"),
                status=OrderStatus.DELIVERED,
                items=[OrderItem(sku="A", name="Mug", unit_price_cents=2500, quantity=2)],
                shipment=Shipment(
                    carrier="DemoPost",
                    tracking_number="TRK1",
                    shipped_at=now - timedelta(days=8),
                    delivered_at=now - timedelta(days=5),
                ),
            )
        )
    yield eng
    eng.dispose()


@pytest.fixture
def client(engine: Engine) -> TestClient:
    return TestClient(create_app(make_settings(), engine))


def order_body(**extra: object) -> dict[str, object]:
    return {"order_number": "ORD-1001", "customer_email": EMAIL, **extra}


def refund_count(engine: Engine) -> int:
    with session_scope(engine) as session:
        return session.scalar(select(func.count()).select_from(RefundRequest)) or 0


# ---------------------------------------------------------------- health
def test_health_and_readiness(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/health/ready").json() == {"status": "ready"}


def test_readiness_reports_an_unreachable_database(engine: Engine) -> None:
    broken = create_engine("sqlite:////nonexistent-folder/none.db")
    app = create_app(make_settings(), broken)
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 503
    assert "not reachable" in response.json()["detail"]
    assert "nonexistent" not in response.text  # no internal details leak


# ---------------------------------------------------------------- order lookup
def test_order_lookup_for_the_owner(client: TestClient) -> None:
    response = client.post("/orders/lookup", json=order_body())
    assert response.status_code == 200
    data = response.json()
    assert data["order_number"] == "ORD-1001" and data["total_cents"] == 5000
    assert data["status"] == "delivered" and data["has_pending_refund"] is False


def test_wrong_email_and_missing_order_are_indistinguishable(client: TestClient) -> None:
    wrong = client.post("/orders/lookup", json=order_body(customer_email="x@example.com"))
    missing = client.post("/orders/lookup", json=order_body(order_number="ORD-0"))
    assert wrong.status_code == missing.status_code == 404
    assert wrong.json() == missing.json()


@pytest.mark.parametrize(
    "bad",
    [
        {"order_number": "", "customer_email": EMAIL},
        {"order_number": "ORD-1001", "customer_email": "not-an-email"},
        {"order_number": "ORD-1001"},
        {"order_number": "ORD-1001", "customer_email": EMAIL, "unexpected": 1},
    ],
)
def test_invalid_lookup_bodies_are_rejected(client: TestClient, bad: dict[str, object]) -> None:
    assert client.post("/orders/lookup", json=bad).status_code == 422


# ---------------------------------------------------------------- refund requests
def test_eligibility_check_creates_nothing(client: TestClient, engine: Engine) -> None:
    response = client.post("/refunds/eligibility", json=order_body(reason="damaged"))
    assert response.status_code == 200
    decision = response.json()
    assert decision["eligible"] is True and decision["requires_human_approval"] is True
    assert refund_count(engine) == 0


def test_refund_request_is_saved_and_blocks_a_duplicate(client: TestClient, engine: Engine) -> None:
    first = client.post("/refunds/requests", json=order_body(reason="damaged"))
    assert first.status_code == 201
    body = first.json()
    assert body["created"] is True and body["request"]["status"] == "pending_approval"
    assert body["request"]["amount_cents"] == 5000
    assert refund_count(engine) == 1  # committed before the response was returned

    second = client.post("/refunds/requests", json=order_body(reason="damaged"))
    assert second.status_code == 200
    assert second.json()["created"] is False and second.json()["request"] is None
    assert second.json()["decision"]["reason_codes"] == ["pending_request_exists"]
    assert refund_count(engine) == 1


def test_partial_amount_and_invalid_reason(client: TestClient) -> None:
    ok = client.post(
        "/refunds/requests", json=order_body(reason="changed_mind", requested_cents=1200)
    )
    assert ok.json()["request"]["amount_cents"] == 1200
    assert (
        client.post("/refunds/eligibility", json=order_body(reason="nonsense")).status_code == 422
    )
    assert (
        client.post(
            "/refunds/eligibility", json=order_body(reason="damaged", requested_cents=0)
        ).status_code
        == 422
    )


def test_a_stranger_cannot_request_a_refund(client: TestClient, engine: Engine) -> None:
    body = order_body(customer_email="intruder@example.com", reason="damaged")
    assert client.post("/refunds/requests", json=body).status_code == 404
    assert refund_count(engine) == 0


# ---------------------------------------------------------------- staff: authentication
def test_staff_routes_need_the_right_key(client: TestClient) -> None:
    assert client.get("/staff/refunds/pending").status_code == 401
    assert client.get("/staff/refunds/pending", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/staff/refunds/pending", headers=STAFF).status_code == 200
    # authentication is checked before the body, so strangers learn nothing about the schema
    assert client.post("/staff/refunds/1/reject", json={}).status_code == 401


def test_staff_routes_are_disabled_without_a_configured_key(engine: Engine) -> None:
    app = create_app(make_settings(staff_api_key=None), engine)
    response = TestClient(app).get("/staff/refunds/pending", headers=STAFF)
    assert response.status_code == 503 and "STAFF_API_KEY" in response.json()["detail"]


def test_the_staff_key_never_appears_in_responses(client: TestClient) -> None:
    for response in (
        client.get("/staff/refunds/pending"),
        client.get("/staff/refunds/pending", headers={"X-API-Key": "wrong"}),
        client.get("/openapi.json"),
    ):
        assert "test-staff-key" not in response.text


# ---------------------------------------------------------------- staff: decisions
def create_pending(client: TestClient) -> int:
    return int(
        client.post("/refunds/requests", json=order_body(reason="damaged")).json()["request"]["id"]
    )


def test_full_flow_request_list_approve(client: TestClient, engine: Engine) -> None:
    refund_id = create_pending(client)
    pending = client.get("/staff/refunds/pending", headers=STAFF).json()
    assert [r["id"] for r in pending] == [refund_id] and pending[0]["order_number"] == "ORD-1001"

    approved = client.post(
        f"/staff/refunds/{refund_id}/approve",
        headers=STAFF,
        json={"approver": "agent.jane", "note": "Photo confirms damage"},
    )
    assert approved.status_code == 200
    data = approved.json()
    assert data["status"] == "approved" and data["decided_by"] == "agent.jane"
    assert data["decided_at"] is not None
    assert client.get("/staff/refunds/pending", headers=STAFF).json() == []
    with session_scope(engine) as session:
        assert session.get(RefundRequest, refund_id).status is RefundStatus.APPROVED  # type: ignore[union-attr]


def test_reject_needs_a_reason_and_a_second_decision_conflicts(client: TestClient) -> None:
    refund_id = create_pending(client)
    url = f"/staff/refunds/{refund_id}/reject"
    assert client.post(url, headers=STAFF, json={"approver": "agent.jane"}).status_code == 422
    assert (
        client.post(url, headers=STAFF, json={"approver": "agent.jane", "note": "  "}).status_code
        == 422
    )
    assert (
        client.post(url, headers=STAFF, json={"approver": "", "note": "no photo"}).status_code
        == 422
    )

    rejected = client.post(url, headers=STAFF, json={"approver": "agent.jane", "note": "No photo"})
    assert rejected.status_code == 200 and rejected.json()["status"] == "rejected"

    again = client.post(
        f"/staff/refunds/{refund_id}/approve", headers=STAFF, json={"approver": "agent.bob"}
    )
    assert again.status_code == 409 and "already rejected" in again.json()["detail"]


def test_unknown_refund_id_is_404(client: TestClient) -> None:
    response = client.post(
        "/staff/refunds/999/approve", headers=STAFF, json={"approver": "agent.jane"}
    )
    assert response.status_code == 404


# ---------------------------------------------------------------- app-level behavior
def test_api_docs_are_hidden_in_production(engine: Engine) -> None:
    prod = TestClient(create_app(make_settings(app_env="production"), engine))
    assert prod.get("/docs").status_code == 404 and prod.get("/openapi.json").status_code == 404
    dev = TestClient(create_app(make_settings(app_env="development"), engine))
    assert dev.get("/docs").status_code == 200


def test_misconfiguration_returns_a_generic_503(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app(make_settings(), engine)

    def boom(*_: object, **__: object) -> None:
        raise ConfigurationError("DATABASE_URL=postgres://user:secret@host/db is broken")

    monkeypatch.setattr("supportx.services.orders.OrderService.get_order_summary", boom)
    response = TestClient(app).post("/orders/lookup", json=order_body())
    assert response.status_code == 503
    assert "secret" not in response.text and "DATABASE_URL" not in response.text


def test_database_is_untouched_by_read_only_calls(client: TestClient, engine: Engine) -> None:
    client.post("/orders/lookup", json=order_body())
    client.post("/refunds/eligibility", json=order_body(reason="damaged"))
    with engine.connect() as conn:
        assert conn.execute(text("select count(*) from refund_requests")).scalar_one() == 0
