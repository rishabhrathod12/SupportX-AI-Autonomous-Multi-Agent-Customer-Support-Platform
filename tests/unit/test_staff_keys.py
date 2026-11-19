"""Personal staff keys: the recorded approver comes from the key, not from the request body."""

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from supportx.api import create_app
from supportx.core import ConfigurationError, Settings
from supportx.db import Customer, Order, OrderItem, RefundRequest, init_db, session_scope
from supportx.db.enums import RefundReason, RefundStatus

JANE = "jane-key-0123456789"
BOB = "bob-key-0123456789ab"
SHARED = "shared-key"


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg,arg-type]


# ---------------------------------------------------------------- settings validation
def test_named_keys_are_parsed_with_the_shared_key_last() -> None:
    settings = make_settings(
        staff_api_keys=SecretStr(f"jane:{JANE}, bob:{BOB}"), staff_api_key=SecretStr(SHARED)
    )
    assert settings.staff_identities() == [(JANE, "jane"), (BOB, "bob"), (SHARED, None)]


@pytest.mark.parametrize(
    "raw",
    [
        "jane",  # no key at all
        f"ja ne:{JANE}",  # name with a space
        "jane:short",  # key too short
        f"jane:{JANE},jane:{BOB}",  # duplicate name
        f"jane:{JANE},bob:{JANE}",  # duplicate key
    ],
)
def test_bad_staff_keys_fail_at_startup(raw: str) -> None:
    with pytest.raises(ConfigurationError):
        make_settings(staff_api_keys=SecretStr(raw))


def test_a_named_key_may_not_equal_the_shared_key() -> None:
    with pytest.raises(ConfigurationError, match="identical"):
        make_settings(staff_api_keys=SecretStr(f"jane:{JANE}"), staff_api_key=SecretStr(JANE))


# ---------------------------------------------------------------- through the API
@pytest.fixture
def client_and_engine():  # type: ignore[no-untyped-def]
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    init_db(engine)
    with session_scope(engine) as session:
        order = Order(
            order_number="ORD-1",
            customer=Customer(email="a@example.com", name="A"),
            items=[OrderItem(sku="M", name="Mug", unit_price_cents=2500, quantity=1)],
        )
        session.add(order)
        session.flush()
        session.add(
            RefundRequest(
                order_id=order.id,
                amount_cents=2500,
                reason=RefundReason.DAMAGED,
                status=RefundStatus.PENDING_APPROVAL,
            )
        )
    settings = make_settings(
        staff_api_keys=SecretStr(f"jane:{JANE},bob:{BOB}"), staff_api_key=SecretStr(SHARED)
    )
    yield TestClient(create_app(settings, engine)), engine
    engine.dispose()


def _decider(engine) -> str | None:  # type: ignore[no-untyped-def]
    with session_scope(engine) as session:
        return session.scalars(select(RefundRequest)).one().decided_by


def test_every_configured_key_works_and_a_wrong_one_does_not(client_and_engine) -> None:  # type: ignore[no-untyped-def]
    client, _ = client_and_engine
    for key in (JANE, BOB, SHARED):
        assert client.get("/staff/refunds/pending", headers={"X-API-Key": key}).status_code == 200
    assert client.get("/staff/refunds/pending", headers={"X-API-Key": "nope"}).status_code == 401
    assert client.get("/staff/refunds/pending").status_code == 401
    assert client.get("/staff/refunds/pending", headers={"X-API-Key": ""}).status_code == 401


def test_a_personal_key_decides_the_recorded_approver_even_if_the_body_lies(
    client_and_engine,  # type: ignore[no-untyped-def]
) -> None:
    client, engine = client_and_engine
    response = client.post(
        "/staff/refunds/1/approve",
        headers={"X-API-Key": JANE},
        json={"approver": "the.ceo", "note": "ok"},  # claims to be someone else
    )
    assert response.status_code == 200
    assert _decider(engine) == "jane"


def test_a_personal_key_needs_no_approver_in_the_body(
    client_and_engine,  # type: ignore[no-untyped-def]
) -> None:
    client, engine = client_and_engine
    response = client.post(
        "/staff/refunds/1/reject", headers={"X-API-Key": BOB}, json={"note": "no"}
    )
    assert response.status_code == 200
    assert _decider(engine) == "bob"


def test_the_shared_key_still_requires_a_stated_approver(client_and_engine) -> None:  # type: ignore[no-untyped-def]
    client, engine = client_and_engine
    missing = client.post("/staff/refunds/1/approve", headers={"X-API-Key": SHARED}, json={})
    assert missing.status_code == 422 and "approver" in missing.json()["detail"]
    assert _decider(engine) is None
    ok = client.post(
        "/staff/refunds/1/approve", headers={"X-API-Key": SHARED}, json={"approver": "agent.x"}
    )
    assert ok.status_code == 200 and _decider(engine) == "agent.x"


def test_with_no_keys_configured_staff_endpoints_are_off() -> None:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    init_db(engine)
    client = TestClient(create_app(make_settings(), engine))
    assert client.get("/staff/refunds/pending", headers={"X-API-Key": "x"}).status_code == 503
