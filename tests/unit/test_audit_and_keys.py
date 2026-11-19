"""Audit trail and database-backed staff keys (create, rotate, revoke without a restart)."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine, select
from sqlalchemy.exc import OperationalError

from supportx.agents import HumanDecision
from supportx.api import create_app
from supportx.core import Settings, WorkflowError, get_settings
from supportx.core.exceptions import DataError
from supportx.db import AuditLog, RefundReason, StaffKey, session_scope
from supportx.rag import Retriever
from supportx.services import CancellationWorkflow, RefundWorkflow, StaffKeyService, list_audit
from supportx.services.staff_keys import main as keys_cli
from tests.unit.helpers import EMAIL, POLICIES, make_engine, make_service, sqlite_saver

NOW = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)


@pytest.fixture
def engine() -> Engine:
    return make_engine()


def rows(engine: Engine, **filters: str) -> list[AuditLog]:
    with session_scope(engine) as session:
        return list_audit(session, limit=100, **filters)


# ---------------------------------------------------------------- audit rows
def test_a_cancellation_is_audited_once_even_if_repeated(engine: Engine) -> None:
    for approver in ("jane", "bob"):
        with session_scope(engine) as session:
            CancellationWorkflow(session).cancel("ORD-3003", EMAIL, approver)
    found = rows(engine, action="order.cancelled")
    assert len(found) == 1 and found[0].actor == "jane" and found[0].target_id == "ORD-3003"


def test_a_refused_cancellation_leaves_no_audit_row(engine: Engine) -> None:
    with pytest.raises(WorkflowError):
        with session_scope(engine) as session:
            CancellationWorkflow(session).cancel("ORD-1001", EMAIL, "jane")  # already delivered
    assert rows(engine) == []


def test_refund_request_approval_and_rejection_are_audited(engine: Engine) -> None:
    with session_scope(engine) as session:
        _, first = RefundWorkflow(session).request_refund("ORD-1001", EMAIL, RefundReason.DAMAGED)
        assert first is not None
        refund_id = first.id
        RefundWorkflow(session).approve(refund_id, "jane", "photo ok")
    with session_scope(engine) as session:
        _, second = RefundWorkflow(session).request_refund("ORD-2002", EMAIL, RefundReason.DAMAGED)
        assert second is None  # outside the window: nothing created, nothing audited
    actions = [(r.actor, r.action) for r in reversed(rows(engine))]
    assert actions == [("customer", "refund.requested"), ("jane", "refund.approved")]
    detail = json.loads(rows(engine, action="refund.approved")[0].detail or "{}")
    assert detail["note"] == "photo ok" and detail["order"] == "ORD-1001"


def test_audit_details_never_contain_customer_emails(engine: Engine) -> None:
    with session_scope(engine) as session:
        _, req = RefundWorkflow(session).request_refund("ORD-1001", EMAIL, RefundReason.DAMAGED)
        assert req is not None
        RefundWorkflow(session).reject(req.id, "bob", "not eligible")
    assert all(EMAIL not in (r.detail or "") and EMAIL not in r.actor for r in rows(engine))


def test_ticket_decisions_are_audited_with_the_approver(engine: Engine, tmp_path: Path) -> None:
    saver, connection = sqlite_saver(tmp_path / "cp.db")
    try:
        service = make_service(engine, Retriever.from_documents(POLICIES), saver)
        view = service.submit("My mug arrived damaged, I want a refund", EMAIL, "ORD-1001")
        service.decide_ticket(view.ticket_id, HumanDecision(approved=True, approver="jane"))
    finally:
        connection.close()
    decided = rows(engine, action="ticket.decided")
    assert len(decided) == 1 and decided[0].actor == "jane"
    assert json.loads(decided[0].detail or "{}")["kind"] == "refund"
    assert rows(engine, action="refund.approved")  # the money decision was audited too


# ---------------------------------------------------------------- staff key service
def test_a_key_is_shown_once_and_only_its_hash_is_stored(engine: Engine) -> None:
    with session_scope(engine) as session:
        key = StaffKeyService(session).create("jane", now=NOW)
    assert key.startswith("sx_") and len(key) > 30
    with session_scope(engine) as session:
        stored = session.scalars(select(StaffKey)).one()
        assert stored.key_hash != key and key not in stored.key_hash
        service = StaffKeyService(session)
        assert service.authenticate(key, NOW) == "jane"
        assert service.authenticate(key + "x", NOW) is None
        assert service.authenticate("", NOW) is None
    audit = rows(engine, action="staff_key.created")[0]
    assert key not in (audit.detail or "") and audit.actor == "cli"


def test_revoked_and_expired_keys_stop_working(engine: Engine) -> None:
    with session_scope(engine) as session:
        service = StaffKeyService(session)
        permanent = service.create("jane", now=NOW)
        temporary = service.create("bob", expires_in=timedelta(hours=1), now=NOW)
        assert service.authenticate(temporary, NOW + timedelta(minutes=30)) == "bob"
        assert service.authenticate(temporary, NOW + timedelta(hours=2)) is None  # expired
        assert service.revoke(name="jane", now=NOW) == 1
        assert service.authenticate(permanent, NOW) is None
        assert service.revoke(name="jane", now=NOW) == 0  # nothing left to revoke
    assert len(rows(engine, action="staff_key.revoked")) == 1


def test_revoke_by_id_and_argument_checks(engine: Engine) -> None:
    with session_scope(engine) as session:
        service = StaffKeyService(session)
        service.create("jane", now=NOW)
        service.create("jane", now=NOW)  # two keys for one person (rotation overlap)
        first_id = service.all_keys()[0].id
        assert service.revoke(key_id=first_id, now=NOW) == 1
        assert service.has_active_keys(NOW)
        with pytest.raises(DataError):
            service.revoke()
        with pytest.raises(DataError):
            service.revoke(name="jane", key_id=1)
        with pytest.raises(DataError):
            service.create("bad name!")


def test_command_line_creates_lists_and_revokes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    url = f"sqlite:///{(tmp_path / 'keys.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        from supportx.db import init_db, make_engine

        init_db(make_engine(url))
        assert keys_cli(["create", "jane", "--days", "30"]) == 0
        assert "sx_" in capsys.readouterr().out
        assert keys_cli(["list"]) == 0 and "jane\tactive" in capsys.readouterr().out
        assert keys_cli(["revoke", "jane"]) == 0 and "Revoked 1" in capsys.readouterr().out
        assert keys_cli(["list"]) == 0 and "inactive" in capsys.readouterr().out
        assert keys_cli(["create", "bad name"]) == 2
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------- through the API
def make_client(engine: Engine, **settings: object) -> TestClient:
    values: dict[str, object] = dict(settings)
    config = Settings(_env_file=None, **values)  # type: ignore[call-arg,arg-type]
    return TestClient(create_app(config, engine))


def new_key(engine: Engine, name: str) -> str:
    with session_scope(engine) as session:
        return StaffKeyService(session).create(name)


def test_database_keys_authenticate_and_a_revoked_key_fails_at_once(engine: Engine) -> None:
    client = make_client(engine)  # no keys in the environment at all
    assert client.get("/staff/refunds/pending", headers={"X-API-Key": "x"}).status_code == 503

    old = new_key(engine, "jane")
    assert client.get("/staff/refunds/pending", headers={"X-API-Key": old}).status_code == 200
    assert client.get("/staff/refunds/pending", headers={"X-API-Key": "wrong"}).status_code == 401

    new = new_key(engine, "jane")  # rotation: add the new key, then revoke the old one
    with session_scope(engine) as session:
        StaffKeyService(session).revoke(name=None, key_id=1)
    assert client.get("/staff/refunds/pending", headers={"X-API-Key": old}).status_code == 401
    assert client.get("/staff/refunds/pending", headers={"X-API-Key": new}).status_code == 200


def test_a_database_key_decides_the_approver_even_if_the_body_lies(engine: Engine) -> None:
    client = make_client(engine)
    key = new_key(engine, "jane")
    with session_scope(engine) as session:
        _, req = RefundWorkflow(session).request_refund("ORD-1001", EMAIL, RefundReason.DAMAGED)
        assert req is not None
        refund_id = req.id
    response = client.post(
        f"/staff/refunds/{refund_id}/approve",
        headers={"X-API-Key": key},
        json={"approver": "the.ceo", "note": "ok"},
    )
    assert response.status_code == 200
    assert response.json()["decided_by"] == "jane"
    assert rows(engine, action="refund.approved")[0].actor == "jane"


def test_environment_keys_keep_working_if_the_key_table_is_unavailable(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(engine, staff_api_key=SecretStr("env-shared-key"))

    def broken(*args: object, **kwargs: object) -> str:
        raise OperationalError("select", {}, Exception("database down"))

    monkeypatch.setattr(StaffKeyService, "authenticate", broken)
    ok = client.get("/staff/refunds/pending", headers={"X-API-Key": "env-shared-key"})
    assert ok.status_code == 200  # matched before the database was asked
    bad = client.get("/staff/refunds/pending", headers={"X-API-Key": "something-else"})
    assert bad.status_code == 401  # a database fault never lets an unknown key in


def test_audit_endpoint_needs_a_key_filters_and_limits(engine: Engine) -> None:
    client = make_client(engine, staff_api_key=SecretStr("k"))
    with session_scope(engine) as session:
        CancellationWorkflow(session).cancel("ORD-3003", EMAIL, "jane")
        _, req = RefundWorkflow(session).request_refund("ORD-1001", EMAIL, RefundReason.DAMAGED)
        assert req is not None
    assert client.get("/staff/audit").status_code == 401

    everything = client.get("/staff/audit", headers={"X-API-Key": "k"}).json()
    assert [e["action"] for e in everything] == [
        "refund.requested",
        "order.cancelled",
    ]  # newest first
    assert everything[1]["actor"] == "jane" and everything[0]["detail"]["order"] == "ORD-1001"

    only = client.get("/staff/audit?action=order.cancelled", headers={"X-API-Key": "k"}).json()
    assert [e["target_id"] for e in only] == ["ORD-3003"]
    assert len(client.get("/staff/audit?limit=1", headers={"X-API-Key": "k"}).json()) == 1
