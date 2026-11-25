"""Transactional outbox: follow-ups are saved with the change and delivered later, repeatably."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import Engine

from supportx.api import create_app
from supportx.core import Settings, WorkflowError, get_settings
from supportx.db import OutboxEvent, OutboxStatus, RefundReason, init_db, session_scope
from supportx.integrations import worker
from supportx.integrations.mocks import MockInventoryService, MockNotifier, MockPaymentProvider
from supportx.integrations.outbox import (
    NOTIFY,
    PAYMENT_REFUND,
    STOCK_RELEASE,
    OutboxDispatcher,
    Providers,
    enqueue,
    list_events,
    requeue,
)
from supportx.services import CancellationWorkflow, RefundWorkflow, list_audit
from tests.unit.helpers import EMAIL, make_engine

# Events are stamped with the real clock when enqueued, so the test clock starts an hour ahead.
NOW = datetime.now(UTC) + timedelta(hours=1)


@pytest.fixture
def engine() -> Engine:
    return make_engine()


@pytest.fixture
def providers() -> Providers:
    return Providers(MockPaymentProvider(), MockInventoryService(), MockNotifier())


def events(engine: Engine, status: OutboxStatus | None = None) -> list[OutboxEvent]:
    with session_scope(engine) as session:
        return list(reversed(list_events(session, status, 100)))


def cancel(engine: Engine, number: str = "ORD-3003", approver: str = "jane") -> None:
    with session_scope(engine) as session:
        CancellationWorkflow(session).cancel(number, EMAIL, approver)


def approved_refund(engine: Engine, approve: bool = True) -> int:
    with session_scope(engine) as session:
        _, req = RefundWorkflow(session).request_refund("ORD-1001", EMAIL, RefundReason.DAMAGED)
        assert req is not None
        refund_id = req.id
        if approve:
            RefundWorkflow(session).approve(refund_id, "jane", "ok")
        else:
            RefundWorkflow(session).reject(refund_id, "jane", "no")
    return refund_id


# ---------------------------------------------------------------- saved with the change
def test_a_cancellation_saves_payment_stock_and_notice_events_with_it(engine: Engine) -> None:
    cancel(engine)
    saved = events(engine)
    assert [e.event_type for e in saved] == [PAYMENT_REFUND, STOCK_RELEASE, NOTIFY]
    assert all(e.status is OutboxStatus.PENDING and e.attempts == 0 for e in saved)
    assert json.loads(saved[0].payload) == {"order_number": "ORD-3003", "amount_cents": 2500}
    assert json.loads(saved[1].payload)["lines"] == [{"sku": "M", "quantity": 1}]


def test_repeating_a_cancellation_or_a_refused_one_adds_no_events(engine: Engine) -> None:
    cancel(engine)
    cancel(engine, approver="bob")  # idempotent repeat
    assert len(events(engine)) == 3
    with pytest.raises(WorkflowError):
        cancel(engine, "ORD-1001")  # delivered: refused
    assert len(events(engine)) == 3


def test_an_approved_refund_saves_a_payment_and_a_notice_a_rejection_only_a_notice(
    engine: Engine,
) -> None:
    approved_refund(engine)
    kinds = [e.event_type for e in events(engine)]
    assert kinds == [PAYMENT_REFUND, NOTIFY]
    payment = json.loads(events(engine)[0].payload)
    assert payment == {"order_number": "ORD-1001", "amount_cents": 5000}

    other = make_engine()
    approved_refund(other, approve=False)
    assert [e.event_type for e in events(other)] == [NOTIFY]


def test_payloads_carry_ids_not_emails_and_enqueue_is_idempotent(engine: Engine) -> None:
    cancel(engine)
    assert all(EMAIL not in e.payload for e in events(engine))
    with session_scope(engine) as session:
        first = enqueue(session, PAYMENT_REFUND, "cancel-ORD-3003-payment", {"other": 1})
        assert first.payload == events(engine)[0].payload  # the first event wins, nothing new
    assert len(events(engine)) == 3


# ---------------------------------------------------------------- delivery
def test_delivery_marks_events_sent_labels_mocks_and_audits(
    engine: Engine, providers: Providers
) -> None:
    cancel(engine)
    summary = OutboxDispatcher(engine, providers).run_once(NOW)
    assert (summary.sent, summary.retried, summary.dead) == (3, 0, 0)
    sent = events(engine, OutboxStatus.SENT)
    assert len(sent) == 3 and all(e.is_mock for e in sent)
    assert all((e.provider_reference or "").startswith("mock-") for e in sent)
    assert providers.payments and providers.payments.refunded == [("ORD-3003", 2500)]  # type: ignore[attr-defined]
    with session_scope(engine) as session:
        assert len(list_audit(session, 100, action="outbox.sent")) == 3
    assert OutboxDispatcher(engine, providers).run_once(NOW).sent == 0  # nothing left to do


def test_redelivery_after_a_crash_does_not_repeat_the_effect(
    engine: Engine, providers: Providers
) -> None:
    """The provider acted but the row was never marked sent (a crash): the retry reuses the key."""
    cancel(engine)
    payment = events(engine)[0]
    assert isinstance(providers.payments, MockPaymentProvider)
    providers.payments.refund(payment.idempotency_key, "ORD-3003", 2500)  # the lost first delivery
    OutboxDispatcher(engine, providers).run_once(NOW)
    assert providers.payments.calls == 1 and providers.payments.refunded == [("ORD-3003", 2500)]
    assert events(engine, OutboxStatus.SENT)[0].event_type == PAYMENT_REFUND


def test_transient_failures_back_off_then_succeed(engine: Engine, providers: Providers) -> None:
    cancel(engine)
    assert isinstance(providers.payments, MockPaymentProvider)
    providers.payments.fail_next(2)
    dispatcher = OutboxDispatcher(engine, providers, base_backoff_seconds=30)

    first = dispatcher.run_once(NOW)
    assert (first.sent, first.retried) == (2, 1)  # stock and notice went through, payment waits
    waiting = events(engine, OutboxStatus.PENDING)[0]
    assert waiting.attempts == 1 and "unavailable" in (waiting.last_error or "")
    assert (waiting.next_attempt_at.replace(tzinfo=UTC) - NOW) == timedelta(seconds=30)

    assert dispatcher.run_once(NOW + timedelta(seconds=10)).retried == 0  # not due yet
    second = dispatcher.run_once(NOW + timedelta(seconds=31))
    assert second.retried == 1
    still = events(engine, OutboxStatus.PENDING)[0]
    assert (still.next_attempt_at.replace(tzinfo=UTC) - (NOW + timedelta(seconds=31))) == timedelta(
        seconds=60
    )  # backoff doubled

    third = dispatcher.run_once(NOW + timedelta(seconds=200))
    assert third.sent == 1 and events(engine, OutboxStatus.PENDING) == []


def test_too_many_attempts_and_permanent_errors_go_dead_without_blocking_others(
    engine: Engine, providers: Providers
) -> None:
    cancel(engine)
    assert isinstance(providers.payments, MockPaymentProvider)
    providers.payments.fail_permanently()
    summary = OutboxDispatcher(engine, providers).run_once(NOW)
    assert (summary.sent, summary.dead) == (2, 1)  # one bad event never blocks the others
    dead = events(engine, OutboxStatus.DEAD)
    assert len(dead) == 1 and dead[0].attempts == 1 and dead[0].event_type == PAYMENT_REFUND

    other = make_engine()
    cancel(other)
    flaky = Providers(MockPaymentProvider(), MockInventoryService(), MockNotifier())
    assert isinstance(flaky.payments, MockPaymentProvider)
    flaky.payments.fail_next(99)
    dispatcher = OutboxDispatcher(other, flaky, max_attempts=3, base_backoff_seconds=1)
    for step in range(3):
        dispatcher.run_once(NOW + timedelta(hours=step))
    assert events(other, OutboxStatus.DEAD)[0].attempts == 3
    with session_scope(other) as session:
        assert len(list_audit(session, 100, action="outbox.dead")) == 1


def test_a_dead_event_can_be_requeued_and_then_delivered(
    engine: Engine, providers: Providers
) -> None:
    cancel(engine)
    assert isinstance(providers.payments, MockPaymentProvider)
    providers.payments.fail_permanently()
    dispatcher = OutboxDispatcher(engine, providers)
    dispatcher.run_once(NOW)
    dead_id = events(engine, OutboxStatus.DEAD)[0].id

    providers.payments = MockPaymentProvider()  # the cause was fixed
    with session_scope(engine) as session:
        assert requeue(session, dead_id, "jane", NOW).status is OutboxStatus.PENDING
    with pytest.raises(WorkflowError):  # only dead events can be requeued
        with session_scope(engine) as session:
            requeue(session, dead_id, "jane", NOW)
    assert dispatcher.run_once(NOW).sent == 1
    with session_scope(engine) as session:
        assert len(list_audit(session, 100, action="outbox.requeued")) == 1


def test_events_without_a_configured_adapter_wait_untouched(engine: Engine) -> None:
    cancel(engine)
    summary = OutboxDispatcher(engine, Providers(notifier=MockNotifier())).run_once(NOW)
    assert (summary.sent, summary.skipped) == (1, 2)
    assert [e.event_type for e in events(engine, OutboxStatus.PENDING)] == [
        PAYMENT_REFUND,
        STOCK_RELEASE,
    ]
    assert all(e.attempts == 0 for e in events(engine, OutboxStatus.PENDING))


def test_an_unexpected_provider_error_is_retried_not_lost(engine: Engine) -> None:
    class Broken(MockNotifier):
        def send(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            raise RuntimeError("bug")

    cancel(engine)
    summary = OutboxDispatcher(engine, Providers(notifier=Broken())).run_once(NOW)
    assert summary.retried == 1
    notice = next(e for e in events(engine, OutboxStatus.PENDING) if e.event_type == NOTIFY)
    assert "RuntimeError" in (notice.last_error or "") and notice.attempts == 1


# ---------------------------------------------------------------- worker command and API
def test_worker_refuses_to_run_while_integrations_are_disabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("INTEGRATIONS_MODE", "disabled")
    get_settings.cache_clear()
    try:
        assert worker.main(["--once"]) == 2
        assert "disabled" in capsys.readouterr().err
    finally:
        get_settings.cache_clear()


def test_worker_in_mock_mode_delivers_and_says_it_is_a_mock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from supportx.db import make_engine

    url = f"sqlite:///{(tmp_path / 'worker.db').as_posix()}"
    engine = make_engine(url)
    init_db(engine)
    with session_scope(engine) as session:
        enqueue(session, NOTIFY, "k1", {"customer_id": 1, "template": "t", "params": {}})
    engine.dispose()
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("INTEGRATIONS_MODE", "mock")
    get_settings.cache_clear()
    try:
        assert worker.main(["--once"]) == 0
        captured = capsys.readouterr()
        assert "MOCK ADAPTERS" in captured.err and "sent=1" in captured.out
    finally:
        get_settings.cache_clear()


def make_client(engine: Engine) -> TestClient:
    settings = Settings(_env_file=None, staff_api_keys=SecretStr("jane:" + "j" * 20))  # type: ignore[call-arg]
    return TestClient(create_app(settings, engine))


JANE = {"X-API-Key": "j" * 20}


def test_staff_can_list_events_and_retry_a_dead_one(engine: Engine, providers: Providers) -> None:
    client = make_client(engine)
    cancel(engine)
    assert client.get("/staff/outbox").status_code == 401

    listed = client.get("/staff/outbox", headers=JANE).json()
    assert [e["event_type"] for e in listed] == [
        NOTIFY,
        STOCK_RELEASE,
        PAYMENT_REFUND,
    ]  # newest first
    assert client.get("/staff/outbox?status=sent", headers=JANE).json() == []

    assert isinstance(providers.payments, MockPaymentProvider)
    providers.payments.fail_permanently()
    OutboxDispatcher(engine, providers).run_once(NOW)
    dead = client.get("/staff/outbox?status=dead", headers=JANE).json()
    assert len(dead) == 1 and dead[0]["last_error"]

    retried = client.post(f"/staff/outbox/{dead[0]['id']}/retry", headers=JANE)
    assert retried.status_code == 200 and retried.json()["status"] == "pending"
    assert client.post(f"/staff/outbox/{dead[0]['id']}/retry", headers=JANE).status_code == 409
    assert client.post("/staff/outbox/9999/retry", headers=JANE).status_code == 404
    with session_scope(engine) as session:
        assert list_audit(session, 5, action="outbox.requeued")[0].actor == "jane"


def test_staff_model_info_shows_the_integrations_mode(engine: Engine) -> None:
    body = make_client(engine).get("/staff/model-info", headers=JANE).json()
    assert body["integrations_mode"] == "disabled"


# ---------------------------------------------------------------- Phase 8 additions
def test_worker_refuses_live_mode_because_no_real_provider_exists(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("INTEGRATIONS_MODE", "live")
    get_settings.cache_clear()
    try:
        assert worker.main(["--once"]) == 2
        assert "no real" in capsys.readouterr().err
    finally:
        get_settings.cache_clear()


def test_a_mock_refuses_a_reused_key_with_different_content() -> None:
    from supportx.integrations.ports import PermanentProviderError

    payments = MockPaymentProvider()
    payments.refund("k", "ORD-1", 500)
    assert payments.refund("k", "ORD-1", 500).is_mock  # same content: harmless replay
    with pytest.raises(PermanentProviderError):
        payments.refund("k", "ORD-1", 900)  # a real provider would refuse this too
    assert payments.refunded == [("ORD-1", 500)]
    stock = MockInventoryService()
    notifier = MockNotifier()
    from supportx.integrations.ports import StockLine

    stock.release("s", "ORD-1", [StockLine("A", 1)])
    notifier.send("n", 1, "t", {"a": "1"})
    with pytest.raises(PermanentProviderError):
        stock.release("s", "ORD-1", [StockLine("A", 2)])
    with pytest.raises(PermanentProviderError):
        notifier.send("n", 1, "t", {"a": "2"})


def test_re_enqueueing_a_key_with_a_different_payload_keeps_the_first_event(
    engine: Engine, caplog: pytest.LogCaptureFixture
) -> None:
    with session_scope(engine) as session:
        first = enqueue(session, PAYMENT_REFUND, "dup", {"order_number": "A", "amount_cents": 100})
        first_id = first.id
    with session_scope(engine) as session:
        again = enqueue(session, PAYMENT_REFUND, "dup", {"order_number": "A", "amount_cents": 999})
        assert again.id == first_id and json.loads(again.payload)["amount_cents"] == 100
    assert "different payload" in caplog.text


def test_outbox_stats_for_monitoring(engine: Engine, providers: Providers) -> None:
    client = make_client(engine)
    cancel(engine)
    assert client.get("/staff/outbox/stats").status_code == 401
    stats = client.get("/staff/outbox/stats", headers=JANE).json()
    assert stats["counts"]["pending"] == 3 and not stats["needs_attention"]
    assert stats["oldest_pending_age_seconds"] is not None

    assert isinstance(providers.payments, MockPaymentProvider)
    providers.payments.fail_permanently()
    OutboxDispatcher(engine, providers).run_once(NOW)
    stats = client.get("/staff/outbox/stats", headers=JANE).json()
    assert stats["counts"]["dead"] == 1 and stats["counts"]["sent"] == 2
    assert stats["needs_attention"] is True
