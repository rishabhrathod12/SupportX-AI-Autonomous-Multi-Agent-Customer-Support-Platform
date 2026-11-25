"""Metrics, trace logging and the error hook. The ticket fixtures are code-path data only."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from supportx.agents import TicketService
from supportx.api.app import create_app
from supportx.core import Settings
from supportx.core.metrics import METRICS, Metrics, register_error_hook, report_error
from supportx.rag import Retriever
from tests.unit.helpers import EMAIL, POLICIES, make_engine, make_service, sqlite_saver


@pytest.fixture(autouse=True)
def clean_metrics() -> None:
    METRICS.reset()


def test_counters_labels_and_observations() -> None:
    m = Metrics()
    m.inc("a")
    m.inc("a", 2)
    m.inc("b", reason="x")
    m.observe("s", 0.5)
    m.observe("s", 1.5)
    snap = m.snapshot()
    assert snap["counters"] == {"a": 3, "b{reason=x}": 1}
    assert snap["observed"]["s"] == {"count": 2, "mean": 1.0, "min": 0.5, "max": 1.5}


def test_error_hooks_receive_errors_and_cannot_break_the_caller() -> None:
    seen: list[tuple[str, dict[str, str]]] = []
    register_error_hook(lambda exc, ctx: seen.append((type(exc).__name__, ctx)))

    def broken(_exc: BaseException, _ctx: dict[str, str]) -> None:
        raise RuntimeError("tracker down")

    register_error_hook(broken)
    report_error(ValueError("x"), route="/r")
    assert ("ValueError", {"route": "/r"}) in seen
    assert METRICS.snapshot()["counters"]["errors.unexpected{kind=ValueError}"] == 1


def test_tickets_and_retrieval_are_counted_and_traced(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    saver, connection = sqlite_saver(tmp_path / "c.db")
    caplog.set_level("INFO")
    try:
        retriever = Retriever.from_documents(POLICIES)
        service: TicketService = make_service(make_engine(), retriever, saver)
        service.submit("What is your return policy? This is private text 4242", EMAIL)
        service.submit("My mug arrived damaged, I want a refund", EMAIL, "ORD-1001")
    finally:
        connection.close()
    counters = METRICS.snapshot()["counters"]
    assert counters["tickets.recorded{status=completed}"] == 1
    assert counters["tickets.recorded{status=awaiting_approval}"] == 1
    assert counters.get("retrieval.answered", 0) + counters.get("retrieval.abstained", 0) >= 1
    trace_lines = [r.getMessage() for r in caplog.records if "trace=" in r.getMessage()]
    assert len(trace_lines) == 2 and "understand" in trace_lines[0]
    assert "private text" not in caplog.text and "4242" not in caplog.text  # no message text


def test_staff_metrics_endpoint_requires_a_key() -> None:
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, app_env="development", staff_api_key="k" * 20  # type: ignore[arg-type]
    )
    client = TestClient(create_app(settings, engine=make_engine()))
    METRICS.inc("demo.counter")
    assert client.get("/staff/metrics").status_code == 401
    body = client.get("/staff/metrics", headers={"X-API-Key": "k" * 20}).json()
    assert body["counters"]["demo.counter"] == 1
