"""Request ids, access log, JSON log format and model fingerprint."""

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from supportx.api import create_app
from supportx.core import Settings, get_logger, setup_logging
from supportx.core.logging_config import request_id_var
from supportx.core.model_info import describe_model_file
from supportx.db import init_db

STAFF = {"X-API-Key": "k"}


def _client(**overrides: object) -> TestClient:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    init_db(engine)
    values: dict[str, object] = {"staff_api_key": SecretStr("k"), **overrides}
    settings = Settings(_env_file=None, **values)  # type: ignore[call-arg,arg-type]
    return TestClient(create_app(settings, engine))


def test_response_carries_a_generated_request_id() -> None:
    first = _client().get("/health").headers["X-Request-ID"]
    second = _client().get("/health").headers["X-Request-ID"]
    assert len(first) == 32 and first != second


def test_harmless_supplied_request_id_is_reused() -> None:
    response = _client().get("/health", headers={"X-Request-ID": "abc-123.x"})
    assert response.headers["X-Request-ID"] == "abc-123.x"


def test_unsafe_supplied_request_id_is_replaced() -> None:
    bad = "x\r\nSet-Cookie: a=b" if False else "has space and ; chars"
    response = _client().get("/health", headers={"X-Request-ID": bad})
    assert response.headers["X-Request-ID"] != bad
    assert len(response.headers["X-Request-ID"]) == 32


def test_access_log_has_no_query_string_or_secrets(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="supportx.access"):
        _client().get("/health?email=private@example.com", headers=STAFF)
    lines = [r.getMessage() for r in caplog.records if r.name == "supportx.access"]
    assert len(lines) == 1
    assert lines[0].startswith("GET /health -> 200")
    assert "private@example.com" not in lines[0]


def test_log_lines_include_the_request_id(capsys: pytest.CaptureFixture[str]) -> None:
    client = _client()
    client.get("/health", headers={"X-Request-ID": "trace-42"})
    assert "trace-42" in capsys.readouterr().err
    assert request_id_var.get() == "-"  # reset after the request


def test_json_log_format(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO", "json")
    token = request_id_var.set("rid-1")
    try:
        get_logger("supportx.demo").info("hello %s", "world")
    finally:
        request_id_var.reset(token)
    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["message"] == "hello world"
    assert record["level"] == "INFO"
    assert record["request_id"] == "rid-1"


def test_describe_model_file_is_a_content_fingerprint(tmp_path: Path) -> None:
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"model-bytes")
    b.write_bytes(b"model-bytes")
    info = describe_model_file(a)
    assert info["file"] == "a.bin" and info["size_bytes"] == 11
    assert info["sha256"] == describe_model_file(b)["sha256"]  # same bytes, same identity
    b.write_bytes(b"other")
    assert info["sha256"] != describe_model_file(b)["sha256"]


def test_staff_model_info_needs_key_and_reports_no_model_when_disabled() -> None:
    client = _client()
    assert client.get("/staff/model-info").status_code in (401, 403)
    body = client.get("/staff/model-info", headers=STAFF).json()
    assert body["intent_model"] is None
    assert body["app_env"] == "development"


def test_demo_page_is_served_in_development_but_not_in_production() -> None:
    page = _client(app_env="development").get("/demo")
    assert page.status_code == 200 and "SupportX AI" in page.text
    assert "text/html" in page.headers["content-type"]
    assert _client(app_env="production").get("/demo").status_code == 404
