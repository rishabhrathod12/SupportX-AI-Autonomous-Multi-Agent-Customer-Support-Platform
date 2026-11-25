import pytest
from fastapi.testclient import TestClient

from supportx.api.app import create_app
from supportx.core import Settings
from tests.unit.helpers import make_engine


def client(**overrides: object) -> TestClient:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        app_env="development",
        staff_api_key="k" * 20,  # type: ignore[arg-type]
        **overrides,  # type: ignore[arg-type]
    )
    return TestClient(create_app(settings, engine=make_engine()))


def test_security_headers_on_normal_and_error_responses() -> None:
    c = client()
    for response in (c.get("/health"), c.get("/staff/refunds/pending")):  # 200 and 401
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["cache-control"] == "no-store"
        assert "strict-transport-security" not in response.headers


def test_hsts_only_when_enabled() -> None:
    headers = client(hsts_enabled=True).get("/health").headers
    assert "max-age" in headers["strict-transport-security"]


def test_oversized_body_is_rejected_with_413() -> None:
    c = client(max_request_bytes=100)
    response = c.post("/tickets", content="x" * 500, headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert response.headers["x-content-type-options"] == "nosniff"


def test_invalid_content_length_is_rejected() -> None:
    response = client().get("/health", headers={"content-length": "abc"})
    assert response.status_code == 400


def test_trusted_hosts() -> None:
    c = client(allowed_hosts="api.example.com")
    assert c.get("/health", headers={"host": "evil.example.net"}).status_code == 400
    assert c.get("/health", headers={"host": "api.example.com"}).status_code == 200


def test_failed_staff_login_is_logged_without_the_key(caplog: pytest.LogCaptureFixture) -> None:
    c = client()
    secret = "wrong-key-should-never-be-logged"
    caplog.set_level("WARNING")
    assert c.get("/staff/refunds/pending", headers={"x-api-key": secret}).status_code == 401
    assert "Staff authentication failed (wrong key)" in caplog.text
    assert secret not in caplog.text


def test_cross_origin_requests_get_no_cors_permission() -> None:
    """No CORS middleware is installed on purpose: browsers on other sites cannot read responses."""
    c = client()
    preflight = c.options(
        "/tickets",
        headers={"origin": "https://evil.example", "access-control-request-method": "POST"},
    )
    assert "access-control-allow-origin" not in preflight.headers
    response = c.get("/health", headers={"origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers
