from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from supportx.api import create_app
from supportx.api.ratelimit import SlidingWindowLimiter
from supportx.core import Settings
from supportx.db import init_db


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------- the limiter itself
def test_allows_up_to_the_limit_then_reports_a_wait() -> None:
    clock = FakeClock()
    limiter = SlidingWindowLimiter(3, window=60, clock=clock)
    for _ in range(3):
        assert limiter.retry_after("a") == 0
        limiter.record("a")
    wait = limiter.retry_after("a")
    assert 1 <= wait <= 61
    assert limiter.retry_after("b") == 0  # other clients are unaffected


def test_the_window_slides() -> None:
    clock = FakeClock()
    limiter = SlidingWindowLimiter(2, window=60, clock=clock)
    limiter.record("a")
    clock.now += 30
    limiter.record("a")
    assert limiter.retry_after("a") > 0
    clock.now += 31  # the first event is now older than the window
    assert limiter.retry_after("a") == 0


def test_zero_means_unlimited_and_idle_clients_are_forgotten() -> None:
    unlimited = SlidingWindowLimiter(0)
    for _ in range(1000):
        unlimited.record("a")
    assert unlimited.retry_after("a") == 0 and unlimited._events == {}

    clock = FakeClock()
    limiter = SlidingWindowLimiter(5, window=60, clock=clock)
    limiter.record("old")
    clock.now += 120
    limiter._forget_idle()
    assert "old" not in limiter._events


# ---------------------------------------------------------------- through the API
def _client(**overrides: object) -> TestClient:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    init_db(engine)
    values: dict[str, object] = {"staff_api_key": SecretStr("k"), **overrides}
    settings = Settings(_env_file=None, **values)  # type: ignore[call-arg,arg-type]
    return TestClient(create_app(settings, engine))


def test_too_many_requests_get_429_with_retry_after() -> None:
    client = _client(rate_limit_per_minute=3)
    body = {"message": "hello", "customer_email": "a@example.com"}
    codes = [
        client.post(
            "/tickets/status", json={"ticket_id": "x", "customer_email": "a@b.c"}
        ).status_code
        for _ in range(5)
    ]
    assert 429 not in codes[:3] and codes[3:] == [429, 429]  # 503: no ticket service in this app
    limited = client.post("/tickets/status", json=body)
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1
    assert "X-Request-ID" in limited.headers  # still logged and traceable


def test_health_checks_are_never_limited() -> None:
    client = _client(rate_limit_per_minute=1)
    assert all(client.get("/health").status_code == 200 for _ in range(10))
    assert client.get("/health/ready").status_code == 200


def test_repeated_wrong_staff_keys_lock_that_client_out_even_with_the_right_key() -> None:
    client = _client(auth_failures_per_minute=3)
    bad = {"X-API-Key": "wrong"}
    assert [client.get("/staff/refunds/pending", headers=bad).status_code for _ in range(3)] == [
        401
    ] * 3
    locked = client.get("/staff/refunds/pending", headers={"X-API-Key": "k"})
    assert locked.status_code == 429 and "Retry-After" in locked.headers
    assert client.get("/health").status_code == 200  # customers' endpoints and probes still work


def test_correct_staff_key_is_not_counted_as_a_failure() -> None:
    client = _client(auth_failures_per_minute=2)
    for _ in range(6):
        assert client.get("/staff/refunds/pending", headers={"X-API-Key": "k"}).status_code == 200
