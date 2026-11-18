"""Per-client rate limiting (in memory, per process).

Two budgets, both counted per client address over a sliding window:
  * requests: every endpoint except /health*, so one client cannot hammer the API or guess
    customer emails / order numbers quickly;
  * auth failures: wrong or missing staff keys on /staff/*; once used up, that client is blocked from
    /staff/* (even with the right key) until the window passes, which makes guessing the key impractical.

LIMITS (read before relying on it): the counters live in this process, so with several server
processes each has its own budget, and a restart clears them. Behind a reverse proxy every client
looks like the proxy's address unless the proxy is trusted to supply the real one; this module does
NOT read X-Forwarded-For (it can be forged). For production, also limit at the proxy or gateway.
"""

import time
from collections import deque
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from supportx.core import Settings, get_logger

logger = get_logger("supportx.ratelimit")

_MAX_TRACKED_CLIENTS = 10_000


class SlidingWindowLimiter:
    """At most `limit` events per `window` seconds per key. limit <= 0 means unlimited."""

    def __init__(
        self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window = window
        self._clock = clock
        self._events: dict[str, deque[float]] = {}

    def _trim(self, key: str) -> deque[float]:
        events = self._events.setdefault(key, deque())
        cutoff = self._clock() - self.window
        while events and events[0] <= cutoff:
            events.popleft()
        return events

    def retry_after(self, key: str) -> int:
        """Seconds until `key` may act again; 0 when it may act now."""
        if self.limit <= 0:
            return 0
        events = self._trim(key)
        if len(events) < self.limit:
            return 0
        return max(1, int(events[0] + self.window - self._clock()) + 1)

    def record(self, key: str) -> None:
        """Count one event for `key`."""
        if self.limit <= 0:
            return
        if len(self._events) >= _MAX_TRACKED_CLIENTS:
            self._forget_idle()
        self._trim(key).append(self._clock())

    def _forget_idle(self) -> None:
        """Bound memory: drop clients with no events left in the window."""
        for key in list(self._events):
            if not self._trim(key):
                del self._events[key]


def _too_many(seconds: int) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please try again later."},
        headers={"Retry-After": str(seconds)},
    )


def add_rate_limiting(app: FastAPI, settings: Settings) -> None:
    requests = SlidingWindowLimiter(settings.rate_limit_per_minute)
    auth_failures = SlidingWindowLimiter(settings.auth_failures_per_minute)

    @app.middleware("http")
    async def rate_limit(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        if path.startswith("/health"):
            return await call_next(request)  # probes must always get through
        client = request.client.host if request.client else "unknown"
        is_staff = path.startswith("/staff")

        wait = requests.retry_after(client)
        if is_staff:
            wait = max(wait, auth_failures.retry_after(client))
        if wait:
            logger.warning("Rate limit hit by a client on %s (retry in %ds)", path, wait)
            return _too_many(wait)

        requests.record(client)
        response = await call_next(request)
        if is_staff and response.status_code in (401, 403):
            auth_failures.record(client)
        return response
