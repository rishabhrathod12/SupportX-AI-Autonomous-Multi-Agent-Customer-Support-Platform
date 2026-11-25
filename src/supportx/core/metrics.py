"""Tiny in-process metrics registry and error hook. No external dependency or service.

LIMITS: counters live in this process only (a restart resets them and several server processes
each have their own). For fleet-wide dashboards export them to a real metrics system; this module
exists so operators can see what the app is doing without adding infrastructure first.
Never put customer text, emails or keys in a metric name or label.
"""

import threading
from collections.abc import Callable
from typing import Any

from supportx.core.logging_config import get_logger

logger = get_logger(__name__)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._observed: dict[str, list[float]] = {}  # name -> [count, total, min, max]

    @staticmethod
    def _key(name: str, labels: dict[str, str]) -> str:
        if not labels:
            return name
        return name + "{" + ",".join(f"{k}={v}" for k, v in sorted(labels.items())) + "}"

    def inc(self, name: str, amount: float = 1, **labels: str) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + amount

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            stats = self._observed.get(name)
            if stats is None:
                self._observed[name] = [1, value, value, value]
            else:
                stats[0] += 1
                stats[1] += value
                stats[2] = min(stats[2], value)
                stats[3] = max(stats[3], value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            observed = {
                name: {"count": int(s[0]), "mean": round(s[1] / s[0], 4), "min": s[2], "max": s[3]}
                for name, s in self._observed.items()
            }
            return {"counters": dict(sorted(self._counters.items())), "observed": observed}

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._observed.clear()


METRICS = Metrics()

_error_hooks: list[Callable[[BaseException, dict[str, str]], None]] = []


def register_error_hook(hook: Callable[[BaseException, dict[str, str]], None]) -> None:
    """Call `hook(exception, context)` for unexpected errors (e.g. forward to Sentry). Context holds
    short non-sensitive strings only (route, request id). A hook that raises is ignored."""
    _error_hooks.append(hook)


def report_error(exc: BaseException, **context: str) -> None:
    METRICS.inc("errors.unexpected", kind=type(exc).__name__)
    for hook in _error_hooks:
        try:
            hook(exc, context)
        except Exception:  # an error tracker must never break request handling
            logger.exception("Error hook failed")
