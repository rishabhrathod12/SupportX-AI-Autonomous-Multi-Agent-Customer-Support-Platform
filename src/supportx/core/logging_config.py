"""Central logging setup. Call setup_logging() once at application start-up.

Every log record carries the current request id (set by the API middleware; "-" outside a
request). The format is human-readable text by default, or one JSON object per line
(LOG_FORMAT=json) for log collectors.
"""

import json
import logging
from contextvars import ContextVar
from typing import Literal

LogFormat = Literal["text", "json"]

_HANDLER_NAME = "supportx-console"
_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(request_id)s | %(message)s"

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Copy the current request id onto each record so formatters can print it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Exception text is included; nothing else is added."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _formatter(fmt: LogFormat) -> logging.Formatter:
    return JsonFormatter() if fmt == "json" else logging.Formatter(_TEXT_FORMAT)


def setup_logging(level: str = "INFO", fmt: LogFormat = "text") -> None:
    """Configure the root logger. Safe to call many times (no duplicate handlers)."""
    root = logging.getLogger()
    root.setLevel(level)

    for handler in root.handlers:
        if handler.get_name() == _HANDLER_NAME:
            handler.setLevel(level)
            handler.setFormatter(_formatter(fmt))
            return

    handler = logging.StreamHandler()
    handler.set_name(_HANDLER_NAME)
    handler.setLevel(level)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(_formatter(fmt))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger. Use `get_logger(__name__)` at the top of each module."""
    return logging.getLogger(name)
