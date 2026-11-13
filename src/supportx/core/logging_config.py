"""Central logging setup. Call setup_logging() once at application start-up."""

import logging

_HANDLER_NAME = "supportx-console"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger. Safe to call many times (no duplicate handlers)."""
    root = logging.getLogger()
    root.setLevel(level)

    for handler in root.handlers:
        if handler.get_name() == _HANDLER_NAME:
            handler.setLevel(level)
            return

    handler = logging.StreamHandler()
    handler.set_name(_HANDLER_NAME)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger. Use `get_logger(__name__)` at the top of each module."""
    return logging.getLogger(name)
