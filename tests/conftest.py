"""Shared test fixtures."""

import logging
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def isolate_root_logger() -> Iterator[None]:
    """Undo logging changes after every test.

    create_app() and setup_logging() install a console handler on the ROOT logger. That handler
    would otherwise outlive the test (still pointing at its captured stderr) and break later tests
    that expect a fresh handler.
    """
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
