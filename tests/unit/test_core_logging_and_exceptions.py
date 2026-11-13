import logging
from collections.abc import Iterator

import pytest

from supportx.core import (
    ConfigurationError,
    ExternalServiceError,
    SupportXError,
    get_logger,
    setup_logging,
)


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def test_setup_logging_is_idempotent() -> None:
    setup_logging("INFO")
    setup_logging("DEBUG")
    ours = [h for h in logging.getLogger().handlers if h.get_name() == "supportx-console"]
    assert len(ours) == 1
    assert logging.getLogger().level == logging.DEBUG


def test_get_logger_uses_given_name() -> None:
    assert get_logger("supportx.test").name == "supportx.test"


def test_logging_writes_formatted_line(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO")
    get_logger("supportx.demo").info("hello")
    err = capsys.readouterr().err
    assert "INFO" in err and "supportx.demo" in err and "hello" in err


def test_all_custom_errors_share_one_base() -> None:
    for exc in (ConfigurationError, ExternalServiceError):
        assert issubclass(exc, SupportXError)
    with pytest.raises(SupportXError):
        raise ConfigurationError("boom")
