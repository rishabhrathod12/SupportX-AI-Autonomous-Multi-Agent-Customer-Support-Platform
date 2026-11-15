import pytest
from pydantic import ValidationError

from supportx.core import ConfigurationError, Settings, get_settings

ENV_KEYS = [
    "APP_ENV", "LOG_LEVEL", "OPENAI_API_KEY", "OPENAI_CHAT_MODEL",
    "OPENAI_EMBEDDING_MODEL", "DATABASE_URL", "MLFLOW_TRACKING_URI",
]  # fmt: skip


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test with none of our env vars set, so a real shell/.env can't leak in."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def make_settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]  # ignore any real .env file


def test_defaults() -> None:
    s = make_settings()
    assert s.app_env == "development"
    assert s.log_level == "INFO"
    assert s.openai_api_key is None
    assert s.mlflow_tracking_uri == "sqlite:///mlflow.db"


def test_reads_environment_and_normalizes_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("APP_ENV", "Production")
    s = make_settings()
    assert s.log_level == "DEBUG"
    assert s.app_env == "production"


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(ValidationError):
        make_settings()


def test_blank_value_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert make_settings().openai_api_key is None


def test_secret_is_hidden_but_retrievable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-test-key")
    s = make_settings()
    assert "dummy-test-key" not in repr(s)
    assert "dummy-test-key" not in str(s.openai_api_key)
    assert s.require_openai_api_key() == "dummy-test-key"


def test_require_methods_raise_when_missing() -> None:
    s = make_settings()
    with pytest.raises(ConfigurationError):
        s.require_openai_api_key()
    with pytest.raises(ConfigurationError):
        s.require_database_url()


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()
