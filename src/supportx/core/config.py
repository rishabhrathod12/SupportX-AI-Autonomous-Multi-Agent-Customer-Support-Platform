"""Application settings, loaded from environment variables and an optional .env file."""

from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from supportx.core.exceptions import ConfigurationError

AppEnv = Literal["development", "testing", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Typed, validated configuration. Field names map to env vars (case-insensitive)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # `OPENAI_API_KEY=` (blank) is treated as "not set"
        extra="ignore",
    )

    app_env: AppEnv = "development"
    log_level: LogLevel = "INFO"

    # Secrets use SecretStr so they print as '**********' in logs and repr().
    openai_api_key: SecretStr | None = None
    openai_chat_model: str | None = None
    openai_embedding_model: str | None = None

    database_url: SecretStr | None = None  # contains a password, so also secret

    # Shared key protecting the staff endpoints (X-API-Key header). Unset = those endpoints are off.
    staff_api_key: SecretStr | None = None

    # MLflow 3 deprecates the './mlruns' file store; SQLite is its recommended local backend.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    # Below this intent confidence a ticket goes to a human instead of being answered automatically.
    # 0.6 (not 0.4): measured on out-of-domain messages it catches many more confidently-wrong cases
    # for about 1% extra escalations on in-domain validation data (see docs/EXPERIMENTS.md, E5).
    intent_min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)

    @field_validator("log_level", "app_env", mode="before")
    @classmethod
    def _normalize_case(cls, value: Any, info: Any) -> Any:
        """Accept 'info' for LOG_LEVEL and 'Production' for APP_ENV."""
        if not isinstance(value, str):
            return value
        return value.upper() if info.field_name == "log_level" else value.lower()

    def require_openai_api_key(self) -> str:
        """Return the OpenAI key, or raise a clear error if it is not configured."""
        if self.openai_api_key is None:
            raise ConfigurationError("OPENAI_API_KEY is not set. Add it to your .env file.")
        return self.openai_api_key.get_secret_value()

    def require_database_url(self) -> str:
        """Return the database URL, or raise a clear error if it is not configured."""
        if self.database_url is None:
            raise ConfigurationError("DATABASE_URL is not set. Add it to your .env file.")
        return self.database_url.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Create Settings once and reuse it (cached). Tests can call get_settings.cache_clear()."""
    return Settings()
