"""Application settings, loaded from environment variables and an optional .env file."""

import re
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from supportx.core.exceptions import ConfigurationError

AppEnv = Literal["development", "testing", "production"]
_STAFF_NAME = re.compile(r"^[A-Za-z0-9._-]{1,60}$")
MIN_NAMED_KEY_LENGTH = 16
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LogFormat = Literal["text", "json"]


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
    log_format: LogFormat = "text"  # "json" = one JSON object per log line

    # Secrets use SecretStr so they print as '**********' in logs and repr().
    openai_api_key: SecretStr | None = None
    openai_chat_model: str | None = None
    openai_embedding_model: str | None = None

    database_url: SecretStr | None = None  # contains a password, so also secret

    # Shared key protecting the staff endpoints (X-API-Key header). Unset = those endpoints are off.
    staff_api_key: SecretStr | None = None
    # Per-person keys, "jane:KEY1,bob:KEY2" (each key at least 16 characters). The approver
    # recorded on a decision is then the name that goes with the key, not what the caller types.
    staff_api_keys: SecretStr | None = None

    # Per-client limits over a sliding minute, 0 = unlimited (caveats: supportx.api.ratelimit).
    rate_limit_per_minute: int = Field(default=120, ge=0)
    auth_failures_per_minute: int = Field(default=10, ge=0)  # wrong staff keys before a lockout

    # MLflow 3 deprecates the './mlruns' file store; SQLite is its recommended local backend.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    # Ticket workflow: where paused tickets are saved, the intent model, the policy documents.
    checkpoint_db_path: str = "supportx_checkpoints.db"
    intent_model_path: str = "artifacts/models/intent_tfidf_lr_bitext_aug.joblib"
    policy_docs_dir: str = "data/sample/policies"

    # Business rules. THE DEFAULTS ARE DEMO NUMBERS, not a real policy: set the real ones and
    # then BUSINESS_RULES_CONFIRMED=true (production start-up warns while it is false).
    business_rules_version: str = Field(default="demo-1", min_length=1, max_length=40)
    refund_return_window_days: int = Field(default=30, gt=0, le=3650)  # days after delivery
    refund_not_received_after_days: int = Field(default=14, gt=0, le=3650)  # days after shipping
    business_rules_confirmed: bool = False

    # Payment / inventory / notification follow-ups (transactional outbox). "disabled": events are
    # saved but nothing delivers them. "mock": the worker uses MOCK adapters (nothing real happens).
    integrations_mode: Literal["disabled", "mock"] = "disabled"
    outbox_max_attempts: int = Field(default=5, ge=1, le=50)

    # A policy answer needs its best passage to score at least this (cosine, 0-1); else the ticket
    # goes to a person ("no_policy_match"). 0.1 answers almost everything, including questions the
    # documents cannot answer; higher values hand more to people. Trade-off: docs/EXPERIMENTS.md E9.
    retrieval_min_score: float = Field(default=0.1, ge=0.0, le=1.0)

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

    @model_validator(mode="after")
    def _check_staff_keys(self) -> "Settings":
        self.staff_identities()  # fail at start-up, not on the first staff request
        return self

    def staff_identities(self) -> list[tuple[str, str | None]]:
        """Every accepted staff key as (key, name). The shared key has no name (None).

        Raises ConfigurationError for a malformed STAFF_API_KEYS (bad name, short or duplicate
        key, duplicate name, or a named key equal to the shared key).
        """
        found: list[tuple[str, str | None]] = []
        raw = self.staff_api_keys.get_secret_value() if self.staff_api_keys else ""
        names: set[str] = set()
        for entry in (e.strip() for e in raw.split(",") if e.strip()):
            name, sep, key = entry.partition(":")
            name, key = name.strip(), key.strip()
            if not sep or not _STAFF_NAME.match(name):
                raise ConfigurationError(
                    "STAFF_API_KEYS entries must look like name:key, where the name uses only "
                    "letters, digits, '.', '_' or '-'."
                )
            if len(key) < MIN_NAMED_KEY_LENGTH:
                raise ConfigurationError(
                    f"The key for staff member '{name}' is shorter than "
                    f"{MIN_NAMED_KEY_LENGTH} characters."
                )
            if name in names:
                raise ConfigurationError(f"STAFF_API_KEYS lists '{name}' twice.")
            names.add(name)
            found.append((key, name))
        if self.staff_api_key is not None:
            found.append((self.staff_api_key.get_secret_value(), None))
        if len({key for key, _ in found}) != len(found):
            raise ConfigurationError("Two staff keys are identical; every key must be unique.")
        return found

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
