"""Thin wrapper around the OpenAI SDK: timeouts, retries, error mapping, usage and cost tracking.

The wrapper only produces TEXT and VECTORS. It has no access to the database or to any
business operation, so a model can never refund, cancel or change an account by itself.

Logs contain model names, token counts and error class names, never prompts, replies or keys.
"""

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import openai

from supportx.core import ConfigurationError, ExternalServiceError, Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pricing:
    """USD per one million tokens. Supplied by you in settings: no prices are built in,
    because they change and a wrong default would give false cost numbers."""

    input_per_mtok: float
    output_per_mtok: float = 0.0


@dataclass(frozen=True)
class UsageSnapshot:
    calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float | None  # None = no pricing configured, so the cost is unknown


class UsageTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls = 0
        self._prompt = 0
        self._completion = 0
        self._cost = 0.0
        self._cost_known = True

    def add(self, prompt: int, completion: int, pricing: Pricing | None) -> None:
        with self._lock:
            self._calls += 1
            self._prompt += prompt
            self._completion += completion
            if pricing is None:
                self._cost_known = False
            else:
                self._cost += (
                    prompt * pricing.input_per_mtok + completion * pricing.output_per_mtok
                ) / 1_000_000

    def snapshot(self) -> UsageSnapshot:
        with self._lock:
            cost = round(self._cost, 6) if self._cost_known else None
            return UsageSnapshot(self._calls, self._prompt, self._completion, cost)


class LLMClient:
    """Chat completions and embeddings with uniform error handling.

    `sdk` is an `openai.OpenAI`-compatible object; tests pass a fake. Retries with backoff
    for rate limits, timeouts and 5xx errors are done by the SDK (`max_retries`).
    """

    def __init__(
        self,
        sdk: Any,
        chat_model: str | None,
        embedding_model: str | None,
        chat_pricing: Pricing | None = None,
        embedding_pricing: Pricing | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._sdk = sdk
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._chat_pricing = chat_pricing
        self._embedding_pricing = embedding_pricing
        self._timeout = timeout_seconds
        self.usage = UsageTracker()

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMClient":
        sdk = openai.OpenAI(
            api_key=settings.require_openai_api_key(),
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
        chat = (
            Pricing(settings.openai_chat_input_per_mtok, settings.openai_chat_output_per_mtok)
            if settings.openai_chat_input_per_mtok is not None
            else None
        )
        emb = (
            Pricing(settings.openai_embedding_per_mtok)
            if settings.openai_embedding_per_mtok is not None
            else None
        )
        return cls(
            sdk,
            settings.openai_chat_model,
            settings.openai_embedding_model,
            chat,
            emb,
            settings.openai_timeout_seconds,
        )

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        if not self._chat_model:
            raise ConfigurationError("OPENAI_CHAT_MODEL is not set.")
        response = self._call(
            lambda: self._sdk.chat.completions.create(
                model=self._chat_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=max_tokens,
                temperature=0,
                timeout=self._timeout,
            )
        )
        usage = response.usage
        self.usage.add(
            getattr(usage, "prompt_tokens", 0),
            getattr(usage, "completion_tokens", 0),
            self._chat_pricing,
        )
        logger.info("LLM chat call ok model=%s", self._chat_model)
        choices = response.choices
        text = choices[0].message.content if choices else None
        if not text or not str(text).strip():
            raise ExternalServiceError("The language model returned an empty answer.")
        return str(text).strip()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Float32 rows, L2-normalized (zero rows stay zero), as the retriever expects."""
        if not self._embedding_model:
            raise ConfigurationError("OPENAI_EMBEDDING_MODEL is not set.")
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        response = self._call(
            lambda: self._sdk.embeddings.create(
                model=self._embedding_model, input=list(texts), timeout=self._timeout
            )
        )
        vectors = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ExternalServiceError("The embedding service returned an unexpected result.")
        self.usage.add(getattr(response.usage, "prompt_tokens", 0), 0, self._embedding_pricing)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized: np.ndarray = np.divide(
            vectors, norms, out=np.zeros_like(vectors), where=norms > 0
        )
        return normalized

    def _call(self, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except openai.RateLimitError as exc:
            logger.warning("OpenAI rate limit after retries")
            raise ExternalServiceError("The language service is busy; try again later.") from exc
        except openai.APITimeoutError as exc:
            logger.warning("OpenAI timeout after retries")
            raise ExternalServiceError("The language service timed out.") from exc
        except openai.AuthenticationError as exc:
            logger.error("OpenAI rejected the API key")
            raise ExternalServiceError("The language service is not authorized.") from exc
        except openai.OpenAIError as exc:  # connection, status, bad request, ...
            logger.error("OpenAI call failed: %s", type(exc).__name__)
            raise ExternalServiceError("The language service call failed.") from exc
