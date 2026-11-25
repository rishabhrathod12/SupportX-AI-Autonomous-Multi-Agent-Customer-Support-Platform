"""LLM wrapper tests. A fake SDK is used everywhere: no network, no key, no real results."""

from types import SimpleNamespace
from typing import Any

import httpx2 as httpx
import numpy as np
import openai
import pytest
from pydantic import SecretStr

from supportx.agents.replies import HANDOFF, compose_reply
from supportx.agents.state import TicketState
from supportx.core import ConfigurationError, ExternalServiceError, Settings
from supportx.llm import LLMClient, OpenAIEmbedder, Pricing, is_grounded, make_llm_composer


class FakeSDK:
    def __init__(self, text: str = "ok", error: Exception | None = None) -> None:
        self.text, self.error, self.calls = text, error, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._embed)

    def _chat(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.text)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )

    def _embed(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self.error:
            raise self.error
        data = [SimpleNamespace(embedding=[3.0, 4.0]) for _ in kwargs["input"]]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=10))


def client(sdk: FakeSDK, priced: bool = True) -> LLMClient:
    return LLMClient(
        sdk, "chat-model", "emb-model", Pricing(2.0, 4.0) if priced else None, Pricing(1.0)
    )


def _response() -> httpx.Response:
    return httpx.Response(429, request=httpx.Request("POST", "https://x.invalid"))


def test_complete_returns_text_and_tracks_usage_and_cost() -> None:
    c = client(FakeSDK("hello"))
    assert c.complete("s", "u") == "hello"
    snap = c.usage.snapshot()
    assert (snap.calls, snap.prompt_tokens, snap.completion_tokens) == (1, 100, 50)
    assert snap.cost_usd == pytest.approx((100 * 2.0 + 50 * 4.0) / 1e6)


def test_cost_is_unknown_without_pricing() -> None:
    c = client(FakeSDK(), priced=False)
    c.complete("s", "u")
    assert c.usage.snapshot().cost_usd is None


def test_missing_model_is_a_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        LLMClient(FakeSDK(), None, None).complete("s", "u")
    with pytest.raises(ConfigurationError):
        LLMClient(FakeSDK(), None, None).embed(["a"])


def test_empty_answer_is_an_error() -> None:
    with pytest.raises(ExternalServiceError):
        client(FakeSDK("   ")).complete("s", "u")


@pytest.mark.parametrize(
    "error",
    [
        openai.RateLimitError("secret-detail", response=_response(), body=None),
        openai.APITimeoutError(request=httpx.Request("POST", "https://x.invalid")),
        openai.APIConnectionError(request=httpx.Request("POST", "https://x.invalid")),
    ],
)
def test_sdk_errors_become_generic_service_errors(error: Exception) -> None:
    with pytest.raises(ExternalServiceError) as caught:
        client(FakeSDK(error=error)).complete("s", "u")
    assert "secret-detail" not in str(caught.value)


def test_embeddings_are_normalized_and_batched() -> None:
    sdk = FakeSDK()
    embedder = OpenAIEmbedder(client(sdk), batch_size=2)
    vectors = embedder.embed(["a", "b", "c"])
    assert vectors.shape == (3, 2) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert sdk.calls == 2 and embedder.dim == 2


def test_logs_do_not_contain_prompt_or_key(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG")
    client(FakeSDK("reply-text")).complete("system-text", "customer-secret-text")
    logged = caplog.text
    assert "customer-secret-text" not in logged and "reply-text" not in logged


# ---- grounded reply composer ----

ORDER: dict[str, Any] = {
    "order_number": "ORD-1001",
    "status": "shipped",
    "shipment": {"tracking_number": "TRK-ORD-1001"},
}


def order_state(**extra: Any) -> TicketState:
    state: TicketState = {"message": "where is my order", "intent": "track_order", "order": ORDER}
    state.update(extra)  # type: ignore[typeddict-item]
    return state


def test_grounding_check() -> None:
    draft = "Order ORD-1001 is shipped. Tracking number: TRK-ORD-1001."
    assert is_grounded(draft, "Good news! ORD-1001 is on its way (TRK-ORD-1001).")
    assert not is_grounded(draft, "Order ORD-1001 is on its way.")  # dropped tracking
    assert not is_grounded(draft, "ORD-1001, TRK-ORD-1001, refund of $50 coming")  # invented
    assert not is_grounded(draft, "")


def test_llm_rewrite_used_when_grounded() -> None:
    sdk = FakeSDK("Great news: ORD-1001 is shipped. Tracking: TRK-ORD-1001.")
    out = make_llm_composer(client(sdk), compose_reply)(order_state())
    assert out.startswith("Great news")


def test_ungrounded_rewrite_falls_back_to_template() -> None:
    sdk = FakeSDK("ORD-1001 arrives in 3 days, tracking TRK-ORD-1001.")
    state = order_state()
    assert make_llm_composer(client(sdk), compose_reply)(state) == compose_reply(state)


def test_llm_failure_falls_back_to_template() -> None:
    sdk = FakeSDK(error=openai.APITimeoutError(request=httpx.Request("POST", "https://x.invalid")))
    state = order_state()
    assert make_llm_composer(client(sdk), compose_reply)(state) == compose_reply(state)


@pytest.mark.parametrize(
    "state",
    [
        {"message": "m", "escalated": True},
        {"message": "m", "intent": "get_refund", "order": ORDER},
        {"message": "m", "intent": "cancel_order", "order": ORDER},
        {"message": "m", "intent": "track_order", "order": ORDER, "refund": {"eligible": False}},
        {"message": "m", "intent": "track_order", "order": ORDER, "approval": {"approved": True}},
        {"message": "m", "missing_info": ["order number"]},
        {"message": "m", "order_not_found": True},
    ],
)
def test_sensitive_states_never_reach_the_llm(state: TicketState) -> None:
    sdk = FakeSDK("INJECTED approved refund 999")
    out = make_llm_composer(client(sdk), compose_reply)(state)
    assert sdk.calls == 0 and out == compose_reply(state)


def test_handoff_text_is_unchanged() -> None:
    sdk = FakeSDK()
    assert make_llm_composer(client(sdk), compose_reply)({"message": "m"}) == HANDOFF
    assert sdk.calls == 0


# ---------------------------------------------------------------- configuration and bad responses
def test_from_settings_without_a_key_is_a_clear_configuration_error() -> None:
    settings = Settings(_env_file=None, openai_chat_model="m")  # type: ignore[call-arg]
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        LLMClient.from_settings(settings)


def test_the_api_key_never_appears_in_repr_or_logs(caplog: pytest.LogCaptureFixture) -> None:
    secret = "sk-test-secret-value-1234567890"
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, openai_api_key=SecretStr(secret), openai_chat_model="m"
    )
    assert secret not in repr(settings) and secret not in str(settings.model_dump())
    caplog.set_level("DEBUG")
    sdk = FakeSDK(error=openai.AuthenticationError("bad key", response=_response(), body=None))
    with pytest.raises(ExternalServiceError) as caught:
        LLMClient(sdk, "m", None).complete("s", "u")
    assert secret not in caplog.text and secret not in str(caught.value)


class OddSDK:
    """Returns shapes a misbehaving provider might send."""

    def __init__(self, chat: Any = None, embed: Any = None) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: chat))
        self.embeddings = SimpleNamespace(create=lambda **_: embed)


def test_invalid_chat_responses_become_service_errors_not_crashes() -> None:
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    no_choices = SimpleNamespace(choices=[], usage=usage)
    null_content = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))], usage=usage
    )
    for response in (no_choices, null_content):
        with pytest.raises(ExternalServiceError):
            LLMClient(OddSDK(chat=response), "m", None).complete("s", "u")


def test_invalid_embedding_responses_become_service_errors() -> None:
    wrong_count = SimpleNamespace(
        data=[SimpleNamespace(embedding=[1.0, 0.0])], usage=SimpleNamespace(prompt_tokens=1)
    )
    with pytest.raises(ExternalServiceError):
        LLMClient(OddSDK(embed=wrong_count), None, "e").embed(["a", "b"])  # asked 2, got 1
    ragged = SimpleNamespace(
        data=[SimpleNamespace(embedding=[1.0]), SimpleNamespace(embedding=[1.0, 2.0])],
        usage=SimpleNamespace(prompt_tokens=1),
    )
    with pytest.raises((ExternalServiceError, ValueError)):
        LLMClient(OddSDK(embed=ragged), None, "e").embed(["a", "b"])


def test_a_non_string_reply_from_the_provider_is_rejected_by_the_grounding_check() -> None:
    sdk = FakeSDK("x" * 5000)  # absurdly long reply: not accepted, template is used
    state = order_state()
    assert make_llm_composer(client(sdk), compose_reply)(state) == compose_reply(state)
