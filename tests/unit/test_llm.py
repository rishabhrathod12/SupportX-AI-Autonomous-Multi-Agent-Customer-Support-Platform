"""LLM wrapper tests. A fake SDK is used everywhere: no network, no key, no real results."""

from types import SimpleNamespace
from typing import Any

import httpx2 as httpx
import numpy as np
import openai
import pytest

from supportx.agents.replies import HANDOFF, compose_reply
from supportx.agents.state import TicketState
from supportx.core import ConfigurationError, ExternalServiceError
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
