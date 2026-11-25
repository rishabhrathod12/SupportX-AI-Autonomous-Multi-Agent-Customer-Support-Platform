"""Sentiment tests. The tiny training sets below are SYNTHETIC and only exercise the code paths:
their scores say nothing about real customer sentiment."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from supportx.agents.stubs import stub_sentiment
from supportx.core import ModelError
from supportx.ml.sentiment import (
    LABELS,
    SentimentClassifier,
    make_sentiment_agent,
    prepare_splits,
    train_and_evaluate,
)

NEG = ["this is bad and slow", "very poor quality item", "I hate the broken box", "worst service"]
POS = ["love it great product", "excellent fast delivery", "very happy thanks", "perfect item"]
NEU = ["order arrived tuesday", "item is blue sized medium", "I ordered a kettle", "box is small"]


def toy_frame(n: int = 12) -> pd.DataFrame:
    rows = []
    for label, texts in (("negative", NEG), ("positive", POS), ("neutral", NEU)):
        for i in range(n):
            rows.append({"text": f"{texts[i % 4]} number {i}", "label": label})
    return pd.DataFrame(rows)


def test_rejects_unknown_labels() -> None:
    with pytest.raises(ModelError):
        SentimentClassifier().fit(["a", "b"], ["happy", "sad"])


def test_prepare_splits_has_no_overlap_and_checks_input() -> None:
    train, val, test = prepare_splits(toy_frame(30))
    sets = [set(x["text"]) for x in (train, val, test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert set(train["label"]) == set(LABELS)
    with pytest.raises(ModelError):
        prepare_splits(pd.DataFrame({"text": ["a"], "label": ["negative"]}))
    with pytest.raises(ModelError):
        prepare_splits(pd.DataFrame({"x": [1]}))


def test_train_evaluate_save_load_round_trip(tmp_path: Path) -> None:
    model, report = train_and_evaluate(toy_frame(30))
    assert "test" not in report  # the test split is only scored on request
    assert 0.0 <= report["validation"]["accuracy"] <= 1.0  # type: ignore[index]
    path = model.save(tmp_path / "s.joblib")
    loaded = SentimentClassifier.load(path)
    assert loaded.classes_ == list(LABELS)
    assert np.allclose(loaded.predict_proba(["love it"]), model.predict_proba(["love it"]))
    _, with_test = train_and_evaluate(toy_frame(30), score_test=True)
    assert "test" in with_test


class FakeModel(SentimentClassifier):
    def __init__(self, label: str, confidence: float) -> None:
        super().__init__()
        self._answer = (label, confidence)

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        return self._answer


def test_confident_model_label_is_used() -> None:
    agent = make_sentiment_agent(FakeModel("positive", 0.9), stub_sentiment)
    assert agent("thanks for the quick help") == ("positive", "low")


def test_low_confidence_falls_back_to_the_rules() -> None:
    agent = make_sentiment_agent(FakeModel("positive", 0.4), stub_sentiment)
    assert agent("where is my parcel")[0] == "neutral"  # rules: neutral, not the weak guess
    assert agent("this is unacceptable")[0] == "negative"


def test_mixed_message_with_negative_wording_is_negative() -> None:
    agent = make_sentiment_agent(FakeModel("positive", 0.95), stub_sentiment)
    assert agent("great product but the delivery was terrible")[0] == "negative"


def test_urgency_always_comes_from_the_rules() -> None:
    agent = make_sentiment_agent(FakeModel("negative", 0.95), stub_sentiment)
    assert agent("my order is late")[1] == "low"
    assert agent("I need this ASAP")[1] == "high"
