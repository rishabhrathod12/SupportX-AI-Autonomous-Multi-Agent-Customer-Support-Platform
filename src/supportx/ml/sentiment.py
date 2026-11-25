"""Sentiment (negative / neutral / positive) for customer messages.

The model class is the same TF-IDF + Logistic Regression pipeline as the intent classifier, with
the label set restricted. NO TRAINED MODEL IS SHIPPED: there is no verified, licensed, in-domain
labelled data yet (see docs/DATASETS.md). Until one is trained and evaluated on suitable data,
the serving path keeps the keyword rules.

Training data format: a CSV with `text` and `label` columns. Product-review star ratings are a
WEAK and different-domain proxy for support-message sentiment; report that with any score.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from supportx.core import ModelError, get_logger
from supportx.ml.evaluate import evaluate_classification, per_class_report
from supportx.ml.intent import IntentClassifier
from supportx.nlp import clean_text

logger = get_logger(__name__)

LABELS = ("negative", "neutral", "positive")


class SentimentClassifier(IntentClassifier):
    """Three-class sentiment model (inherits fit/predict/predict_proba/save/load)."""

    def __init__(self, c: float = 1.0, seed: int = 42) -> None:
        super().__init__(model="logreg", c=c, seed=seed, class_weight="balanced")

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "SentimentClassifier":
        unknown = sorted(set(labels) - set(LABELS))
        if unknown:
            raise ModelError(f"Unknown sentiment labels {unknown}; use {list(LABELS)}.")
        super().fit(texts, labels)
        return self


def make_sentiment_agent(
    model: SentimentClassifier,
    rule_based: Callable[[str], tuple[str, str]],
    min_confidence: float = 0.6,
) -> Callable[[str], tuple[str, str]]:
    """Sentiment Agent: message -> (sentiment, urgency).

    - Urgency has no labelled data, so it always comes from the keyword rules.
    - Low model confidence -> the rule-based sentiment is used (never a guess).
    - A message is negative when EITHER the model (confident) OR the rules say so: a missed angry
      customer costs more than an extra human look, and "negative" only escalates together with
      high urgency. This also covers mixed messages ("great product, terrible delivery").
    """

    def analyze(message: str) -> tuple[str, str]:
        rule_sentiment, urgency = rule_based(message)
        label, confidence = model.predict_with_confidence(message)
        sentiment = label if confidence >= min_confidence else rule_sentiment
        if rule_sentiment == "negative":
            sentiment = "negative"
        return sentiment, urgency

    return analyze


def prepare_splits(
    data: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean, deduplicate on cleaned text, then stratified 70/15/15 train/validation/test.
    Deduplicating BEFORE splitting keeps the same text out of both train and test."""
    if not {"text", "label"} <= set(data.columns):
        raise ModelError("The CSV needs 'text' and 'label' columns.")
    frame = data[["text", "label"]].dropna().copy()
    frame["text"] = frame["text"].astype(str).map(clean_text)
    frame = frame[frame["text"].str.len() > 0].drop_duplicates(subset="text")
    if set(frame["label"]) - set(LABELS):
        raise ModelError(f"Labels must be in {list(LABELS)}.")
    if frame["label"].value_counts().min() < 10:
        raise ModelError("Each class needs at least 10 distinct messages.")
    train, rest = train_test_split(frame, test_size=0.3, stratify=frame["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=seed)
    return train, val, test


def train_and_evaluate(
    data: pd.DataFrame, seed: int = 42, score_test: bool = False
) -> tuple[SentimentClassifier, dict[str, object]]:
    train, val, test = prepare_splits(data, seed)
    model = SentimentClassifier(seed=seed).fit(train["text"].tolist(), train["label"].tolist())
    predicted = model.predict(val["text"].tolist())
    report: dict[str, object] = {
        "n_train": len(train),
        "n_val": len(val),
        "validation": evaluate_classification(val["label"].tolist(), predicted),
        "validation_per_class": per_class_report(val["label"].tolist(), predicted),
    }
    if score_test:  # spend the test split once, at the very end
        test_pred = model.predict(test["text"].tolist())
        report["test"] = evaluate_classification(test["label"].tolist(), test_pred)
        report["n_test"] = len(test)
    return model, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment from a CSV (text,label).")
    parser.add_argument("csv", type=Path)
    default_out = Path("artifacts/models/sentiment_tfidf_lr.joblib")
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--score-test", action="store_true", help="score the test split (once!)")
    args = parser.parse_args()
    trained, result = train_and_evaluate(
        pd.read_csv(args.csv, encoding="utf-8"), score_test=args.score_test
    )
    trained.save(args.out)
    print(json.dumps(result, indent=2))
