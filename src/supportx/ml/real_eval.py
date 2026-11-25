"""Score an intent model on REAL, human-labeled customer messages (see `supportx.data.real`).

Kept apart from every synthetic evaluation on purpose: the report always says `data_origin: real`,
the number of messages and labelers, and refuses to run on too little data (a score from 20
messages is noise). The TEST split is scored once per model file (the hash is recorded and a second
run is refused), exactly like `supportx.ml.final_test`.

    python -m supportx.ml.real_eval artifacts/models/intent_tfidf_lr_bitext_aug.joblib \
        data/real/processed/all.csv --split val
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from supportx.core import DataError
from supportx.data.real import usable_labels
from supportx.ml.evaluate import evaluate_classification, per_class_report
from supportx.ml.intent import IntentClassifier

MIN_MESSAGES = {"val": 100, "test": 100}
REPORT_DIR = Path("artifacts/evaluation")


def evaluate_real(
    model: Any,
    frame: pd.DataFrame,
    split: str = "val",
    threshold: float = 0.6,
    min_messages: int | None = None,
) -> dict[str, Any]:
    if split not in MIN_MESSAGES:
        raise DataError("Real data is evaluated on 'val' or 'test' (never on train).")
    rows = usable_labels(frame, split)
    needed = MIN_MESSAGES[split] if min_messages is None else min_messages
    if len(rows) < needed:
        raise DataError(
            f"Only {len(rows)} usable labeled {split} messages; at least {needed} are needed for a "
            "meaningful score. Collect and label more before evaluating."
        )
    known = set(model.classes_)
    unknown = sorted(set(rows["label"]) - known)
    texts = rows["text"].tolist()
    truth = rows["label"].tolist()
    predictions = [model.predict_with_confidence(t) for t in texts]
    predicted = [label for label, _ in predictions]
    confident = [conf >= threshold for _, conf in predictions]
    useful = sum(c and p == t for c, p, t in zip(confident, predicted, truth, strict=True))
    wrong = sum(c and p != t for c, p, t in zip(confident, predicted, truth, strict=True))
    return {
        "data_origin": "real",
        "split": split,
        "n": len(rows),
        "labelers": sorted(set(rows["labeler"])),
        "labels_unknown_to_model": unknown,
        "threshold": threshold,
        "handled_automatically_right": useful / len(rows),
        "escalated": 1 - sum(confident) / len(rows),
        "confidently_wrong": wrong / len(rows),
        "metrics": evaluate_classification(truth, predicted),
        "per_class": per_class_report(truth, predicted),
    }


def run(model_path: Path, data_path: Path, split: str, threshold: float) -> dict[str, Any]:
    model = IntentClassifier.load(model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    marker = REPORT_DIR / f"real_{split}_{digest[:12]}.json"
    if split == "test" and marker.exists():
        raise DataError(f"The real test split was already scored for this model ({marker}).")
    frame = pd.read_csv(data_path, encoding="utf-8", dtype=str, keep_default_na=False)
    frame["needs_adjudication"] = frame["needs_adjudication"].astype(str).str.lower() == "true"
    report = evaluate_real(model, frame, split, threshold)
    report["model_sha256"] = digest
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a model on real labeled messages.")
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    print(json.dumps(run(args.model, args.data, args.split, args.threshold), indent=2))
