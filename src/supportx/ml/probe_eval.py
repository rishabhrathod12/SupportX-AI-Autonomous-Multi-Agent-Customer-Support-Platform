"""Score an intent model on the hand-written probe set (`data/sample/intent_probe.csv`).

Accuracy alone hides what matters for routing, so three outcomes are reported per message:
  useful          right intent AND confidence >= threshold (the ticket is handled automatically)
  escalated       confidence below the threshold (a person takes over: safe, but costs staff time)
  confidently_wrong  wrong intent (or any answer to an out-of-scope message, label "OOS") at
                  or above the threshold: the dangerous case
The probe set has a `dev` part (used for decisions) and a `holdout` part (score once, at the end).

    python -m supportx.ml.probe_eval artifacts/models/intent_tfidf_lr_bitext.joblib --split dev
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from supportx.ml.intent import IntentClassifier

OOS = "OOS"
PROBE_PATH = Path("data/sample/intent_probe.csv")


def score_probes(model: Any, probes: pd.DataFrame, threshold: float = 0.6) -> dict[str, Any]:
    useful = escalated = wrong = 0
    in_scope = oos_total = oos_escalated = 0
    by_intent: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [useful, n]
    failures: list[tuple[str, str, str, float]] = []
    for text, truth in zip(probes["text"], probes["intent"], strict=True):
        label, confidence = model.predict_with_confidence(text)
        confident = confidence >= threshold
        if truth == OOS:
            oos_total += 1
            if confident:
                wrong += 1
                failures.append((truth, label, text, confidence))
            else:
                oos_escalated += 1
            continue
        in_scope += 1
        by_intent[truth][1] += 1
        if not confident:
            escalated += 1
            failures.append((truth, f"(escalated, said {label})", text, confidence))
        elif label == truth:
            useful += 1
            by_intent[truth][0] += 1
        else:
            wrong += 1
            failures.append((truth, label, text, confidence))
    return {
        "threshold": threshold,
        "in_scope": in_scope,
        "useful": useful,
        "escalated": escalated,
        "confidently_wrong": wrong,
        "oos_total": oos_total,
        "oos_escalated": oos_escalated,
        "useful_rate": useful / in_scope if in_scope else 0.0,
        "per_intent_useful": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_intent.items())},
        "failures": failures,
    }


def load_probes(split: str, path: Path = PROBE_PATH) -> pd.DataFrame:
    probes = pd.read_csv(path, encoding="utf-8")
    return probes[probes["split"] == split].reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a model on the probe set.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()
    result = score_probes(
        IntentClassifier.load(args.model), load_probes(args.split), args.threshold
    )
    failures = result.pop("failures")
    for key, value in result.items():
        print(f"{key}: {value}")
    if args.show_failures:
        for truth, got, text, conf in failures:
            print(f"  [{truth}] -> {got} ({conf:.2f}): {text}")
