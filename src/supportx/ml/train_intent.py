"""Train and evaluate a TF-IDF intent classifier (LogReg or XGBoost), tracked in MLflow.

Run:  python -m supportx.ml.train_intent bitext
      python -m supportx.ml.train_intent banking77 --model xgboost

Only the TRAIN and VALIDATION splits are used here. The test split is reserved for the
final model comparison, so choosing settings never "sees" it.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from supportx.core import DataError, get_logger, setup_logging
from supportx.data.augment import build_augmentation
from supportx.data.splits import load_split, load_split_info
from supportx.ml.evaluate import per_class_report
from supportx.ml.experiment import evaluate_with_baseline, log_experiment_run
from supportx.ml.intent import IntentClassifier, ModelKind
from supportx.ml.probe_eval import PROBE_PATH

logger = get_logger(__name__)
LABEL_COLUMN = "intent"
EXPERIMENT = "supportx-intent"


def run_training(
    dataset: str,
    processed_root: Path = Path("data/processed"),
    models_root: Path = Path("artifacts/models"),
    tracking_uri: str | None = None,
    artifact_root: Path | None = None,
    c: float = 10.0,
    seed: int = 42,
    model_kind: ModelKind = "logreg",
    n_estimators: int = 200,
    augment_per_intent: int = 0,
    probe_path: Path = PROBE_PATH,
) -> dict[str, float]:
    """Train on the train split, score on the validation split, log everything to MLflow."""
    train = load_split(processed_root, dataset, "train", LABEL_COLUMN)
    val = load_split(processed_root, dataset, "val", LABEL_COLUMN)
    split_info = load_split_info(processed_root, dataset)
    variant = ""
    if augment_per_intent > 0:
        # hand-written realistic phrasings, TRAIN split only; probe texts are never included
        probes = pd.read_csv(probe_path, encoding="utf-8")
        extra = build_augmentation(augment_per_intent, seed, drop_texts=probes["text"])
        train = pd.concat([train[["text", LABEL_COLUMN]], extra], ignore_index=True)
        variant = "_aug"

    started = time.perf_counter()
    model = IntentClassifier(model=model_kind, c=c, seed=seed, n_estimators=n_estimators).fit(
        train["text"], train[LABEL_COLUMN]
    )
    train_seconds = time.perf_counter() - started

    predictions = model.predict(val["text"])
    metrics = evaluate_with_baseline(train[LABEL_COLUMN], val[LABEL_COLUMN], predictions)
    metrics["train_seconds"] = train_seconds

    report = per_class_report(val[LABEL_COLUMN], predictions)
    worst = sorted(
        ((name, stats["f1-score"]) for name, stats in report.items() if name in model.classes_),
        key=lambda item: item[1],
    )[:5]

    models_root.mkdir(parents=True, exist_ok=True)
    tag = {"logreg": "tfidf_lr", "xgboost": "tfidf_xgb"}[model_kind]
    model_path = model.save(models_root / f"intent_{tag}_{dataset}{variant}.joblib")
    report_path = models_root / f"intent_{tag}_{dataset}{variant}.val_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log_experiment_run(
        experiment=EXPERIMENT,
        run_name=f"{tag}_{dataset}{variant}",
        params={
            **model.params,
            "dataset": dataset,
            "dataset_synthetic": split_info.get("synthetic"),
            "n_train": len(train),
            "n_val": len(val),
            "augment_per_intent": augment_per_intent,
            "n_classes": len(model.classes_),
        },
        metrics=metrics,
        artifacts=[report_path, model_path],
        tracking_uri=tracking_uri,
        artifact_root=artifact_root,
    )
    logger.info(
        "Validation metrics for %s: %s", dataset, {k: round(v, 4) for k, v in metrics.items()}
    )
    logger.info("Weakest validation intents (F1): %s", [(n, round(f, 3)) for n, f in worst])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a TF-IDF intent classifier.")
    parser.add_argument("dataset", choices=["bitext", "banking77"])
    parser.add_argument("--model", choices=["logreg", "xgboost"], default="logreg")
    parser.add_argument("--c", type=float, default=10.0, help="logreg regularization strength")
    parser.add_argument("--n-estimators", type=int, default=200, help="xgboost trees")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augment", type=int, default=0, help="hand-written rows per routed intent (0 = none)"
    )
    parser.add_argument("--models-root", type=Path, default=Path("artifacts/models"))
    args = parser.parse_args()
    setup_logging("INFO")
    try:
        run_training(
            args.dataset,
            models_root=args.models_root,
            c=args.c,
            seed=args.seed,
            model_kind=args.model,
            n_estimators=args.n_estimators,
            augment_per_intent=args.augment,
        )
    except DataError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
