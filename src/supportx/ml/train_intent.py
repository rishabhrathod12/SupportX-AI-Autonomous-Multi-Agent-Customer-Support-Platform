"""Train and evaluate the intent baseline, tracked in MLflow.

Run:  python -m supportx.ml.train_intent bitext
      python -m supportx.ml.train_intent banking77

Only the TRAIN and VALIDATION splits are used here. The test split is reserved for the
final model comparison, so choosing settings never "sees" it.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import mlflow
import pandas as pd

from supportx.core import DataError, get_logger, get_settings, setup_logging
from supportx.ml.evaluate import evaluate_classification, per_class_report
from supportx.ml.intent import IntentClassifier

logger = get_logger(__name__)
LABEL_COLUMN = "intent"
EXPERIMENT = "supportx-intent"


def _load_split(root: Path, dataset: str, split: str) -> pd.DataFrame:
    path = root / dataset / f"{split}.csv"
    if not path.exists():
        raise DataError(f"{path} not found. Run: python -m supportx.data.prepare {dataset}")
    return pd.read_csv(path, encoding="utf-8").dropna(subset=["text", LABEL_COLUMN])


def run_training(
    dataset: str,
    processed_root: Path = Path("data/processed"),
    models_root: Path = Path("artifacts/models"),
    tracking_uri: str | None = None,
    artifact_root: Path | None = None,
    c: float = 10.0,
    seed: int = 42,
) -> dict[str, float]:
    """Train on the train split, score on the validation split, log everything to MLflow."""
    train = _load_split(processed_root, dataset, "train")
    val = _load_split(processed_root, dataset, "val")
    split_info = json.loads((processed_root / dataset / "SPLIT_INFO.json").read_text("utf-8"))

    model = IntentClassifier(c=c, seed=seed).fit(train["text"], train[LABEL_COLUMN])
    predictions = model.predict(val["text"])
    metrics = evaluate_classification(val[LABEL_COLUMN], predictions)

    # Sanity baseline: always predict the most common training intent.
    majority = Counter(train[LABEL_COLUMN]).most_common(1)[0][0]
    baseline = evaluate_classification(val[LABEL_COLUMN], [majority] * len(val))
    metrics = {f"val_{k}": v for k, v in metrics.items()}
    metrics.update({f"majority_baseline_val_{k}": v for k, v in baseline.items()})

    report = per_class_report(val[LABEL_COLUMN], predictions)
    worst = sorted(
        ((name, stats["f1-score"]) for name, stats in report.items() if name in model.classes_),
        key=lambda item: item[1],
    )[:5]

    models_root.mkdir(parents=True, exist_ok=True)
    model_path = model.save(models_root / f"intent_tfidf_lr_{dataset}.joblib")
    report_path = models_root / f"intent_tfidf_lr_{dataset}.val_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)
    if artifact_root is not None and mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=artifact_root.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"tfidf_lr_{dataset}"):
        mlflow.log_params(
            {
                **model.params,
                "model": "tfidf+logreg",
                "dataset": dataset,
                "dataset_synthetic": split_info.get("synthetic"),
                "n_train": len(train),
                "n_val": len(val),
                "n_classes": len(model.classes_),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(model_path))

    logger.info(
        "Validation metrics for %s: %s", dataset, {k: round(v, 4) for k, v in metrics.items()}
    )
    logger.info("Weakest validation intents (F1): %s", [(n, round(f, 3)) for n, f in worst])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TF-IDF + LogReg intent baseline.")
    parser.add_argument("dataset", choices=["bitext", "banking77"])
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    setup_logging("INFO")
    try:
        run_training(args.dataset, c=args.c, seed=args.seed)
    except DataError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
