"""Train and evaluate the intent baseline, tracked in MLflow.

Run:  python -m supportx.ml.train_intent bitext
      python -m supportx.ml.train_intent banking77

Only the TRAIN and VALIDATION splits are used here. The test split is reserved for the
final model comparison, so choosing settings never "sees" it.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import mlflow
import pandas as pd

from supportx.core import DataError, get_logger, get_settings, setup_logging
from supportx.ml.evaluate import evaluate_classification, per_class_report
from supportx.ml.intent import IntentClassifier, ModelKind

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
    model_kind: ModelKind = "logreg",
    n_estimators: int = 200,
) -> dict[str, float]:
    """Train on the train split, score on the validation split, log everything to MLflow."""
    train = _load_split(processed_root, dataset, "train")
    val = _load_split(processed_root, dataset, "val")
    split_info = json.loads((processed_root / dataset / "SPLIT_INFO.json").read_text("utf-8"))

    started = time.perf_counter()
    model = IntentClassifier(model=model_kind, c=c, seed=seed, n_estimators=n_estimators).fit(
        train["text"], train[LABEL_COLUMN]
    )
    train_seconds = time.perf_counter() - started
    predictions = model.predict(val["text"])
    metrics = evaluate_classification(val[LABEL_COLUMN], predictions)

    # Sanity baseline: always predict the most common training intent.
    majority = Counter(train[LABEL_COLUMN]).most_common(1)[0][0]
    baseline = evaluate_classification(val[LABEL_COLUMN], [majority] * len(val))
    metrics = {f"val_{k}": v for k, v in metrics.items()}
    metrics.update({f"majority_baseline_val_{k}": v for k, v in baseline.items()})
    metrics["train_seconds"] = train_seconds

    report = per_class_report(val[LABEL_COLUMN], predictions)
    worst = sorted(
        ((name, stats["f1-score"]) for name, stats in report.items() if name in model.classes_),
        key=lambda item: item[1],
    )[:5]

    models_root.mkdir(parents=True, exist_ok=True)
    tag = {"logreg": "tfidf_lr", "xgboost": "tfidf_xgb"}[model_kind]
    model_path = model.save(models_root / f"intent_{tag}_{dataset}.joblib")
    report_path = models_root / f"intent_{tag}_{dataset}.val_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)
    if artifact_root is not None and mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=artifact_root.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"{tag}_{dataset}"):
        mlflow.log_params(
            {
                **model.params,
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
    parser = argparse.ArgumentParser(description="Train a TF-IDF intent classifier.")
    parser.add_argument("dataset", choices=["bitext", "banking77"])
    parser.add_argument("--model", choices=["logreg", "xgboost"], default="logreg")
    parser.add_argument("--c", type=float, default=10.0, help="logreg regularization strength")
    parser.add_argument("--n-estimators", type=int, default=200, help="xgboost trees")
    parser.add_argument("--seed", type=int, default=42)
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
        )
    except DataError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
