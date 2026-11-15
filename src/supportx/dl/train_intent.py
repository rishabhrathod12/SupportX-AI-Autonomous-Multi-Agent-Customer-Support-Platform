"""Train and evaluate the BiLSTM intent model, tracked in MLflow.

Run:  python -m supportx.dl.train_intent bitext
      python -m supportx.dl.train_intent banking77

TRAIN data fits the network; the VALIDATION split decides when to stop (early stopping) and is
what we report. The test split is still untouched, reserved for the final model comparison.
Note: because validation picks the best epoch, its score is slightly optimistic; the untouched
test split gives the unbiased number.
"""

import argparse
import json
import time
from pathlib import Path

from supportx.core import DataError, get_logger, setup_logging
from supportx.data.splits import load_split, load_split_info
from supportx.dl.intent import KerasIntentClassifier
from supportx.ml.evaluate import per_class_report
from supportx.ml.experiment import evaluate_with_baseline, log_experiment_run

logger = get_logger(__name__)
LABEL_COLUMN = "intent"
EXPERIMENT = "supportx-intent"  # same experiment as the classical models, for side-by-side runs


def run_training(
    dataset: str,
    processed_root: Path = Path("data/processed"),
    models_root: Path = Path("artifacts/models"),
    tracking_uri: str | None = None,
    artifact_root: Path | None = None,
    epochs: int = 30,
    embedding_dim: int = 64,
    lstm_units: int = 64,
    seed: int = 42,
) -> dict[str, float]:
    """Train on the train split, early-stop and score on validation, log everything to MLflow."""
    train = load_split(processed_root, dataset, "train", LABEL_COLUMN)
    val = load_split(processed_root, dataset, "val", LABEL_COLUMN)
    split_info = load_split_info(processed_root, dataset)

    started = time.perf_counter()
    model = KerasIntentClassifier(
        embedding_dim=embedding_dim, lstm_units=lstm_units, epochs=epochs, seed=seed
    ).fit(train["text"], train[LABEL_COLUMN], val["text"], val[LABEL_COLUMN])
    train_seconds = time.perf_counter() - started

    predictions = model.predict(val["text"])
    metrics = evaluate_with_baseline(train[LABEL_COLUMN], val[LABEL_COLUMN], predictions)
    metrics["train_seconds"] = train_seconds
    metrics["epochs_run"] = float(model.epochs_run)

    report = per_class_report(val[LABEL_COLUMN], predictions)
    worst = sorted(
        ((name, stats["f1-score"]) for name, stats in report.items() if name in model.classes_),
        key=lambda item: item[1],
    )[:5]

    model_dir = model.save(models_root / f"intent_bilstm_{dataset}")
    report_path = models_root / f"intent_bilstm_{dataset}.val_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log_experiment_run(
        experiment=EXPERIMENT,
        run_name=f"bilstm_{dataset}",
        params={
            **model.params,
            "model": "bilstm",
            "dataset": dataset,
            "dataset_synthetic": split_info.get("synthetic"),
            "n_train": len(train),
            "n_val": len(val),
            "n_classes": len(model.classes_),
        },
        metrics=metrics,
        artifacts=[report_path, model_dir / "config.json"],
        tracking_uri=tracking_uri,
        artifact_root=artifact_root,
    )
    logger.info(
        "Validation metrics for %s: %s", dataset, {k: round(v, 4) for k, v in metrics.items()}
    )
    logger.info("Weakest validation intents (F1): %s", [(n, round(f, 3)) for n, f in worst])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BiLSTM intent classifier.")
    parser.add_argument("dataset", choices=["bitext", "banking77"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-root", type=Path, default=Path("artifacts/models"))
    args = parser.parse_args()
    setup_logging("INFO")
    try:
        run_training(
            args.dataset,
            models_root=args.models_root,
            epochs=args.epochs,
            embedding_dim=args.embedding_dim,
            lstm_units=args.lstm_units,
            seed=args.seed,
        )
    except DataError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
