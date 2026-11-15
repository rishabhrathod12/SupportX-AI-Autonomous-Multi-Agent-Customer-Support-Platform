"""Helpers shared by every training script (ML and DL): sanity baseline and MLflow logging."""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow

from supportx.core import get_settings
from supportx.ml.evaluate import evaluate_classification


def evaluate_with_baseline(
    train_labels: Sequence[str], val_labels: Sequence[str], val_predictions: Sequence[str]
) -> dict[str, float]:
    """Validation metrics plus a 'always predict the most common class' sanity baseline."""
    majority = Counter(train_labels).most_common(1)[0][0]
    model_scores = evaluate_classification(val_labels, val_predictions)
    baseline_scores = evaluate_classification(val_labels, [majority] * len(val_labels))
    metrics = {f"val_{k}": v for k, v in model_scores.items()}
    metrics.update({f"majority_baseline_val_{k}": v for k, v in baseline_scores.items()})
    return metrics


def log_experiment_run(
    *,
    experiment: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: Sequence[Path],
    tracking_uri: str | None = None,
    artifact_root: Path | None = None,
) -> None:
    """Record one training run (parameters, metrics, files) in MLflow."""
    mlflow.set_tracking_uri(tracking_uri or get_settings().mlflow_tracking_uri)
    if artifact_root is not None and mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=artifact_root.resolve().as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
