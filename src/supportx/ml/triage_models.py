"""Trainable models behind `supportx.services.triage`: priority, team and resolution time.

NO MODEL IS TRAINED OR SHIPPED. There is no real labelled ticket data yet, and synthetic data
would only measure the generator. This module is the pipeline for when such data exists:

  CSV columns: message, intent, sentiment, urgency, approval_kind (may be empty)   <- features
               priority, team, resolution_hours                                    <- targets
               created_at (optional)  -> if present, split chronologically (past -> future)

LEAKAGE RULES: features are only facts known when the ticket arrives (the FEATURES list). Closing
time, final status, customer rating etc. must not be added. Messages are deduplicated before the
split. The test split is only scored on request.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from supportx.core import ModelError, get_logger
from supportx.ml.evaluate import evaluate_classification
from supportx.services.triage import PRIORITIES, TEAMS

logger = get_logger(__name__)

CATEGORICAL = ["intent", "sentiment", "urgency", "approval_kind"]
FEATURES = ["message", *CATEGORICAL]
MIN_PER_CLASS = 10
MIN_TICKETS = 100


def _frame(facts: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = [{c: ("" if f.get(c) is None else str(f.get(c))) for c in FEATURES} for f in facts]
    return pd.DataFrame(rows, columns=FEATURES)


def _features() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "message"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


class TriageClassifier:
    """Predicts one label (priority or team) from ticket facts, with a confidence."""

    def __init__(self, allowed: Sequence[str], seed: int = 42) -> None:
        self.allowed = tuple(allowed)
        self._pipeline = Pipeline(
            [
                ("features", _features()),
                (
                    "clf",
                    LogisticRegression(
                        C=3.0, max_iter=1000, class_weight="balanced", random_state=seed
                    ),
                ),
            ]
        )
        self._fitted = False

    def fit(self, facts: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> Self:
        unknown = sorted(set(labels) - set(self.allowed))
        if unknown:
            raise ModelError(f"Unknown labels {unknown}; allowed: {list(self.allowed)}.")
        if len(facts) != len(labels) or not labels:
            raise ModelError("facts and labels must be non-empty and the same length.")
        self._pipeline.fit(_frame(facts), list(labels))
        self._fitted = True
        return self

    def __call__(self, facts: Mapping[str, Any]) -> tuple[str, float]:
        if not self._fitted:
            raise ModelError("TriageClassifier is not trained yet.")
        probs = self._pipeline.predict_proba(_frame([facts]))[0]
        best = int(np.argmax(probs))
        return str(self._pipeline.classes_[best]), float(probs[best])

    def predict(self, facts: Sequence[Mapping[str, Any]]) -> list[str]:
        return [str(p) for p in self._pipeline.predict(_frame(facts))]


class ResolutionTimeModel:
    """Expected hours to resolve (Ridge on log1p(hours)); point estimate only, no interval."""

    def __init__(self) -> None:
        self._pipeline = Pipeline([("features", _features()), ("reg", Ridge(alpha=1.0))])
        self._fitted = False

    def fit(self, facts: Sequence[Mapping[str, Any]], hours: Sequence[float]) -> Self:
        values = np.asarray(hours, dtype=float)
        if len(facts) != len(values) or not len(values) or (values <= 0).any():
            raise ModelError("hours must be positive and match the facts.")
        self._pipeline.fit(_frame(facts), np.log1p(values))
        self._fitted = True
        return self

    def __call__(self, facts: Mapping[str, Any]) -> float:
        if not self._fitted:
            raise ModelError("ResolutionTimeModel is not trained yet.")
        return float(np.expm1(self._pipeline.predict(_frame([facts]))[0]))

    def predict(self, facts: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not self._fitted:
            raise ModelError("ResolutionTimeModel is not trained yet.")
        return np.asarray(np.expm1(self._pipeline.predict(_frame(facts))))


def split_tickets(
    data: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15. Chronological when `created_at` exists (train on the past, test on the future),
    otherwise a seeded shuffle. Duplicate messages are removed first."""
    needed = {"message", "intent", "sentiment", "urgency", "priority", "team", "resolution_hours"}
    missing = needed - set(data.columns)
    if missing:
        raise ModelError(f"Missing columns: {sorted(missing)}.")
    frame = data.copy()
    if "approval_kind" not in frame:
        frame["approval_kind"] = ""
    frame = frame.drop_duplicates(subset="message")
    if "created_at" in frame:
        frame = frame.sort_values("created_at")
    else:
        frame = frame.sample(frac=1.0, random_state=seed)
    n = len(frame)
    if n < MIN_TICKETS:
        raise ModelError(f"Need at least {MIN_TICKETS} distinct tickets to train and evaluate.")
    a, b = int(n * 0.7), int(n * 0.85)
    return frame.iloc[:a], frame.iloc[a:b], frame.iloc[b:]


def _facts(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {c: (None if pd.isna(row[c]) or row[c] == "" else row[c]) for c in FEATURES}
        for _, row in frame.iterrows()
    ]


def train_triage(
    data: pd.DataFrame, seed: int = 42, score_test: bool = False
) -> tuple[TriageClassifier, TriageClassifier, ResolutionTimeModel, dict[str, Any]]:
    train, val, test = split_tickets(data, seed)
    for column, allowed in (("priority", PRIORITIES), ("team", TEAMS)):
        counts = train[column].value_counts()
        if set(counts.index) - set(allowed) or counts.min() < MIN_PER_CLASS:
            raise ModelError(f"'{column}' needs labels {list(allowed)}, {MIN_PER_CLASS}+ each.")
    priority = TriageClassifier(PRIORITIES, seed).fit(_facts(train), train["priority"].tolist())
    team = TriageClassifier(TEAMS, seed).fit(_facts(train), train["team"].tolist())
    time_model = ResolutionTimeModel().fit(_facts(train), train["resolution_hours"].tolist())
    baseline = float(np.median(train["resolution_hours"]))  # "always guess the median"

    def score(part: pd.DataFrame) -> dict[str, Any]:
        facts = _facts(part)
        hours = part["resolution_hours"].to_numpy(dtype=float)
        return {
            "n": len(part),
            "priority": evaluate_classification(part["priority"].tolist(), priority.predict(facts)),
            "team": evaluate_classification(part["team"].tolist(), team.predict(facts)),
            "resolution_mae_hours": float(mean_absolute_error(hours, time_model.predict(facts))),
            "resolution_mae_median_baseline": float(
                mean_absolute_error(hours, np.full_like(hours, baseline))
            ),
        }

    report: dict[str, Any] = {"split": "chronological" if "created_at" in data else "random"}
    report["validation"] = score(val)
    if score_test:  # spend the test split once, at the very end
        report["test"] = score(test)
    return priority, team, time_model, report


MANIFEST = "manifest.json"
MODEL_NAMES = ("priority", "team", "resolution")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_models(
    directory: Path,
    *,
    priority: TriageClassifier,
    team: TriageClassifier,
    resolution: ResolutionTimeModel,
    data_origin: str,
    report: Mapping[str, Any] | None = None,
) -> Path:
    """Save the three models plus a manifest (version, time, data origin, file hashes, report).
    `data_origin` must be stated by the caller ("real" or "synthetic"): a model is never
    presented as real-customer performance unless it was trained and scored on real data."""
    if data_origin not in ("real", "synthetic"):
        raise ModelError("data_origin must be 'real' or 'synthetic'.")
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for name, model in zip(MODEL_NAMES, (priority, team, resolution), strict=True):
        path = directory / f"{name}.joblib"
        joblib.dump(model, path)
        files[path.name] = _sha256(path)
    manifest = {
        "version": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "trained_at": datetime.now(UTC).isoformat(),
        "data_origin": data_origin,
        "files": files,
        "report": dict(report or {}),
    }
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return directory / MANIFEST


def load_models(
    directory: Path,
) -> tuple[TriageClassifier, TriageClassifier, ResolutionTimeModel, dict[str, Any]]:
    """Load models saved by `save_models`, refusing files whose hash differs from the manifest
    (joblib is pickle: only load directories you created yourself)."""
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        raise ModelError(f"No {MANIFEST} in {directory}; train with supportx.ml.triage_models.")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded: list[Any] = []
    for name in MODEL_NAMES:
        path = directory / f"{name}.joblib"
        if not path.exists() or manifest["files"].get(path.name) != _sha256(path):
            raise ModelError(f"{path.name} is missing or does not match the manifest hash.")
        loaded.append(joblib.load(path))
    priority, team, resolution = loaded
    if not (
        isinstance(priority, TriageClassifier)
        and isinstance(team, TriageClassifier)
        and isinstance(resolution, ResolutionTimeModel)
    ):
        raise ModelError("The model files do not contain the expected model types.")
    return priority, team, resolution, manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train triage models from a ticket CSV.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/models/triage"))
    parser.add_argument("--score-test", action="store_true", help="score the test split (once!)")
    parser.add_argument(
        "--data-origin", choices=["real", "synthetic"], required=True, help="stored in manifest"
    )
    args = parser.parse_args()
    p, t, r, result = train_triage(
        pd.read_csv(args.csv, encoding="utf-8"), score_test=args.score_test
    )
    save_models(
        args.out, priority=p, team=t, resolution=r, data_origin=args.data_origin, report=result
    )
    print(json.dumps(result, indent=2))
