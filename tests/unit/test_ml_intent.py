import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from supportx.core import DataError, ModelError
from supportx.data.sample import generate_sample_tickets
from supportx.ml import IntentClassifier, evaluate_classification, per_class_report
from supportx.ml.train_intent import run_training


@pytest.fixture(scope="module")
def tickets() -> pd.DataFrame:
    """SYNTHETIC demo data: only proves the code works, says nothing about real accuracy."""
    return generate_sample_tickets(400, seed=7)


@pytest.fixture(scope="module")
def trained(tickets: pd.DataFrame) -> IntentClassifier:
    return IntentClassifier().fit(tickets["text"], tickets["intent"])


def test_predicts_the_right_intent_on_clear_examples(trained: IntentClassifier) -> None:
    assert trained.predict(["Where is my order ORD-12345?"]) == ["track_order"]
    assert trained.predict(["What is your refund policy?"]) == ["check_refund_policy"]


def test_probabilities_are_valid(trained: IntentClassifier) -> None:
    probs = trained.predict_proba(["I want to cancel order ORD-55555.", "hello"])
    assert probs.shape == (2, len(trained.classes_))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_predict_with_confidence(trained: IntentClassifier) -> None:
    intent, confidence = trained.predict_with_confidence("Please cancel my order ORD-99999.")
    assert intent == "cancel_order"
    assert 0.0 < confidence <= 1.0


def test_xgboost_variant_trains_predicts_and_round_trips(
    tickets: pd.DataFrame, tmp_path: Path
) -> None:
    model = IntentClassifier(model="xgboost", n_estimators=20).fit(
        tickets["text"], tickets["intent"]
    )
    assert model.predict(["Where is my order ORD-12345?"]) == ["track_order"]
    probs = model.predict_proba(["I want a refund for order ORD-5."])
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert set(model.classes_) == set(tickets["intent"])  # string labels survive encoding
    loaded = IntentClassifier.load(model.save(tmp_path / "xgb.joblib"))
    assert loaded.predict(["cancel order ORD-9"]) == model.predict(["cancel order ORD-9"])
    assert loaded.params["model"] == "xgboost"


def test_unknown_model_kind_is_rejected() -> None:
    with pytest.raises(ModelError, match="Unknown model"):
        IntentClassifier(model="svm")  # type: ignore[arg-type]


def test_unfitted_model_raises() -> None:
    with pytest.raises(ModelError, match="not trained"):
        IntentClassifier().predict(["hi"])


def test_fit_rejects_bad_input() -> None:
    with pytest.raises(ModelError):
        IntentClassifier().fit(["a", "b"], ["x"])
    with pytest.raises(ModelError):
        IntentClassifier().fit([], [])


def test_save_and_load_round_trip(trained: IntentClassifier, tmp_path: Path) -> None:
    path = trained.save(tmp_path / "models" / "m.joblib")
    loaded = IntentClassifier.load(path)
    sample = ["Where is my order ORD-1?", "refund order ORD-2 please", "unhappy with order ORD-3"]
    assert loaded.predict(sample) == trained.predict(sample)
    assert np.allclose(loaded.predict_proba(sample), trained.predict_proba(sample))


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="not found"):
        IntentClassifier.load(tmp_path / "missing.joblib")
    import joblib

    joblib.dump({"not": "a model"}, tmp_path / "wrong.joblib")
    with pytest.raises(ModelError, match="does not contain"):
        IntentClassifier.load(tmp_path / "wrong.joblib")


def test_evaluate_classification_known_values() -> None:
    metrics = evaluate_classification(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert metrics["accuracy"] == pytest.approx(0.75)
    # class a: P=1, R=.5, F1=.667 ; class b: P=.667, R=1, F1=.8  -> macro F1 = .7333
    assert metrics["macro_f1"] == pytest.approx(0.7333, abs=1e-3)
    assert metrics["macro_precision"] == pytest.approx((1 + 2 / 3) / 2)


def test_per_class_report_has_each_class() -> None:
    report = per_class_report(["a", "a", "b"], ["a", "b", "b"])
    assert {"a", "b"} <= set(report)
    assert report["a"]["support"] == 2


def _write_processed(root: Path, tickets: pd.DataFrame) -> None:
    folder = root / "toy"
    folder.mkdir(parents=True)
    data = tickets[["text", "intent"]]
    data.iloc[:300].to_csv(folder / "train.csv", index=False)
    data.iloc[300:].to_csv(folder / "val.csv", index=False)
    (folder / "SPLIT_INFO.json").write_text(json.dumps({"synthetic": True}), encoding="utf-8")


def test_run_training_logs_to_mlflow_and_saves_model(tickets: pd.DataFrame, tmp_path: Path) -> None:
    _write_processed(tmp_path / "processed", tickets)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    metrics = run_training(
        "toy",
        processed_root=tmp_path / "processed",
        models_root=tmp_path / "models",
        tracking_uri=uri,
        artifact_root=tmp_path / "mlartifacts",
    )
    assert metrics["val_accuracy"] > metrics["majority_baseline_val_accuracy"]
    assert (tmp_path / "models" / "intent_tfidf_lr_toy.joblib").exists()
    assert (tmp_path / "models" / "intent_tfidf_lr_toy.val_report.json").exists()

    client = mlflow.MlflowClient(tracking_uri=uri)  # read back what MLflow stored
    experiment = client.get_experiment_by_name("supportx-intent")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.params["dataset"] == "toy"
    assert runs[0].data.metrics["val_accuracy"] == pytest.approx(metrics["val_accuracy"])


def test_run_training_reports_missing_data(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="prepare"):
        run_training("nope", processed_root=tmp_path, models_root=tmp_path)
