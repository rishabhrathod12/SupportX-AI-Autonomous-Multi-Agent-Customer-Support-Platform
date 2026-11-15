from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from supportx.core import DataError, ModelError
from supportx.data.sample import generate_sample_tickets
from supportx.dl.intent import OOV_ID, PAD_ID, KerasIntentClassifier
from supportx.dl.train_intent import run_training


@pytest.fixture(scope="module")
def tickets() -> pd.DataFrame:
    """SYNTHETIC demo data: proves the code runs, says nothing about real accuracy."""
    return generate_sample_tickets(300, seed=11)


@pytest.fixture(scope="module")
def trained(tickets: pd.DataFrame) -> KerasIntentClassifier:
    train, val = tickets.iloc[:240], tickets.iloc[240:]
    model = KerasIntentClassifier(embedding_dim=16, lstm_units=16, epochs=15, batch_size=32)
    return model.fit(train["text"], train["intent"], val["text"], val["intent"])


def test_learns_the_toy_task(trained: KerasIntentClassifier, tickets: pd.DataFrame) -> None:
    holdout = tickets.iloc[240:]
    accuracy = np.mean(np.array(trained.predict(holdout["text"])) == holdout["intent"].to_numpy())
    assert accuracy > 0.8
    assert trained.predict(["Where is my order ORD-12345?"]) == ["track_order"]


def test_probabilities_and_classes(trained: KerasIntentClassifier, tickets: pd.DataFrame) -> None:
    probs = trained.predict_proba(["refund order ORD-1 please", "hello"])
    assert probs.shape == (2, len(trained.classes_))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert set(trained.classes_) == set(tickets["intent"])  # string labels survive encoding
    intent, confidence = trained.predict_with_confidence("cancel order ORD-77")
    assert intent in trained.classes_ and 0.0 < confidence <= 1.0


def test_early_stopping_ends_training_when_validation_stops_improving(
    tickets: pd.DataFrame,
) -> None:
    train, val = tickets.iloc[:240], tickets.iloc[240:]
    # Shuffled validation labels cannot be learned, so validation loss soon stops improving.
    shuffled = val["intent"].sample(frac=1.0, random_state=0).to_numpy()
    model = KerasIntentClassifier(embedding_dim=8, lstm_units=8, epochs=40, patience=1)
    model.fit(train["text"], train["intent"], val["text"], list(shuffled))
    assert 2 <= model.epochs_run < 40
    assert len(model.history["val_loss"]) == model.epochs_run


def test_encoding_pads_truncates_and_marks_unknown_words(
    trained: KerasIntentClassifier,
) -> None:
    ids = trained._encode(["where is my order zzzunknownzzz", ""])
    assert ids.shape == (2, trained._max_len)
    assert OOV_ID in ids[0]
    assert (ids[1] == PAD_ID).all()


def test_save_and_load_round_trip(trained: KerasIntentClassifier, tmp_path: Path) -> None:
    folder = trained.save(tmp_path / "bilstm")
    loaded = KerasIntentClassifier.load(folder)
    sample = ["Where is my order ORD-1?", "refund order ORD-2 please", "unhappy with ORD-3"]
    assert loaded.predict(sample) == trained.predict(sample)
    assert np.allclose(loaded.predict_proba(sample), trained.predict_proba(sample), atol=1e-5)
    assert loaded.classes_ == trained.classes_


def test_errors(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="not trained"):
        KerasIntentClassifier().predict(["hi"])
    with pytest.raises(ModelError):
        KerasIntentClassifier().fit(["a", "b"], ["x"])
    with pytest.raises(ModelError, match="both val_texts and val_labels"):
        KerasIntentClassifier().fit(["a", "b"], ["x", "y"], val_texts=["a"])
    with pytest.raises(ModelError, match="No saved model"):
        KerasIntentClassifier.load(tmp_path / "nothing")


def test_run_training_end_to_end(tickets: pd.DataFrame, tmp_path: Path) -> None:
    folder = tmp_path / "processed" / "toy"
    folder.mkdir(parents=True)
    tickets[["text", "intent"]].iloc[:240].to_csv(folder / "train.csv", index=False)
    tickets[["text", "intent"]].iloc[240:].to_csv(folder / "val.csv", index=False)
    (folder / "SPLIT_INFO.json").write_text('{"synthetic": true}', encoding="utf-8")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    metrics = run_training(
        "toy",
        processed_root=tmp_path / "processed",
        models_root=tmp_path / "models",
        tracking_uri=uri,
        artifact_root=tmp_path / "mlartifacts",
        epochs=10,
        embedding_dim=8,
        lstm_units=8,
    )
    assert metrics["val_accuracy"] > metrics["majority_baseline_val_accuracy"]
    assert (tmp_path / "models" / "intent_bilstm_toy" / "model.keras").exists()

    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("supportx-intent")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert runs[0].data.params["model"] == "bilstm"
    assert runs[0].data.metrics["epochs_run"] >= 1


def test_run_training_reports_missing_data(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="prepare"):
        run_training("nope", processed_root=tmp_path, models_root=tmp_path)
