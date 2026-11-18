from pathlib import Path

import pandas as pd
import pytest

from supportx.core import ConfigurationError
from supportx.ml.final_test import run_final_test
from supportx.ml.intent import IntentClassifier

TEXTS = ["where is my order", "cancel my order", "track my package", "stop my order please"] * 5
LABELS = ["track", "cancel", "track", "cancel"] * 5


def _setup(tmp_path: Path, c: float = 10.0) -> tuple[Path, Path]:
    data, models = tmp_path / "data", tmp_path / "models"
    (data / "toy").mkdir(parents=True)
    models.mkdir()
    for split in ("train", "test"):
        pd.DataFrame({"text": TEXTS, "intent": LABELS}).to_csv(data / "toy" / f"{split}.csv")
    IntentClassifier(c=c).fit(TEXTS, LABELS).save(models / "intent_tfidf_lr_toy.joblib")
    return data, models


def test_reports_once_then_refuses(tmp_path: Path) -> None:
    data, models = _setup(tmp_path)
    result = run_final_test("toy", data, models)
    assert result["n_test"] == 20
    assert (models / "intent_tfidf_lr_toy.test_report.json").exists()
    with pytest.raises(ConfigurationError, match="already used"):
        run_final_test("toy", data, models)
    assert run_final_test("toy", data, models, force=True)["n_test"] == 20


def test_rejects_a_model_that_is_not_the_reference(tmp_path: Path) -> None:
    data, models = _setup(tmp_path, c=100.0)
    with pytest.raises(ConfigurationError, match="not the reference"):
        run_final_test("toy", data, models)
    assert not (models / "intent_tfidf_lr_toy.test_report.json").exists()
