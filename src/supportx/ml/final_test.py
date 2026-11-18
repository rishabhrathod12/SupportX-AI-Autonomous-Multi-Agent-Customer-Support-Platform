"""One-time evaluation of the chosen reference intent model on the held-out TEST split.

The reference model (TF-IDF + LogReg C=10) was chosen using validation scores only. Running this
reads the test split. To keep the test set honest it writes `<dataset>.test_report.json` next to
the model and REFUSES to run again if that file exists (pass --force to overwrite, and say so in
the write-up). Never tune anything after looking at these numbers.

    python -m supportx.ml.final_test bitext
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from supportx.core import ConfigurationError, get_logger, setup_logging
from supportx.core.model_info import describe_model_file
from supportx.data.splits import load_split
from supportx.ml.evaluate import evaluate_classification, per_class_report
from supportx.ml.intent import IntentClassifier

logger = get_logger(__name__)
LABEL_COLUMN = "intent"
EXPECTED_C = 10.0


def run_final_test(
    dataset: str,
    processed_root: Path = Path("data/processed"),
    models_root: Path = Path("artifacts/models"),
    force: bool = False,
) -> dict[str, object]:
    model_path = models_root / f"intent_tfidf_lr_{dataset}.joblib"
    out_path = models_root / f"intent_tfidf_lr_{dataset}.test_report.json"
    if out_path.exists() and not force:
        raise ConfigurationError(
            f"{out_path} already exists: the test split was already used for this model. "
            "Use --force only if you will disclose the re-run."
        )

    model = IntentClassifier.load(model_path)
    if model.params.get("model") != "logreg" or model.params.get("c") != EXPECTED_C:
        raise ConfigurationError(f"{model_path} is not the reference model (logreg, C=10).")

    train = load_split(processed_root, dataset, "train", LABEL_COLUMN)
    test = load_split(processed_root, dataset, "test", LABEL_COLUMN)
    predictions = model.predict(test["text"])

    majority = Counter(train[LABEL_COLUMN]).most_common(1)[0][0]
    result: dict[str, object] = {
        "dataset": dataset,
        "model": describe_model_file(model_path),
        "params": model.params,
        "n_test": len(test),
        "test_scores": evaluate_classification(test[LABEL_COLUMN], predictions),
        "majority_baseline": evaluate_classification(test[LABEL_COLUMN], [majority] * len(test)),
        "per_class": per_class_report(test[LABEL_COLUMN], predictions),
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Test scores for %s: %s", dataset, result["test_scores"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-time test-split evaluation.")
    parser.add_argument("dataset", choices=["bitext", "banking77"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    setup_logging("INFO")
    run_final_test(args.dataset, force=args.force)
