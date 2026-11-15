"""Classification metrics shared by all ML and DL classifiers."""

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def evaluate_classification(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    """Accuracy plus macro and weighted scores.

    Macro averages treat every class equally (good for imbalanced data);
    weighted averages follow class sizes.
    """
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def per_class_report(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, dict[str, float]]:
    """Precision / recall / F1 / support for each class."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {k: v for k, v in report.items() if isinstance(v, dict)}
