"""Intent classifier: TF-IDF features + Logistic Regression (classical-ML baseline).

Same contract the deep-learning models will follow later:
fit -> predict / predict_proba -> save / load.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Self

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from supportx.core import ModelError, get_logger
from supportx.nlp import tokenize

logger = get_logger(__name__)


class IntentClassifier:
    """Predicts a customer's intent (e.g. 'track_order') from the message text."""

    def __init__(self, c: float = 10.0, seed: int = 42, class_weight: str | None = "balanced"):
        self.params = {"c": c, "seed": seed, "class_weight": class_weight}
        self._pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        tokenizer=tokenize,  # our shared cleaning + tokenizing
                        lowercase=False,  # already done by clean_text
                        token_pattern=None,  # silences the "unused token_pattern" warning
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=c, max_iter=1000, class_weight=class_weight, random_state=seed
                    ),
                ),
            ]
        )
        self._fitted = False

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> Self:
        if len(texts) != len(labels) or len(texts) == 0:
            raise ModelError("texts and labels must be non-empty and the same length.")
        self._pipeline.fit(list(texts), list(labels))
        self._fitted = True
        logger.info(
            "Trained IntentClassifier on %d messages, %d intents", len(texts), len(self.classes_)
        )
        return self

    @property
    def classes_(self) -> list[str]:
        self._require_fitted()
        return [str(c) for c in self._pipeline.classes_]

    def predict(self, texts: Sequence[str]) -> list[str]:
        self._require_fitted()
        return [str(p) for p in self._pipeline.predict(list(texts))]

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Probability per intent; columns follow `classes_`."""
        self._require_fitted()
        return np.asarray(self._pipeline.predict_proba(list(texts)))

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        """Best intent and its probability for one message."""
        probs = self.predict_proba([text])[0]
        best = int(np.argmax(probs))
        return self.classes_[best], float(probs[best])

    def save(self, path: Path) -> Path:
        self._require_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a model saved by `save`. Only load files you created yourself (pickle format)."""
        if not path.exists():
            raise ModelError(f"Model file not found: {path}")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise ModelError(f"{path} does not contain an {cls.__name__}.")
        return model

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ModelError("IntentClassifier is not trained yet. Call fit() or load() first.")
