"""Intent classifier: TF-IDF features + a classical model (Logistic Regression or XGBoost).

Same contract the deep-learning models will follow later:
fit -> predict / predict_proba -> save / load.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from supportx.core import ModelError, get_logger
from supportx.nlp import tokenize

logger = get_logger(__name__)

ModelKind = Literal["logreg", "xgboost"]


class IntentClassifier:
    """Predicts a customer's intent (e.g. 'track_order') from the message text.

    `c` and `class_weight` apply to logreg only; `n_estimators` applies to xgboost only.
    """

    def __init__(
        self,
        model: ModelKind = "logreg",
        c: float = 10.0,
        seed: int = 42,
        class_weight: str | None = "balanced",
        n_estimators: int = 200,
    ):
        if model not in ("logreg", "xgboost"):
            raise ModelError(f"Unknown model '{model}'. Use 'logreg' or 'xgboost'.")
        self.params: dict[str, Any] = {"model": model, "seed": seed}
        if model == "logreg":
            self.params |= {"c": c, "class_weight": class_weight}
            clf: Any = LogisticRegression(
                C=c, max_iter=1000, class_weight=class_weight, random_state=seed
            )
        else:
            self.params |= {"n_estimators": n_estimators}
            clf = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=6,
                learning_rate=0.2,
                subsample=0.8,
                colsample_bytree=0.5,
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
            )
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
                ("clf", clf),
            ]
        )
        # XGBoost needs integer labels, so every model trains on encoded labels (0..n-1).
        self._encoder = LabelEncoder()
        self._fitted = False

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> Self:
        if len(texts) != len(labels) or len(texts) == 0:
            raise ModelError("texts and labels must be non-empty and the same length.")
        encoded = self._encoder.fit_transform(list(labels))
        self._pipeline.fit(list(texts), encoded)
        self._fitted = True
        logger.info(
            "Trained %s on %d messages, %d intents",
            self.params["model"],
            len(texts),
            len(self.classes_),
        )
        return self

    @property
    def classes_(self) -> list[str]:
        self._require_fitted()
        return [str(c) for c in self._encoder.classes_]

    def predict(self, texts: Sequence[str]) -> list[str]:
        self._require_fitted()
        encoded = np.asarray(self._pipeline.predict(list(texts)), dtype=int)
        return [str(p) for p in self._encoder.inverse_transform(encoded)]

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
