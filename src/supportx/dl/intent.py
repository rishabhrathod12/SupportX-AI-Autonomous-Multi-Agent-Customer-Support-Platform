"""Deep-learning intent classifier: word embeddings + bidirectional LSTM (Keras).

Same contract as the classical IntentClassifier: fit -> predict / predict_proba -> save / load.
The vocabulary is built here from the training texts, using the shared `tokenize`.
"""

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import keras
import numpy as np
from keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from supportx.core import ModelError, get_logger
from supportx.nlp import tokenize

logger = get_logger(__name__)

PAD_ID = 0  # padding (ignored by the LSTM through masking)
OOV_ID = 1  # word not seen in training
_MODEL_FILE = "model.keras"
_CONFIG_FILE = "config.json"


class KerasIntentClassifier:
    """Embedding -> BiLSTM -> Dropout -> Softmax. Predicts an intent from message text."""

    def __init__(
        self,
        embedding_dim: int = 64,
        lstm_units: int = 64,
        dropout: float = 0.3,
        max_vocab: int = 10_000,
        epochs: int = 30,
        batch_size: int = 64,
        patience: int = 3,
        seed: int = 42,
    ):
        self.params: dict[str, Any] = {
            "embedding_dim": embedding_dim,
            "lstm_units": lstm_units,
            "dropout": dropout,
            "max_vocab": max_vocab,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "seed": seed,
        }
        self._vocab: dict[str, int] = {}
        self._max_len = 0
        self._encoder = LabelEncoder()
        self._model: Any = None
        self.history: dict[str, list[float]] = {}

    # ---- training -------------------------------------------------------------------
    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        val_texts: Sequence[str] | None = None,
        val_labels: Sequence[str] | None = None,
    ) -> Self:
        """Train. If validation data is given, stop early when its loss stops improving."""
        if len(texts) != len(labels) or len(texts) == 0:
            raise ModelError("texts and labels must be non-empty and the same length.")
        if (val_texts is None) != (val_labels is None):
            raise ModelError("Give both val_texts and val_labels, or neither.")
        p = self.params
        keras.utils.set_random_seed(p["seed"])

        self._build_vocab(texts)
        y = self._encoder.fit_transform(list(labels))
        x = self._encode(texts)
        n_classes = len(self._encoder.classes_)
        self._model = self._build_network(n_classes)

        validation = None
        callbacks: list[Any] = []
        if val_texts is not None and val_labels is not None:
            try:
                y_val = self._encoder.transform(list(val_labels))
            except ValueError as exc:
                raise ModelError("Validation labels contain intents unseen in training.") from exc
            validation = (self._encode(val_texts), y_val)
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=p["patience"], restore_best_weights=True
                )
            )

        callbacks.append(
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: logger.info(
                    "epoch %d: %s",
                    epoch + 1,
                    {k: round(float(v), 4) for k, v in (logs or {}).items()},
                )
            )
        )
        weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y)
        result = self._model.fit(
            x,
            y,
            epochs=p["epochs"],
            batch_size=p["batch_size"],
            validation_data=validation,
            class_weight=dict(enumerate(weights)),
            callbacks=callbacks,
            verbose=0,
        )
        self.history = {k: [float(v) for v in vals] for k, vals in result.history.items()}
        logger.info(
            "Trained BiLSTM on %d messages, %d intents, %d epochs (vocab %d, max_len %d)",
            len(texts),
            n_classes,
            self.epochs_run,
            len(self._vocab) + 2,
            self._max_len,
        )
        return self

    @property
    def epochs_run(self) -> int:
        return len(self.history.get("loss", []))

    @property
    def classes_(self) -> list[str]:
        self._require_fitted()
        return [str(c) for c in self._encoder.classes_]

    # ---- inference ------------------------------------------------------------------
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Probability per intent; columns follow `classes_`."""
        self._require_fitted()
        return np.asarray(self._model.predict(self._encode(texts), batch_size=256, verbose=0))

    def predict(self, texts: Sequence[str]) -> list[str]:
        best = np.argmax(self.predict_proba(texts), axis=1)
        return [str(c) for c in self._encoder.inverse_transform(best)]

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        probs = self.predict_proba([text])[0]
        best = int(np.argmax(probs))
        return self.classes_[best], float(probs[best])

    # ---- persistence ----------------------------------------------------------------
    def save(self, directory: Path) -> Path:
        """Write model.keras (network weights) and config.json (vocabulary, labels, settings)."""
        self._require_fitted()
        directory.mkdir(parents=True, exist_ok=True)
        self._model.save(directory / _MODEL_FILE)
        ordered = sorted(self._vocab, key=lambda token: self._vocab[token])
        config = {
            "params": self.params,
            "vocab": ordered,
            "classes": self.classes_,
            "max_len": self._max_len,
            "history": self.history,
        }
        (directory / _CONFIG_FILE).write_text(json.dumps(config), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Load a model written by `save` (Keras safe mode stays on)."""
        model_file, config_file = directory / _MODEL_FILE, directory / _CONFIG_FILE
        if not model_file.exists() or not config_file.exists():
            raise ModelError(f"No saved model found in {directory}")
        config = json.loads(config_file.read_text(encoding="utf-8"))
        instance = cls(**config["params"])
        instance._vocab = {token: i + 2 for i, token in enumerate(config["vocab"])}
        instance._max_len = int(config["max_len"])
        instance._encoder.classes_ = np.array(config["classes"])
        instance.history = config["history"]
        instance._model = keras.models.load_model(model_file)
        return instance

    # ---- internals ------------------------------------------------------------------
    def _build_vocab(self, texts: Sequence[str]) -> None:
        tokenized = [tokenize(t) for t in texts]
        counts = Counter(tok for tokens in tokenized for tok in tokens)
        frequent = sorted(
            (tok for tok, n in counts.items() if n >= 2), key=lambda tok: (-counts[tok], tok)
        )
        self._vocab = {tok: i + 2 for i, tok in enumerate(frequent[: self.params["max_vocab"] - 2])}
        lengths = [len(tokens) for tokens in tokenized]
        self._max_len = int(np.clip(np.percentile(lengths, 99), 4, 64))

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Messages -> fixed-size matrix of word ids (padded with 0, cut at max_len)."""
        matrix = np.zeros((len(texts), self._max_len), dtype="int32")
        for row, text in enumerate(texts):
            ids = [self._vocab.get(tok, OOV_ID) for tok in tokenize(text)][: self._max_len]
            matrix[row, : len(ids)] = ids
        return matrix

    def _build_network(self, n_classes: int) -> Any:
        p = self.params
        inputs = keras.Input(shape=(self._max_len,), dtype="int32")
        x = layers.Embedding(len(self._vocab) + 2, p["embedding_dim"], mask_zero=True)(inputs)
        x = layers.Bidirectional(layers.LSTM(p["lstm_units"]))(x)
        x = layers.Dropout(p["dropout"])(x)
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _require_fitted(self) -> None:
        if self._model is None:
            raise ModelError(
                "KerasIntentClassifier is not trained yet. Call fit() or load() first."
            )
