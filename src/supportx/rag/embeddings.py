"""Turn text into vectors. All embedders return L2-normalized float32 rows, so a dot product
equals cosine similarity (which is what the FAISS inner-product index computes)."""

from collections.abc import Callable, Sequence
from typing import Protocol, Self

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from supportx.core import ModelError
from supportx.nlp import tokenize


def light_stem(word: str) -> str:
    """Tiny suffix stripper so 'cancel', 'cancelling' and 'cancelled' match ("returns" = "return").

    Deliberately crude (no dictionary, no extra dependency); placeholders like <url> are left alone.
    """
    if word.startswith("<") or len(word) <= 3:
        return word
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]  # cancell -> cancel, shipp -> ship
    return word


def make_analyzer(stem: bool = True, drop_stop_words: bool = False) -> Callable[[str], list[str]]:
    """Text -> word tokens, the way the lexical rankers see it (shared so they all agree)."""

    def analyze(text: str) -> list[str]:
        words = tokenize(text)
        if drop_stop_words:
            words = [w for w in words if w not in ENGLISH_STOP_WORDS]
        return [light_stem(w) for w in words] if stem else words

    return analyze


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Shape (len(texts), dim), dtype float32, each row L2-normalized (or all zeros)."""
        ...


class TfidfEmbedder:
    """Local, offline, LEXICAL embedder: it matches shared words, not meaning.

    "send back" will not match "return". Good for tests and a first baseline; a real
    embedding model (added in a later phase) replaces it behind the same interface.
    Fit it on the document chunks before use.
    """

    def __init__(self, stem: bool = False, drop_stop_words: bool = False) -> None:
        self._vectorizer = TfidfVectorizer(
            tokenizer=make_analyzer(stem, drop_stop_words),
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> Self:
        if not texts:
            raise ModelError("Cannot fit an embedder on zero texts.")
        self._vectorizer.fit(list(texts))
        self._fitted = True
        return self

    @property
    def dim(self) -> int:
        self._require_fitted()
        return len(self._vectorizer.vocabulary_)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._require_fitted()
        matrix = self._vectorizer.transform(list(texts)).toarray()  # rows are already L2-normalized
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ModelError("TfidfEmbedder is not fitted yet. Call fit() first.")
