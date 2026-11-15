"""Turn text into vectors. All embedders return L2-normalized float32 rows, so a dot product
equals cosine similarity (which is what the FAISS inner-product index computes)."""

from collections.abc import Sequence
from typing import Protocol, Self

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from supportx.core import ModelError
from supportx.nlp import tokenize


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

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
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
