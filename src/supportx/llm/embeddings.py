"""OpenAI embeddings behind the retriever's `Embedder` interface (semantic, not lexical)."""

from collections.abc import Sequence

import numpy as np

from supportx.core import ModelError
from supportx.llm.client import LLMClient


class OpenAIEmbedder:
    def __init__(self, client: LLMClient, batch_size: int = 64) -> None:
        self._client = client
        self._batch = batch_size
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise ModelError("Embedding size is unknown until the first embed() call.")
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        parts = [
            self._client.embed(texts[i : i + self._batch])
            for i in range(0, len(texts), self._batch)
        ]
        vectors = np.vstack(parts)
        self._dim = int(vectors.shape[1])
        return vectors
