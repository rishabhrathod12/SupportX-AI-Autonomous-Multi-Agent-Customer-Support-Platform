"""FAISS vector index over document chunks."""

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from supportx.core import DataError
from supportx.rag.chunking import Chunk

_VECTORS_FILE = "vectors.faiss"
_CHUNKS_FILE = "chunks.json"


class VectorIndex:
    """Exact (brute-force) inner-product search. Plenty fast for thousands of chunks."""

    def __init__(self, dim: int):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._index: faiss.Index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def chunks(self) -> Sequence[Chunk]:
        return self._chunks

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape != (len(chunks), self._dim):
            raise ValueError(
                f"vectors must have shape {(len(chunks), self._dim)}, got {vectors.shape}"
            )
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._chunks.extend(chunks)

    def search(self, vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """The k best chunks with their similarity score, best first."""
        if k <= 0:
            raise ValueError("k must be positive")
        if vector.shape != (self._dim,):
            raise ValueError(f"query vector must have shape {(self._dim,)}, got {vector.shape}")
        if not self._chunks:
            return []
        query = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
        scores, ids = self._index.search(query, min(k, len(self._chunks)))
        return [
            (self._chunks[i], float(score))
            for score, i in zip(scores[0], ids[0], strict=True)
            if i != -1
        ]

    def save(self, directory: Path) -> Path:
        """Write vectors and chunks. Bytes go through Python file I/O, not faiss.write_index:
        FAISS's own file functions can fail on Windows paths with non-ASCII characters."""
        directory.mkdir(parents=True, exist_ok=True)
        (directory / _VECTORS_FILE).write_bytes(faiss.serialize_index(self._index).tobytes())
        payload = {"dim": self._dim, "chunks": [asdict(c) for c in self._chunks]}
        (directory / _CHUNKS_FILE).write_text(json.dumps(payload), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        vectors_file, chunks_file = directory / _VECTORS_FILE, directory / _CHUNKS_FILE
        if not vectors_file.exists() or not chunks_file.exists():
            raise DataError(f"No saved index found in {directory}")
        payload = json.loads(chunks_file.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in payload["chunks"]]
        index = faiss.deserialize_index(np.frombuffer(vectors_file.read_bytes(), dtype=np.uint8))
        if index.ntotal != len(chunks) or index.d != payload["dim"]:
            raise DataError("The saved index is inconsistent (vectors and chunks do not match).")
        instance = cls(payload["dim"])
        instance._index = index
        instance._chunks = chunks
        return instance
