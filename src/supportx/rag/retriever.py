"""Find the policy passages that best answer a question."""

from dataclasses import dataclass
from pathlib import Path

from supportx.core import DataError, get_logger
from supportx.rag.chunking import Chunk, split_markdown
from supportx.rag.embeddings import Embedder, TfidfEmbedder
from supportx.rag.index import VectorIndex

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float  # cosine similarity, 0..1 (higher = closer)


def load_documents(directory: Path) -> dict[str, str]:
    """Read every *.md file in a folder: {file name: text}."""
    if not directory.is_dir():
        raise DataError(f"Document folder not found: {directory}")
    documents = {p.name: p.read_text(encoding="utf-8") for p in sorted(directory.glob("*.md"))}
    if not documents:
        raise DataError(f"No .md documents found in {directory}")
    return documents


class Retriever:
    def __init__(self, embedder: Embedder, index: VectorIndex):
        self._embedder = embedder
        self._index = index

    @classmethod
    def from_documents(
        cls, directory: Path, max_chars: int = 800, overlap: int = 100
    ) -> "Retriever":
        """Chunk the documents, fit the local embedder on them, and index everything."""
        chunks: list[Chunk] = []
        for name, text in load_documents(directory).items():
            chunks.extend(split_markdown(text, name, max_chars, overlap))
        if not chunks:
            raise DataError(f"The documents in {directory} contain no text to index.")
        embedder = TfidfEmbedder().fit([c.embedding_text for c in chunks])
        index = VectorIndex(embedder.dim)
        index.add(chunks, embedder.embed([c.embedding_text for c in chunks]))
        logger.info("Indexed %d chunks from %s", len(chunks), directory)
        return cls(embedder, index)

    def retrieve(self, query: str, k: int = 3, min_score: float = 0.0) -> list[RetrievedChunk]:
        """Best chunks for a question. Chunks scoring below `min_score` are dropped, so
        "no good answer in the documents" is a possible (and important) outcome."""
        vector = self._embedder.embed([query])[0]
        return [
            RetrievedChunk(chunk, score)
            for chunk, score in self._index.search(vector, k)
            if score >= min_score and score > 0.0
        ]
