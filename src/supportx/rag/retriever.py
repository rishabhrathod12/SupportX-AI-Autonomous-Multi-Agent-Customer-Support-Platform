"""Find the policy passages that best answer a question."""

from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

from supportx.core import DataError, get_logger
from supportx.rag.chunking import Chunk, split_markdown
from supportx.rag.embeddings import Embedder, TfidfEmbedder
from supportx.rag.index import VectorIndex
from supportx.rag.query import expand_query

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
    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        query_rewriter: Callable[[str], str] | None = None,
    ):
        self._embedder = embedder
        self._index = index
        self._rewrite = query_rewriter or (lambda question: question)

    @classmethod
    def from_documents(
        cls,
        directory: Path,
        max_chars: int = 800,
        overlap: int = 100,
        stem: bool = True,
        drop_stop_words: bool = False,
        heading_repeat: int = 2,
        embedder: Embedder | None = None,
        expand_queries: bool = False,
    ) -> "Retriever":
        """Chunk the documents, embed them, and index everything.

        Default: fit the local lexical TF-IDF embedder on the chunks. Pass `embedder` (already
        usable, e.g. `OpenAIEmbedder`) for semantic search; it is used as is and not fitted.
        `expand_queries` adds the thesaurus rewrite (`rag.query`) for the lexical embedder."""
        chunks: list[Chunk] = []
        for name, text in load_documents(directory).items():
            chunks.extend(split_markdown(text, name, max_chars, overlap))
        if not chunks:
            raise DataError(f"The documents in {directory} contain no text to index.")
        texts = [c.embedding_text_with(heading_repeat) for c in chunks]
        if embedder is None:
            embedder = TfidfEmbedder(stem=stem, drop_stop_words=drop_stop_words).fit(texts)
        vectors = embedder.embed(texts)  # before `.dim`: some embedders learn it from this call
        index = VectorIndex(embedder.dim)
        index.add(chunks, vectors)
        logger.info("Indexed %d chunks from %s", len(chunks), directory)
        return cls(embedder, index, expand_query if expand_queries else None)

    def retrieve_for_answer(
        self,
        query: str,
        min_score: float = 0.1,
        max_passages: int = 2,
        relative_cutoff: float = 0.5,
    ) -> list[RetrievedChunk]:
        """The passages a reply should quote: the best one, plus the runner-up when it scores at
        least `relative_cutoff` times as high (lexical scores are often nearly tied, and the right
        passage is frequently second). Empty when nothing reaches `min_score`."""
        hits = self.retrieve(query, k=max_passages, min_score=min_score)
        if not hits:
            return []
        floor = hits[0].score * relative_cutoff
        return [hits[0]] + [h for h in hits[1:] if h.score >= floor]

    @property
    def chunks(self) -> list[Chunk]:
        """Every indexed chunk (read-only use: other rankers align with it by chunk id)."""
        return list(self._index.chunks)

    def retrieve(
        self,
        query: str,
        k: int = 3,
        min_score: float = 0.0,
        sources: Collection[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Best chunks for a question. Chunks scoring below `min_score` are dropped, so
        "no good answer in the documents" is a possible (and important) outcome.

        `sources` restricts the search to those document files (metadata filter), e.g.
        {"cancellation_policy.md"} when the intent is already known to be about cancelling."""
        vector = self._embedder.embed([self._rewrite(query)])[0]
        wanted = len(self._index) if sources is not None else k  # filter AFTER ranking everything
        found = [
            RetrievedChunk(chunk, score)
            for chunk, score in self._index.search(vector, wanted)
            if score >= min_score and score > 0.0 and (sources is None or chunk.source in sources)
        ]
        return found[:k]
