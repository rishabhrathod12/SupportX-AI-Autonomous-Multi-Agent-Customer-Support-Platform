"""Hybrid retrieval: several rankers vote, and their rankings are fused (reciprocal rank fusion).

HONEST SCOPE: every ranker here is LEXICAL (words, or character fragments). None understands
meaning, so "on its way" still does not match "shipped". The design leaves a slot for a SEMANTIC
ranker (an embedding model): implement `Ranker.scores` with it and add it to `HybridRetriever`;
nothing else changes. Whether the lexical mix helps is MEASURED, not assumed (EXPERIMENTS.md, E9).

Rankers produce one score per chunk. The fused order comes from ranks only (RRF), so scores from
different rankers never need to be comparable. The score reported on each result is the PRIMARY
retriever's cosine similarity for that chunk, because that is what the answer cutoff (`min_score`)
and the "hand over to a person" decision were calibrated on.
"""

import math
from collections import Counter
from collections.abc import Collection, Sequence
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from supportx.rag.chunking import Chunk
from supportx.rag.embeddings import make_analyzer
from supportx.rag.retriever import RetrievedChunk, Retriever


class Ranker(Protocol):
    def scores(self, query: str) -> np.ndarray:
        """One relevance score per chunk, in the order of the chunks the ranker was built on."""
        ...


class BM25Ranker:
    """Okapi BM25 over stemmed words: the classic keyword ranker (rewards rare matching words,
    damps repetition, normalizes for passage length)."""

    def __init__(
        self, chunks: Sequence[Chunk], heading_repeat: int = 2, k1: float = 1.5, b: float = 0.75
    ):
        analyze = make_analyzer(stem=True)
        self._analyze = analyze
        self._docs = [Counter(analyze(c.embedding_text_with(heading_repeat))) for c in chunks]
        self._lengths = np.array([sum(d.values()) for d in self._docs], dtype=np.float64)
        self._avg = float(self._lengths.mean()) or 1.0
        self._k1, self._b = k1, b
        n = len(self._docs)
        document_frequency: Counter[str] = Counter()
        for doc in self._docs:
            document_frequency.update(doc.keys())
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        terms = set(self._analyze(query))
        out = np.zeros(len(self._docs))
        for i, doc in enumerate(self._docs):
            norm = self._k1 * (1 - self._b + self._b * self._lengths[i] / self._avg)
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    out[i] += self._idf[term] * tf * (self._k1 + 1) / (tf + norm)
        return out


class CharNgramRanker:
    """TF-IDF over 3-5 character fragments inside words: tolerant of typos and word forms
    ("refunding" ~ "refund", "adress" ~ "address") without any dictionary."""

    def __init__(self, chunks: Sequence[Chunk], heading_repeat: int = 2):
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True
        )
        self._matrix = self._vectorizer.fit_transform(
            [c.embedding_text_with(heading_repeat) for c in chunks]
        )

    def scores(self, query: str) -> np.ndarray:
        vector = self._vectorizer.transform([query])
        return np.asarray((self._matrix @ vector.T).todense()).ravel()


def reciprocal_rank_fusion(
    score_lists: Sequence[np.ndarray], weights: Sequence[float], k: int = 60
) -> np.ndarray:
    """Fuse rankings: each ranker gives `weight / (k + rank)` to each chunk it scores above zero."""
    fused = np.zeros(len(score_lists[0]))
    for scores, weight in zip(score_lists, weights, strict=True):
        order = np.argsort(-scores, kind="stable")
        for rank, index in enumerate(order, start=1):
            if scores[index] > 0:
                fused[index] += weight / (k + rank)
    return fused


class HybridRetriever:
    """The primary `Retriever` plus extra rankers, fused by rank."""

    def __init__(
        self,
        primary: Retriever,
        extra: Sequence[tuple[Ranker, float]],
        primary_weight: float = 1.0,
        rrf_k: int = 60,
    ) -> None:
        self._primary = primary
        self._chunks = primary.chunks
        self._extra = list(extra)
        self._weights = [primary_weight] + [w for _, w in self._extra]
        self._k = rrf_k
        self._position = {c.id: i for i, c in enumerate(self._chunks)}

    def _primary_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self._chunks))
        for hit in self._primary.retrieve(query, k=len(self._chunks)):
            scores[self._position[hit.chunk.id]] = hit.score
        return scores

    def retrieve(
        self,
        query: str,
        k: int = 3,
        min_score: float = 0.0,
        sources: Collection[str] | None = None,
    ) -> list[RetrievedChunk]:
        primary = self._primary_scores(query)
        lists = [primary] + [ranker.scores(query) for ranker, _ in self._extra]
        fused = reciprocal_rank_fusion(lists, self._weights, self._k)
        order = np.argsort(-fused, kind="stable")
        results: list[RetrievedChunk] = []
        for index in order:
            if fused[index] <= 0:
                break
            chunk = self._chunks[int(index)]
            if sources is not None and chunk.source not in sources:
                continue
            score = float(primary[index])
            if score < min_score:
                continue
            results.append(RetrievedChunk(chunk, score))
            if len(results) == k:
                break
        return results
