"""Retrieval-Augmented Generation: chunking, embeddings, FAISS index, retriever."""

from supportx.rag.chunking import Chunk, split_markdown
from supportx.rag.embeddings import Embedder, TfidfEmbedder
from supportx.rag.index import VectorIndex
from supportx.rag.retriever import RetrievedChunk, Retriever, load_documents

__all__ = [
    "Chunk",
    "Embedder",
    "RetrievedChunk",
    "Retriever",
    "TfidfEmbedder",
    "VectorIndex",
    "load_documents",
    "split_markdown",
]
