"""Retrieval metrics and a scorer for hand-written questions (`data/sample/retrieval_eval_v2.csv`).

Each question lists its RELEVANT sections as `file::Section` (several are allowed), or `NONE` for a
question the documents cannot answer. For answerable questions, over the ranked sections:

  Recall@K     relevant sections found in the top K, divided by how many are relevant
  Precision@K  relevant sections found in the top K, divided by K (so it tops out at |R|/K when a
               question has fewer than K relevant sections: read it next to Recall@K, not alone)
  Hit@K        1 if any relevant section is in the top K
  MRR          mean of 1 / (rank of the first relevant section), 0 when none is ranked

For UNANSWERABLE questions the right behaviour is to NOT answer (hand over to a person). The system
"answers" when its best passage scores at least `min_score`, so:

  false_answer_rate   unanswerable questions that were answered anyway (lower is better)
  answerable_answered answerable questions that were answered (higher is better)
  answered_correct    answerable questions answered AND ranked a relevant section first

    python -m supportx.rag.metrics --split dev
"""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from supportx.rag.retriever import RetrievedChunk, Retriever

EVAL_V2_PATH = Path("data/sample/retrieval_eval_v2.csv")
LEGACY_PATH = Path("data/sample/retrieval_eval.csv")
POLICY_DIR = Path("data/sample/policies")
NONE = "NONE"

Ranker = Callable[[str], list[RetrievedChunk]]


def section_key(hit: RetrievedChunk) -> str:
    """`file::Section` of a passage (the last part of its heading path)."""
    return f"{hit.chunk.source}::{hit.chunk.heading.split('>')[-1].strip()}"


def ranked_sections(hits: Sequence[RetrievedChunk]) -> list[str]:
    """Section keys in rank order, each section once (a section may span several chunks)."""
    seen: list[str] = []
    for hit in hits:
        key = section_key(hit)
        if key not in seen:
            seen.append(key)
    return seen


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / k


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for position, key in enumerate(ranked, start=1):
        if key in relevant:
            return 1.0 / position
    return 0.0


def score_ranker(
    rank: Ranker, rows: pd.DataFrame, min_score: float = 0.1, ks: Sequence[int] = (1, 3)
) -> dict[str, Any]:
    """Score `rank(question)` (best first, NO score cutoff applied) on the question rows."""
    answerable: dict[str, list[float]] = {}
    unanswerable_answered = unanswerable = answered = answered_correct = 0
    misses: list[tuple[str, str, float]] = []

    def add(name: str, value: float) -> None:
        answerable.setdefault(name, []).append(value)

    for question, relevant_text in zip(rows["question"], rows["relevant"], strict=True):
        hits = rank(question)
        ranked = ranked_sections(hits)
        best = hits[0].score if hits else 0.0
        answers = best >= min_score
        if relevant_text == NONE:
            unanswerable += 1
            unanswerable_answered += answers
            if answers:
                misses.append((question, f"answered with {ranked[0]}", best))
            continue
        relevant = set(relevant_text.split(";"))
        for k in ks:
            add(f"recall@{k}", recall_at_k(ranked, relevant, k))
            add(f"precision@{k}", precision_at_k(ranked, relevant, k))
            add(f"hit@{k}", hit_at_k(ranked, relevant, k))
        add("mrr", reciprocal_rank(ranked, relevant))
        answered += answers
        correct = bool(ranked) and ranked[0] in relevant
        answered_correct += answers and correct
        if not correct:
            misses.append((question, f"top: {ranked[0] if ranked else '-'}", best))
    n_answerable = len(rows) - unanswerable
    result: dict[str, Any] = {name: sum(v) / len(v) for name, v in answerable.items()}
    result.update(
        {
            "n_answerable": n_answerable,
            "n_unanswerable": unanswerable,
            "answerable_answered": answered / n_answerable if n_answerable else 0.0,
            "answered_correct": answered_correct / n_answerable if n_answerable else 0.0,
            "false_answer_rate": unanswerable_answered / unanswerable if unanswerable else 0.0,
            "misses": misses,
        }
    )
    return result


def load_questions(split: str, include_legacy: bool = False) -> pd.DataFrame:
    """The v2 questions of a split. `include_legacy` adds the 32 earlier answerable questions
    (their old holdout was already scored once, so they only ever count as DEV material)."""
    rows = pd.read_csv(EVAL_V2_PATH, encoding="utf-8")
    rows = rows[rows["split"] == split][["question", "relevant"]]
    if include_legacy:
        old = pd.read_csv(LEGACY_PATH, encoding="utf-8")
        old["relevant"] = old["source"] + "::" + old["section"]
        rows = pd.concat([rows, old[["question", "relevant"]]], ignore_index=True)
    return rows.reset_index(drop=True)


def single_ranker(retriever: Retriever) -> Ranker:
    return lambda question: retriever.retrieve(question, k=50, min_score=0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score retrieval with ranking metrics.")
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--legacy", action="store_true", help="add the 32 earlier dev questions")
    parser.add_argument("--show-misses", action="store_true")
    parser.add_argument(
        "--embedder",
        choices=["tfidf", "openai"],
        default="tfidf",
        help="openai needs OPENAI_API_KEY and OPENAI_EMBEDDING_MODEL and makes paid API calls",
    )
    args = parser.parse_args()
    semantic = None
    if args.embedder == "openai":
        from supportx.core import get_settings
        from supportx.llm import LLMClient, OpenAIEmbedder

        semantic = OpenAIEmbedder(LLMClient.from_settings(get_settings()))
    result = score_ranker(
        single_ranker(Retriever.from_documents(POLICY_DIR, embedder=semantic)),
        load_questions(args.split, args.legacy),
        args.min_score,
    )
    misses = result.pop("misses")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in result.items()})
    if args.show_misses:
        for question, detail, score in misses:
            print(f"  {score:.2f} {detail}  <-  {question}")
