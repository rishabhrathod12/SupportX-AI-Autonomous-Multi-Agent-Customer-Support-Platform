"""Score the retriever on hand-written questions (`data/sample/retrieval_eval.csv`).

For each question the right answer is one policy section (file + heading). Reported:
  top1 / top2 / top3   the right section is ranked first / in the first two / in the first three
  shown_ok      the right section is among the passages a reply would quote (best + close runner-up)
  answered      the best passage scored at least `min_score` (the production cutoff)
  answered_ok   answered AND the best passage is the right section (what the customer would read)

    python -m supportx.rag.evaluate --split dev --show-misses
"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from supportx.rag.retriever import Retriever

EVAL_PATH = Path("data/sample/retrieval_eval.csv")
POLICY_DIR = Path("data/sample/policies")


def score_retrieval(
    retriever: Retriever, rows: pd.DataFrame, min_score: float = 0.1
) -> dict[str, Any]:
    top1 = top2 = top3 = answered = answered_ok = shown_ok = 0
    misses: list[tuple[str, str, float]] = []
    for question, source, section in zip(
        rows["question"], rows["source"], rows["section"], strict=True
    ):
        hits = retriever.retrieve(question, k=3, min_score=0.0)

        def right(hit: Any, source: str = source, section: str = section) -> bool:
            return bool(hit.chunk.source == source and hit.chunk.heading.endswith(section))

        first_ok = bool(hits) and right(hits[0])
        top1 += first_ok
        top2 += any(right(h) for h in hits[:2])
        shown_ok += any(right(h) for h in retriever.retrieve_for_answer(question, min_score))
        top3 += any(right(h) for h in hits)
        if hits and hits[0].score >= min_score:
            answered += 1
            answered_ok += first_ok
        if not first_ok:
            best = hits[0] if hits else None
            got = (
                f"{best.chunk.source} > {best.chunk.heading.split('>')[-1].strip()}"
                if best
                else "-"
            )
            misses.append((question, got, best.score if best else 0.0))
    n = len(rows)
    return {
        "n": n,
        "top1": top1 / n,
        "top2": top2 / n,
        "shown_ok": shown_ok / n,
        "top3": top3 / n,
        "answered": answered / n,
        "answered_ok": answered_ok / n,
        "misses": misses,
    }


def load_eval(split: str, path: Path = EVAL_PATH) -> pd.DataFrame:
    rows = pd.read_csv(path, encoding="utf-8")
    return rows[rows["split"] == split].reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score retrieval on hand-written questions.")
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()
    result = score_retrieval(
        Retriever.from_documents(POLICY_DIR), load_eval(args.split), args.min_score
    )
    misses = result.pop("misses")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in result.items()})
    if args.show_misses:
        for question, got, score in misses:
            print(f"  {score:.2f} {got}  <-  {question}")
