"""Retrieval metrics (checked against hand-computed values), rankers, hybrid fusion, filters."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from supportx.agents import build_ticket_service
from supportx.agents.adapters import build_real_agents
from supportx.core import Settings
from supportx.rag import Chunk, Retriever
from supportx.rag.embeddings import light_stem, make_analyzer
from supportx.rag.hybrid import (
    BM25Ranker,
    CharNgramRanker,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from supportx.rag.metrics import (
    hit_at_k,
    load_questions,
    precision_at_k,
    ranked_sections,
    recall_at_k,
    reciprocal_rank,
    score_ranker,
)
from supportx.rag.retriever import RetrievedChunk
from tests.unit.helpers import POLICIES

POLICY_DIR = POLICIES


# ---------------------------------------------------------------- the metric functions
def test_metric_functions_match_hand_computed_values() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 4) == 1.0
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert precision_at_k(ranked, relevant, 4) == 0.5
    assert hit_at_k(ranked, relevant, 1) == 0.0 and hit_at_k(ranked, relevant, 2) == 1.0
    assert reciprocal_rank(ranked, relevant) == 0.5  # first relevant is at rank 2
    assert reciprocal_rank(["x", "y"], {"z"}) == 0.0


def chunk(name: str, source: str = "f.md") -> Chunk:
    return Chunk(id=f"{source}#{name}", source=source, heading=f"Doc > {name}", text=name)


def test_sections_are_ranked_once_each_in_order() -> None:
    hits = [
        RetrievedChunk(chunk("A"), 0.9),
        RetrievedChunk(chunk("A"), 0.8),
        RetrievedChunk(chunk("B"), 0.5),
    ]
    assert ranked_sections(hits) == ["f.md::A", "f.md::B"]


def test_scorer_counts_answers_abstentions_and_misses_exactly() -> None:
    rows = pd.DataFrame(
        {
            "question": [
                "right first",
                "right second",
                "unanswerable answered",
                "unanswerable refused",
            ],
            "relevant": ["f.md::A", "f.md::A", "NONE", "NONE"],
        }
    )
    canned = {
        "right first": [RetrievedChunk(chunk("A"), 0.5), RetrievedChunk(chunk("B"), 0.2)],
        "right second": [RetrievedChunk(chunk("B"), 0.4), RetrievedChunk(chunk("A"), 0.3)],
        "unanswerable answered": [RetrievedChunk(chunk("B"), 0.3)],
        "unanswerable refused": [RetrievedChunk(chunk("B"), 0.05)],
    }
    result = score_ranker(lambda q: canned[q], rows, min_score=0.1, ks=(1, 2))
    assert result["n_answerable"] == 2 and result["n_unanswerable"] == 2
    assert result["recall@1"] == 0.5 and result["recall@2"] == 1.0
    assert result["mrr"] == pytest.approx((1.0 + 0.5) / 2)
    assert result["answerable_answered"] == 1.0
    assert result["answered_correct"] == 0.5  # only the first is answered AND right
    assert result["false_answer_rate"] == 0.5  # one of the two unanswerable was answered
    assert len(result["misses"]) == 2  # the second question's top hit, and the answered NONE


def test_a_higher_cutoff_trades_answers_for_fewer_false_answers() -> None:
    rows = pd.DataFrame({"question": ["q", "n"], "relevant": ["f.md::A", "NONE"]})
    canned = {"q": [RetrievedChunk(chunk("A"), 0.25)], "n": [RetrievedChunk(chunk("B"), 0.2)]}
    low = score_ranker(lambda q: canned[q], rows, min_score=0.1)
    high = score_ranker(lambda q: canned[q], rows, min_score=0.22)
    assert (low["answerable_answered"], low["false_answer_rate"]) == (1.0, 1.0)
    assert (high["answerable_answered"], high["false_answer_rate"]) == (1.0, 0.0)


def test_the_frozen_question_sets_have_the_expected_shape() -> None:
    dev, holdout = load_questions("dev"), load_questions("holdout")
    assert len(dev) == 21 and len(holdout) == 20
    assert (dev["relevant"] == "NONE").sum() == 6 and (holdout["relevant"] == "NONE").sum() == 6
    assert len(load_questions("dev", include_legacy=True)) == 21 + 32
    # every relevant section named in the questions exists in the policy documents
    retriever = Retriever.from_documents(POLICY_DIR)
    known = {f"{c.source}::{c.heading.split('>')[-1].strip()}" for c in retriever.chunks}
    for text in pd.concat([dev, holdout])["relevant"]:
        if text != "NONE":
            assert set(text.split(";")) <= known


# ---------------------------------------------------------------- rankers and fusion
def test_rank_fusion_rewards_agreement_and_ignores_zero_scores() -> None:
    first = np.array([0.9, 0.5, 0.0])
    second = np.array([0.1, 0.8, 0.0])
    fused = reciprocal_rank_fusion([first, second], [1.0, 1.0], k=60)
    assert fused[2] == 0.0  # scored zero by every ranker: never retrieved
    assert fused[0] == pytest.approx(1 / 61 + 1 / 62) and fused[1] == pytest.approx(1 / 62 + 1 / 61)
    heavier = reciprocal_rank_fusion([first, second], [2.0, 1.0], k=60)
    assert heavier[0] > heavier[1]  # the first ranker's favourite wins when it counts double


def small_chunks() -> list[Chunk]:
    texts = {
        "Shipping": "Parcels ship within two days and carry a tracking number.",
        "Address": "You can change the delivery address before the order has shipped.",
        "Returns": "Items can be returned within thirty days for a refund.",
    }
    return [Chunk(id=f"d.md#{h}", source="d.md", heading=h, text=t) for h, t in texts.items()]


def test_bm25_puts_the_passage_with_the_rare_matching_word_first() -> None:
    chunks = small_chunks()
    scores = BM25Ranker(chunks).scores("what is the tracking number")
    assert chunks[int(np.argmax(scores))].heading == "Shipping"
    assert BM25Ranker(chunks).scores("zebra giraffe").sum() == 0.0


def test_character_ngrams_survive_a_typo_that_words_cannot() -> None:
    chunks = small_chunks()
    scores = CharNgramRanker(chunks).scores("how do I change my adress")  # misspelled
    assert chunks[int(np.argmax(scores))].heading == "Address"


def test_hybrid_reports_the_primary_cosine_and_respects_filters() -> None:
    primary = Retriever.from_documents(POLICY_DIR)
    hybrid = HybridRetriever(primary, [(BM25Ranker(primary.chunks), 1.0)])
    hits = hybrid.retrieve("Can I cancel my order?", k=3)
    assert 1 <= len(hits) <= 3
    cosine = {h.chunk.id: h.score for h in primary.retrieve("Can I cancel my order?", k=50)}
    assert all(h.score == pytest.approx(cosine.get(h.chunk.id, 0.0)) for h in hits)
    only = hybrid.retrieve("Can I cancel my order?", k=5, sources={"shipping_policy.md"})
    assert only and {h.chunk.source for h in only} == {"shipping_policy.md"}


def test_the_stemming_analyzer_matches_word_forms() -> None:
    analyze = make_analyzer(stem=True)
    assert analyze("Cancelling cancelled orders") == [light_stem("cancelling"), "cancel", "order"]


# ----------------------------------------------------- metadata filter and the cutoff setting
def test_a_source_filter_restricts_results_to_those_documents() -> None:
    retriever = Retriever.from_documents(POLICY_DIR)
    plain = retriever.retrieve("Can I cancel my order?", k=3)
    assert {"cancellation_policy.md"} <= {h.chunk.source for h in plain}
    filtered = retriever.retrieve("Can I cancel my order?", k=3, sources={"refund_policy.md"})
    assert filtered and all(h.chunk.source == "refund_policy.md" for h in filtered)
    assert retriever.retrieve("Can I cancel my order?", k=3, sources=set()) == []


def test_the_cutoff_setting_decides_whether_a_weak_match_is_quoted() -> None:
    from sqlalchemy import create_engine

    retriever = Retriever.from_documents(POLICY_DIR)
    engine = create_engine("sqlite://")
    question = "Do you ship to Australia?"  # the documents cannot answer this
    lenient = build_real_agents(_FakeIntent(), retriever, engine, min_score=0.1)
    strict = build_real_agents(_FakeIntent(), retriever, engine, min_score=0.5)
    assert lenient.search_policy(question) != []  # answers with something loosely related
    assert strict.search_policy(question) == []  # handed to a person instead
    assert Settings(_env_file=None).retrieval_min_score == 0.15  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        Settings(_env_file=None, retrieval_min_score=1.5)  # type: ignore[call-arg]


class _FakeIntent:
    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        return ("check_refund_policy", 0.9)


def test_the_service_builder_passes_the_setting_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sqlalchemy import create_engine

    from supportx.data.sample import generate_sample_tickets
    from supportx.ml import IntentClassifier

    seen: dict[str, Any] = {}
    real = build_real_agents

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr("supportx.agents.tickets.build_real_agents", spy)
    data = generate_sample_tickets(60, seed=1)
    model = IntentClassifier().fit(data["text"], data["intent"]).save(tmp_path / "m.joblib")
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        intent_model_path=str(model),
        policy_docs_dir=str(POLICY_DIR),
        checkpoint_db_path=str(tmp_path / "c.db"),
        retrieval_min_score=0.27,
    )
    _, connection = build_ticket_service(settings, create_engine("sqlite://"))
    connection.close()
    assert seen["min_score"] == 0.27
