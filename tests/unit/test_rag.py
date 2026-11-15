from pathlib import Path

import numpy as np
import pytest

from supportx.core import DataError, ModelError
from supportx.rag import (
    Chunk,
    Retriever,
    TfidfEmbedder,
    VectorIndex,
    load_documents,
    split_markdown,
)

POLICIES = Path(__file__).resolve().parents[2] / "data" / "sample" / "policies"

DOC = """# Policy

Intro text.

## Returns
You can return items within 30 days.

## Shipping
### Express
Express takes 2 days.
"""


# ---------------------------------------------------------------- chunking
def test_headings_become_section_paths() -> None:
    chunks = split_markdown(DOC, "p.md")
    assert [(c.heading, c.text) for c in chunks] == [
        ("Policy", "Intro text."),
        ("Policy > Returns", "You can return items within 30 days."),
        ("Policy > Shipping > Express", "Express takes 2 days."),
    ]
    assert [c.id for c in chunks] == ["p.md#0", "p.md#1", "p.md#2"]
    assert all(c.source == "p.md" for c in chunks)


def test_a_sibling_heading_replaces_the_previous_one() -> None:
    chunks = split_markdown("# A\n## B\nb text\n## C\nc text\n", "x.md")
    assert [c.heading for c in chunks] == ["A > B", "A > C"]


def test_embedding_text_includes_the_heading() -> None:
    chunk = Chunk(id="x#0", source="x", heading="Refunds > Window", text="30 days.")
    assert chunk.embedding_text == "Refunds > Window\n30 days."
    assert Chunk(id="x#1", source="x", heading="", text="plain").embedding_text == "plain"


def test_paragraphs_are_packed_up_to_the_limit() -> None:
    paragraphs = ["a" * 60, "b" * 60, "c" * 60]
    chunks = split_markdown("# T\n\n" + "\n\n".join(paragraphs), "t.md", max_chars=130, overlap=10)
    assert [c.text for c in chunks] == [f"{'a' * 60}\n\n{'b' * 60}", "c" * 60]
    assert all(len(c.text) <= 130 for c in chunks)


def test_a_long_paragraph_is_cut_into_overlapping_windows_at_spaces() -> None:
    words = " ".join(f"w{i:03d}" for i in range(120))  # one long paragraph
    chunks = split_markdown(words, "long.md", max_chars=200, overlap=40)
    assert len(chunks) > 2
    assert all(len(c.text) <= 200 for c in chunks)
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert earlier.text.split()[-1] in later.text  # overlap: the cut word appears again
    rejoined = set(" ".join(c.text for c in chunks).split())
    assert rejoined == set(words.split())  # no word was lost or cut in half


def test_empty_and_heading_only_documents_give_no_chunks() -> None:
    assert split_markdown("", "e.md") == []
    assert split_markdown("# Only a heading\n## Another\n", "e.md") == []


def test_invalid_chunk_settings() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        split_markdown("x", "x.md", max_chars=10)
    with pytest.raises(ValueError, match="overlap"):
        split_markdown("x", "x.md", max_chars=100, overlap=80)


# ---------------------------------------------------------------- embedder
def test_embeddings_are_normalized_float32_rows() -> None:
    embedder = TfidfEmbedder().fit(["refund policy days", "shipping address change"])
    vectors = embedder.embed(["refund days", "shipping"])
    assert vectors.dtype == np.float32 and vectors.shape == (2, embedder.dim)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_unknown_words_give_a_zero_vector() -> None:
    embedder = TfidfEmbedder().fit(["refund policy"])
    assert not embedder.embed(["zzz qqq"]).any()


def test_embedder_must_be_fitted() -> None:
    with pytest.raises(ModelError, match="not fitted"):
        TfidfEmbedder().embed(["x"])
    with pytest.raises(ModelError):
        TfidfEmbedder().fit([])


# ---------------------------------------------------------------- index
def make_index() -> tuple[VectorIndex, list[Chunk]]:
    chunks = [Chunk(id=f"d#{i}", source="d", heading="", text=f"t{i}") for i in range(3)]
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0.6, 0.8, 0]], dtype=np.float32)
    index = VectorIndex(3)
    index.add(chunks, vectors)
    return index, chunks


def test_search_returns_best_first_with_scores() -> None:
    index, chunks = make_index()
    results = index.search(np.array([0, 1, 0], dtype=np.float32), k=2)
    assert [c.id for c, _ in results] == ["d#1", "d#2"]
    assert results[0][1] == pytest.approx(1.0) and results[1][1] == pytest.approx(0.8)
    assert len(index) == 3 and index.dim == 3


def test_k_larger_than_the_index_is_fine() -> None:
    index, _ = make_index()
    assert len(index.search(np.array([1, 0, 0], dtype=np.float32), k=50)) == 3
    assert VectorIndex(3).search(np.zeros(3, dtype=np.float32), k=2) == []


def test_index_rejects_bad_shapes() -> None:
    index, chunks = make_index()
    with pytest.raises(ValueError, match="shape"):
        index.add(chunks, np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="query vector"):
        index.search(np.zeros(5, dtype=np.float32), k=1)
    with pytest.raises(ValueError, match="k must"):
        index.search(np.zeros(3, dtype=np.float32), k=0)
    with pytest.raises(ValueError):
        VectorIndex(0)


def test_save_and_load_round_trip_even_in_a_folder_with_a_dash(tmp_path: Path) -> None:
    index, _ = make_index()
    folder = index.save(tmp_path / "index – v1")  # the real project path has an en dash too
    loaded = VectorIndex.load(folder)
    query = np.array([0.6, 0.8, 0], dtype=np.float32)
    assert [(c.id, round(s, 5)) for c, s in loaded.search(query, 3)] == [
        (c.id, round(s, 5)) for c, s in index.search(query, 3)
    ]
    assert len(loaded) == 3


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="No saved index"):
        VectorIndex.load(tmp_path / "missing")
    index, _ = make_index()
    folder = index.save(tmp_path / "bad")
    (folder / "chunks.json").write_text('{"dim": 3, "chunks": []}', encoding="utf-8")
    with pytest.raises(DataError, match="inconsistent"):
        VectorIndex.load(folder)


# ---------------------------------------------------------------- retriever on the demo policies
@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever.from_documents(POLICIES)


def test_documents_are_loaded_and_errors_are_clear(tmp_path: Path) -> None:
    assert sorted(load_documents(POLICIES)) == [
        "cancellation_policy.md",
        "refund_policy.md",
        "shipping_policy.md",
    ]
    with pytest.raises(DataError, match="not found"):
        load_documents(tmp_path / "nope")
    with pytest.raises(DataError, match="No .md"):
        load_documents(tmp_path)
    (tmp_path / "empty.md").write_text("# Only a heading\n", encoding="utf-8")
    with pytest.raises(DataError, match="no text"):
        Retriever.from_documents(tmp_path)


# Hand-written BEFORE looking at results: (question, source file that should answer it).
# A small regression check, not a benchmark: 6 questions cannot measure real retrieval quality.
QUESTIONS = [
    ("How many days do I have to return an item?", "refund_policy.md"),
    ("Are final sale items refundable?", "refund_policy.md"),
    ("Who approves my refund request?", "refund_policy.md"),
    ("How long does express delivery take?", "shipping_policy.md"),
    ("Can I change my delivery address?", "shipping_policy.md"),
    ("Can I cancel my order after it has shipped?", "cancellation_policy.md"),
]


@pytest.mark.parametrize(("question", "expected_source"), QUESTIONS)
def test_the_right_document_is_in_the_top_three(
    retriever: Retriever, question: str, expected_source: str
) -> None:
    """The Knowledge Agent will hand the top 3 passages to the LLM, so this is the metric that
    matters: is the right document among them?"""
    top = retriever.retrieve(question, k=3)
    assert expected_source in {r.chunk.source for r in top}
    assert top == sorted(top, key=lambda r: r.score, reverse=True)


def test_top_one_baseline_and_its_known_miss(retriever: Retriever) -> None:
    """Measured baseline of the lexical embedder on the questions above: top-1 is right for 5 of 6.

    The miss is 'Can I cancel my order after it has shipped?': the shipping policy ranks first
    (it shares 'order has shipped') and the right cancellation passage ranks second, because
    'cancel' does not match 'cancelling'/'cancelled' (no stemming). A semantic embedding model
    should fix this; if this test starts failing because the number IMPROVED, update the baseline.
    """
    misses = [q for q, source in QUESTIONS if retriever.retrieve(q, k=1)[0].chunk.source != source]
    assert misses == ["Can I cancel my order after it has shipped?"]


def test_a_question_unrelated_to_the_documents_finds_nothing(retriever: Retriever) -> None:
    assert retriever.retrieve("zebra giraffe quantum", k=3) == []


def test_min_score_filters_weak_matches(retriever: Retriever) -> None:
    question = "How many days do I have to return an item?"
    everything = retriever.retrieve(question, k=5)
    strict = retriever.retrieve(question, k=5, min_score=everything[0].score - 1e-6)
    assert len(strict) == 1 and strict[0].chunk.id == everything[0].chunk.id


def test_the_vocabulary_gap_of_the_lexical_embedder(retriever: Retriever) -> None:
    """Documents a KNOWN limit: with no shared words there is nothing to match."""
    assert retriever.retrieve("send my purchase back", k=3, min_score=0.2) == []
