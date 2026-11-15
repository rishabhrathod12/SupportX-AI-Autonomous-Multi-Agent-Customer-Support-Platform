import numpy as np
import pytest

from supportx.core import DataError
from supportx.data.similarity import nearest_similarity

REFERENCE = [
    "where is my order",
    "i want to cancel my order",
    "what is your refund policy",
]


def test_identical_text_scores_one() -> None:
    sims = nearest_similarity(REFERENCE, ["Where is my ORDER?"])  # same words after cleaning
    assert sims[0] == pytest.approx(1.0, abs=1e-6)


def test_unrelated_text_scores_zero() -> None:
    assert nearest_similarity(REFERENCE, ["zebra giraffe"])[0] == 0.0


def test_close_text_scores_between() -> None:
    sims = nearest_similarity(REFERENCE, ["where is the order"])
    assert 0.3 < sims[0] < 1.0


def test_one_score_per_query_and_chunking_is_consistent() -> None:
    queries = ["where is my order", "cancel my order", "refund policy please", "hello"] * 3
    a = nearest_similarity(REFERENCE, queries, chunk_size=2)
    b = nearest_similarity(REFERENCE, queries, chunk_size=1000)
    assert a.shape == (12,)
    assert np.allclose(a, b)


def test_empty_reference_is_rejected() -> None:
    with pytest.raises(DataError):
        nearest_similarity([], ["x"])
