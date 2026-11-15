"""Near-duplicate detection: how similar is each evaluation text to its closest training text?

Exact-duplicate removal misses near-twins ("cancel order" vs "cancel the order"). If many
validation/test texts have a near-twin in train, scores are inflated, so we measure it.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from supportx.core import DataError
from supportx.nlp import tokenize


def nearest_similarity(
    reference: Sequence[str], queries: Sequence[str], chunk_size: int = 500
) -> np.ndarray:
    """For each query, the highest cosine similarity (0..1) against any reference text.

    Uses TF-IDF word 1-2-grams fitted on the reference texts. 1.0 means the same words
    after cleaning; values near 0 mean nothing in common.
    """
    if len(reference) == 0:
        raise DataError("reference texts are empty.")
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        lowercase=False,
        token_pattern=None,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    ref = vectorizer.fit_transform(list(reference))
    qry = vectorizer.transform(list(queries))
    ref_t = ref.T.tocsc()
    best = np.zeros(qry.shape[0])
    for start in range(0, qry.shape[0], chunk_size):  # chunked to bound memory
        sims = qry[start : start + chunk_size] @ ref_t
        best[start : start + chunk_size] = sims.max(axis=1).toarray().ravel()
    return np.clip(best, 0.0, 1.0)
