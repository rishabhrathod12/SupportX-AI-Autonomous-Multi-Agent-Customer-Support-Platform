import pandas as pd

from supportx.data.augment import CORES, build_augmentation
from supportx.nlp import clean_text


def test_is_deterministic_and_covers_every_intent() -> None:
    a = build_augmentation(per_intent=20, seed=1)
    b = build_augmentation(per_intent=20, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["intent"]) == set(CORES)
    assert (a.groupby("intent").size() == 20).all()
    assert not a.duplicated().any()


def test_probe_texts_never_appear_in_training_rows() -> None:
    cores = [c for texts in CORES.values() for c in texts]
    banned = cores[:10]  # even exact (cleaned) matches of the core wording must be removed
    rows = build_augmentation(per_intent=30, seed=2, drop_texts=banned)
    cleaned = {clean_text(t) for t in rows["text"]}
    assert not cleaned & {clean_text(t) for t in banned}


def test_order_intents_get_order_numbers_that_normalize_to_the_slot_token() -> None:
    rows = build_augmentation(per_intent=60, seed=3)
    tracked = rows[rows["intent"] == "track_order"]["text"]
    assert tracked.str.contains("ORD-").any()
    assert any("<order_number>" in clean_text(t) for t in tracked)
    assert not rows[rows["intent"] == "payment_issue"]["text"].str.contains("ORD-").any()
