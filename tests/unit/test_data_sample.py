from pathlib import Path

import pandas as pd

from supportx.data.sample import (
    COLUMNS,
    INTENTS,
    SENTIMENT_SUFFIX,
    generate_sample_tickets,
    write_sample_csv,
)


def test_shape_columns_and_synthetic_flag() -> None:
    df = generate_sample_tickets(50)
    assert list(df.columns) == COLUMNS
    assert len(df) == 50
    assert df["is_synthetic"].all()


def test_labels_are_from_known_sets() -> None:
    df = generate_sample_tickets(200)
    assert set(df["intent"]) <= set(INTENTS)
    assert set(df["sentiment"]) <= set(SENTIMENT_SUFFIX)
    assert set(df["priority"]) <= {"low", "medium", "high"}


def test_same_seed_is_deterministic_and_different_seed_differs() -> None:
    a = generate_sample_tickets(30, seed=1)
    b = generate_sample_tickets(30, seed=1)
    c = generate_sample_tickets(30, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_priority_rule() -> None:
    df = generate_sample_tickets(500)
    angry_refund = df[(df["intent"] == "get_refund") & (df["sentiment"] == "negative")]
    assert (angry_refund["priority"] == "high").all()
    calm_policy = df[(df["intent"] == "check_refund_policy") & (df["sentiment"] != "negative")]
    assert (calm_policy["priority"] == "low").all()


def test_write_csv_round_trip(tmp_path: Path) -> None:
    out = write_sample_csv(tmp_path / "sub" / "t.csv", n=10, seed=3)
    loaded = pd.read_csv(out)
    assert len(loaded) == 10
    assert loaded["is_synthetic"].all()
