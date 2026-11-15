import json
from pathlib import Path

import pandas as pd
import pytest

from supportx.core import DataError
from supportx.data.prepare import (
    _CLEAN,
    assert_no_overlap,
    dedupe_text,
    parse_banking77_labels,
    prepare_banking77,
    prepare_bitext,
    stratified_split,
)


def test_dedupe_removes_duplicates_after_cleaning() -> None:
    df = pd.DataFrame(
        {"text": ["Where is my order?", "where is   my ORDER?", "hi"], "y": ["a", "a", "b"]}
    )
    out, stats = dedupe_text(df, "text", "y")
    assert len(out) == 2
    assert stats == {"conflicting_rows_dropped": 0, "duplicate_rows_dropped": 1}


def test_dedupe_drops_label_conflicts_completely() -> None:
    df = pd.DataFrame({"text": ["same text", "SAME text", "ok"], "y": ["a", "b", "c"]})
    out, stats = dedupe_text(df, "text", "y")
    assert out["text"].tolist() == ["ok"]
    assert stats["conflicting_rows_dropped"] == 2


def _labeled(n_per_class: int = 20) -> pd.DataFrame:
    rows = [(f"msg {c} {i}", c) for c in "abc" for i in range(n_per_class)]
    df = pd.DataFrame(rows, columns=["text", "y"])
    df[_CLEAN] = df["text"]
    return df


def test_split_sizes_stratified_and_disjoint() -> None:
    train, val, test = stratified_split(_labeled(), "y", val_size=0.1, test_size=0.2)
    assert test is not None
    assert (len(train), len(val), len(test)) == (42, 6, 12)
    for part in (train, val, test):
        assert part["y"].value_counts().nunique() == 1  # equal share of every class
    assert not (set(train.index) & set(val.index)) and not (set(train.index) & set(test.index))


def test_split_is_reproducible_and_seed_sensitive() -> None:
    a = stratified_split(_labeled(), "y", 0.1, 0.1, seed=1)[0]
    b = stratified_split(_labeled(), "y", 0.1, 0.1, seed=1)[0]
    c = stratified_split(_labeled(), "y", 0.1, 0.1, seed=2)[0]
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_split_without_test() -> None:
    train, val, test = stratified_split(_labeled(), "y", val_size=0.1, test_size=0.0)
    assert test is None
    assert len(train) + len(val) == 60


def test_overlap_guard() -> None:
    a = pd.DataFrame({_CLEAN: ["x", "y"]})
    b = pd.DataFrame({_CLEAN: ["z"]})
    assert_no_overlap(train=a, test=b)  # fine
    with pytest.raises(DataError, match="train.*test"):
        assert_no_overlap(train=a, test=pd.DataFrame({_CLEAN: ["y"]}))


def test_parse_banking77_labels() -> None:
    body = "\n".join(f"          '{i}': label_{i}" for i in range(77))
    readme = f"features:\n  label:\n      class_label:\n        names:\n{body}\n  splits:\n"
    names = parse_banking77_labels(readme)
    assert names[0] == "label_0" and names[76] == "label_76" and len(names) == 77
    with pytest.raises(DataError):
        parse_banking77_labels("no labels here")
    with pytest.raises(DataError):
        parse_banking77_labels("class_label:\n  names:\n    '0': only_one\n")


def test_prepare_bitext_end_to_end(tmp_path: Path) -> None:
    rows = [
        (f"please help with {c} request number {i} {{{{Order Number}}}}", c, "CAT")
        for c in ("track_order", "get_refund")
        for i in range(30)
    ]
    rows.append((rows[0][0], "track_order", "CAT"))  # an exact duplicate
    raw = tmp_path / "raw" / "bitext"
    raw.mkdir(parents=True)
    pd.DataFrame(rows, columns=["instruction", "intent", "category"]).assign(
        flags="B", response="r"
    ).to_csv(raw / "b.csv", index=False)

    prepare_bitext(tmp_path / "raw", tmp_path / "out", seed=0)

    out = tmp_path / "out" / "bitext"
    info = json.loads((out / "SPLIT_INFO.json").read_text(encoding="utf-8"))
    assert info["synthetic"] is True and info["duplicate_rows_dropped"] == 1
    assert sum(info["rows"].values()) == 60
    frames = {n: pd.read_csv(out / f"{n}.csv") for n in ("train", "val", "test")}
    assert list(frames["train"].columns) == ["text", "intent", "category"]
    assert not (set(frames["train"]["text"]) & set(frames["test"]["text"]))


def test_prepare_banking77_end_to_end(tmp_path: Path) -> None:
    names = "\n".join(f"          '{i}': label_{i}" for i in range(77))
    raw = tmp_path / "raw" / "banking77"
    raw.mkdir(parents=True)
    (raw / "README.md").write_text(
        f"class_label:\n        names:\n{names}\n  splits:\n", encoding="utf-8"
    )
    train = pd.DataFrame(
        {"text": [f"train {i} {i % 77}" for i in range(770)], "label": [i % 77 for i in range(770)]}
    )
    test = pd.DataFrame({"text": ["test one", train.loc[0, "text"]], "label": [0, 0]})  # 2nd leaks
    train.to_parquet(raw / "train-0.parquet")
    test.to_parquet(raw / "test-0.parquet")

    prepare_banking77(tmp_path / "raw", tmp_path / "out", seed=0)

    out = tmp_path / "out" / "banking77"
    info = json.loads((out / "SPLIT_INFO.json").read_text(encoding="utf-8"))
    assert info["train_rows_removed_for_test_overlap"] == 1
    assert len(pd.read_csv(out / "test.csv")) == 2  # official test untouched
    tr = pd.read_csv(out / "train.csv")
    assert train.loc[0, "text"] not in set(tr["text"])
    assert {"text", "label", "intent"} <= set(tr.columns)
    assert tr["intent"].str.startswith("label_").all()


def test_prepare_reports_missing_raw_data(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="download"):
        prepare_bitext(tmp_path, tmp_path / "out")
    with pytest.raises(DataError, match="download"):
        prepare_banking77(tmp_path, tmp_path / "out")
