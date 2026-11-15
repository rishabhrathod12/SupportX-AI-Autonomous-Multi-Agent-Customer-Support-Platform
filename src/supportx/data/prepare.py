"""Turn the raw downloaded datasets into clean, leak-free train/validation/test files.

Run:  python -m supportx.data.prepare all
Reads data/raw/<name>/, writes data/processed/<name>/{train,val,test}.csv and SPLIT_INFO.json.

Rules that protect the honesty of our metrics:
  1. Duplicates are removed on the CLEANED text BEFORE splitting.
  2. A text that appears with two different labels is dropped completely (label conflict).
  3. Splits are stratified by label and reproducible (fixed seed).
  4. Train, validation and test must not share any cleaned text (checked, raises DataError).
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from supportx.core import DataError, get_logger, setup_logging
from supportx.nlp import clean_text

logger = get_logger(__name__)

SEED = 42
_CLEAN = "_clean"  # temporary key column, never written to disk


def dedupe_text(
    df: pd.DataFrame, text_col: str, label_col: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop duplicate texts (judged after cleaning) and texts with conflicting labels."""
    out = df.copy()
    out[_CLEAN] = out[text_col].map(clean_text)
    labels_per_text = out.groupby(_CLEAN)[label_col].transform("nunique")
    conflicts = int((labels_per_text > 1).sum())
    out = out[labels_per_text == 1]
    before = len(out)
    out = out.drop_duplicates(subset=_CLEAN, keep="first")
    stats = {"conflicting_rows_dropped": conflicts, "duplicate_rows_dropped": before - len(out)}
    return out.reset_index(drop=True), stats


def stratified_split(
    df: pd.DataFrame, label_col: str, val_size: float, test_size: float, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Split into (train, val, test) keeping label proportions. test_size=0 returns test=None."""
    if test_size > 0:
        rest, test = train_test_split(
            df, test_size=test_size, stratify=df[label_col], random_state=seed
        )
    else:
        rest, test = df, None
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        rest, test_size=relative_val, stratify=rest[label_col], random_state=seed
    )
    return train, val, test


def assert_no_overlap(**splits: pd.DataFrame) -> None:
    """Raise DataError if any cleaned text appears in more than one split."""
    names = list(splits)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = set(splits[first][_CLEAN]) & set(splits[second][_CLEAN])
            if shared:
                raise DataError(f"{len(shared)} texts appear in both '{first}' and '{second}'.")


def parse_banking77_labels(readme_text: str) -> dict[int, str]:
    """Read the 'id: name' label list from the dataset card's YAML header."""
    start = readme_text.find("class_label:")
    if start == -1:
        raise DataError("Banking77 README has no class_label section.")
    names = {int(i): n for i, n in re.findall(r"^\s+'(\d+)': (\S+)\s*$", readme_text[start:], re.M)}
    if sorted(names) != list(range(len(names))) or len(names) != 77:
        raise DataError(f"Expected 77 contiguous label ids, found {len(names)}.")
    return names


def _write(out_dir: Path, splits: dict[str, pd.DataFrame], info: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.drop(columns=[_CLEAN]).to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
    info["rows"] = {name: len(frame) for name, frame in splits.items()}
    (out_dir / "SPLIT_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.info("Wrote %s: %s", out_dir, info["rows"])


def prepare_bitext(raw_root: Path, out_root: Path, seed: int = SEED) -> None:
    """Bitext (SYNTHETIC data): dedupe, then stratified 80/10/10 split by intent."""
    csv_files = list((raw_root / "bitext").glob("*.csv"))
    if not csv_files:
        raise DataError("Bitext CSV not found. Run: python -m supportx.data.download bitext")
    raw = pd.read_csv(csv_files[0], encoding="utf-8")
    df = raw.rename(columns={"instruction": "text"})[["text", "intent", "category"]]
    df, stats = dedupe_text(df, "text", "intent")
    train, val, test = stratified_split(df, "intent", val_size=0.1, test_size=0.1, seed=seed)
    assert test is not None
    assert_no_overlap(train=train, val=val, test=test)
    info: dict[str, object] = {
        "dataset": "bitext",
        "synthetic": True,
        "seed": seed,
        "raw_rows": len(raw),
        **stats,
    }
    _write(out_root / "bitext", {"train": train, "val": val, "test": test}, info)


def prepare_banking77(raw_root: Path, out_root: Path, seed: int = SEED) -> None:
    """Banking77: use the official test split; carve validation from train; remove leakage."""
    folder = raw_root / "banking77"
    try:
        train_raw = pd.read_parquet(next(folder.glob("train*.parquet")))
        test_raw = pd.read_parquet(next(folder.glob("test*.parquet")))
        names = parse_banking77_labels((folder / "README.md").read_text(encoding="utf-8"))
    except (StopIteration, FileNotFoundError) as exc:
        raise DataError("Banking77 files not found. Run the download command first.") from exc

    train_raw["intent"] = train_raw["label"].map(names)
    test_raw["intent"] = test_raw["label"].map(names)
    train, train_stats = dedupe_text(train_raw, "text", "label")
    test = test_raw.copy()  # official test split is kept exactly as published
    test[_CLEAN] = test["text"].map(clean_text)
    leaked = train[_CLEAN].isin(set(test[_CLEAN]))
    train = train[~leaked].reset_index(drop=True)  # test stays official and untouched
    train, val, _ = stratified_split(train, "label", val_size=0.1, test_size=0.0, seed=seed)
    assert_no_overlap(train=train, val=val, test=test)
    info: dict[str, object] = {
        "dataset": "banking77",
        "synthetic": None,
        "seed": seed,
        "raw_rows": {"train": len(train_raw), "test": len(test_raw)},
        "train_dedupe": train_stats,
        "test_duplicate_rows_kept": int(test[_CLEAN].duplicated().sum()),
        "train_rows_removed_for_test_overlap": int(leaked.sum()),
    }
    _write(out_root / "banking77", {"train": train, "val": val, "test": test}, info)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets into data/processed/.")
    parser.add_argument("name", choices=["bitext", "banking77", "all"])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    setup_logging("INFO")
    try:
        if args.name in ("bitext", "all"):
            prepare_bitext(args.raw, args.out)
        if args.name in ("banking77", "all"):
            prepare_banking77(args.raw, args.out)
    except DataError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
