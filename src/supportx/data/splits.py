"""Load the prepared train/validation/test files written by supportx.data.prepare."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from supportx.core import DataError


def load_split(root: Path, dataset: str, split: str, label_column: str = "intent") -> pd.DataFrame:
    """Read data/processed/<dataset>/<split>.csv; rows with a missing text or label are dropped."""
    path = root / dataset / f"{split}.csv"
    if not path.exists():
        raise DataError(f"{path} not found. Run: python -m supportx.data.prepare {dataset}")
    return pd.read_csv(path, encoding="utf-8").dropna(subset=["text", label_column])


def load_split_info(root: Path, dataset: str) -> dict[str, Any]:
    """Read SPLIT_INFO.json (provenance: seed, synthetic flag, row counts)."""
    path = root / dataset / "SPLIT_INFO.json"
    if not path.exists():
        raise DataError(f"{path} not found. Run: python -m supportx.data.prepare {dataset}")
    info = json.loads(path.read_text(encoding="utf-8"))
    return dict(info)
