"""Identify a trained model file, so every log and the staff info endpoint can say exactly which
model made a decision."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict


class ModelInfo(TypedDict):
    file: str
    sha256: str
    size_bytes: int
    modified_utc: str


def describe_model_file(path: Path) -> ModelInfo:
    """File name, SHA-256 (the real identity: same bytes = same model), size and modified time."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return ModelInfo(
        file=path.name,
        sha256=digest.hexdigest(),
        size_bytes=stat.st_size,
        modified_utc=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(timespec="seconds"),
    )
