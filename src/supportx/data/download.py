"""License-gated download of the public datasets chosen in docs/DATASETS.md.

Run:  python -m supportx.data.download bitext --accept-license
Files are fetched from a pinned Hugging Face commit (reproducible), saved under data/raw/<name>/,
and recorded in a MANIFEST.json with their SHA-256 hashes.
"""

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

from supportx.core import (
    DataError,
    ExternalServiceError,
    SupportXError,
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)

_HF_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo: str
    revision: str  # pinned commit SHA, not "main"
    files: tuple[str, ...]  # paths inside the repo
    license_id: str
    license_note: str
    synthetic: bool | None  # True / False, or None when the source does not state it


REGISTRY: dict[str, DatasetSpec] = {
    "bitext": DatasetSpec(
        name="bitext",
        repo="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        revision="430d1a89bd93bd1fa23c16f29dd53e73f0087443",
        files=("Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",),
        license_id="cdla-sharing-1.0",
        license_note="Share-alike terms apply to shared data. Read the license first.",
        synthetic=True,  # card describes it as a "hybrid synthetic dataset"
    ),
    "banking77": DatasetSpec(
        name="banking77",
        repo="PolyAI/banking77",
        revision="1846984189f2bc0ac51c6fc3a600fa9e1714e0c6",
        files=(
            "data/train-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
            "README.md",  # holds the label id -> name list
        ),
        license_id="cc-by-4.0",
        license_note="Attribution required: Casanueva et al., 2020 (arXiv:2003.04807).",
        synthetic=None,  # data provenance is not stated on the card
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(
    name: str,
    dest_root: Path,
    accept_license: bool,
    client: httpx.Client | None = None,
) -> list[Path]:
    """Download one registered dataset. Refuses to run unless the license is acknowledged."""
    spec = REGISTRY.get(name)
    if spec is None:
        raise DataError(f"Unknown dataset '{name}'. Available: {', '.join(sorted(REGISTRY))}.")
    if not accept_license:
        raise DataError(
            f"Dataset '{name}' is licensed {spec.license_id}. {spec.license_note} "
            "Re-run with --accept-license after you have read the license."
        )

    dest = dest_root / spec.name
    dest.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=60)
    saved: list[Path] = []
    try:
        for remote_path in spec.files:
            target = dest / Path(remote_path).name
            if target.exists():
                logger.info("Skipping %s (already exists)", target)
            else:
                url = _HF_URL.format(repo=spec.repo, revision=spec.revision, path=remote_path)
                _fetch(http, url, target)
            saved.append(target)
    finally:
        if owns_client:
            http.close()

    manifest = {
        "name": spec.name,
        "source_repo": spec.repo,
        "revision": spec.revision,
        "license": spec.license_id,
        "synthetic": spec.synthetic,
        "files": {p.name: _sha256(p) for p in saved},
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Dataset '%s' ready in %s", name, dest)
    return saved


def _fetch(client: httpx.Client, url: str, target: Path) -> None:
    """Stream a file to <target>.part, then rename, so a failed download never looks complete."""
    part = target.with_name(target.name + ".part")
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with part.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
    except httpx.HTTPError as exc:
        part.unlink(missing_ok=True)
        raise ExternalServiceError(f"Download failed for {url}: {exc}") from exc
    part.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a public dataset into data/raw/.")
    parser.add_argument("name", choices=sorted(REGISTRY))
    parser.add_argument("--dest", type=Path, default=Path("data/raw"))
    parser.add_argument("--accept-license", action="store_true")
    args = parser.parse_args()
    setup_logging("INFO")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # hides long signed CDN URLs
    try:
        download_dataset(args.name, args.dest, args.accept_license)
    except SupportXError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
