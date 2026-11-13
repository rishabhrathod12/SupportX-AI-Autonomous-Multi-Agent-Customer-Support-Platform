import hashlib
import json
from pathlib import Path

import httpx
import pytest

from supportx.core import DataError, ExternalServiceError
from supportx.data.download import REGISTRY, download_dataset

PAYLOAD = b"col\nvalue\n"


def mock_client(status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=PAYLOAD)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_registry_pins_commits_and_declares_licenses() -> None:
    for spec in REGISTRY.values():
        assert len(spec.revision) == 40  # a full commit SHA, never "main"
        assert spec.license_id


def test_refuses_without_license_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="cdla-sharing-1.0"):
        download_dataset("bitext", tmp_path, accept_license=False, client=mock_client())
    assert not any(tmp_path.iterdir())  # nothing created


def test_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="Unknown dataset"):
        download_dataset("nope", tmp_path, accept_license=True, client=mock_client())


def test_download_writes_file_and_manifest(tmp_path: Path) -> None:
    files = download_dataset("bitext", tmp_path, accept_license=True, client=mock_client())
    assert files[0].read_bytes() == PAYLOAD
    manifest = json.loads((tmp_path / "bitext" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["license"] == "cdla-sharing-1.0"
    assert manifest["synthetic"] is True
    assert manifest["files"][files[0].name] == hashlib.sha256(PAYLOAD).hexdigest()


def test_failed_download_leaves_no_partial_file(tmp_path: Path) -> None:
    with pytest.raises(ExternalServiceError):
        download_dataset("banking77", tmp_path, accept_license=True, client=mock_client(404))
    assert list((tmp_path / "banking77").glob("*")) == []
