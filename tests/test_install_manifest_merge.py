"""Regression tests for the fail-closed install_manifest.json merge
helper in ``scripts.install_pretrain_corpora``. The installer used to
overwrite the existing manifest with a partial source subset, which
silently destroyed provenance for previously installed sources when
callers passed ``--sources foo,bar``. The merge helper must now:

1. Preserve prior source entries not selected in the current invocation.
2. Replace or add entries actually installed now.
3. Fail closed if the existing manifest belongs to an incompatible
   registry lineage.
4. Fail closed on a malformed existing manifest.
5. Fail closed if the same source re-installs with a different
   provenance payload.

These tests exercise the pure ``_merge_install_manifest`` helper so they
do not require any network or heavy local data.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.install_pretrain_corpora import _merge_install_manifest  # type: ignore


def _path() -> Path:
    return Path("/tmp/manifest.json")  # value is irrelevant for the helper


def test_merge_preserves_prior_source_entries() -> None:
    existing = {
        "registry_name": "orbitune-commercial-safe-v3",
        "sources": {
            "pdmx": {"record_id": 15571083, "files": ["PDMX.csv"]},
            "openscore_lieder": {"commit": "deadbeef" * 5},
        },
    }
    installed = {"openscore_lieder": {"commit": "deadbeef" * 5}}
    out = _merge_install_manifest(
        _path(), existing, installed, registry_name="orbitune-commercial-safe-v3"
    )
    # The pdmx entry is preserved even though the new install did not touch it.
    assert "pdmx" in out["sources"]
    assert out["sources"]["pdmx"] == {"record_id": 15571083, "files": ["PDMX.csv"]}
    assert "openscore_lieder" in out["sources"]


def test_merge_adds_new_source() -> None:
    existing = {
        "registry_name": "orbitune-commercial-safe-v3",
        "sources": {"pdmx": {"record_id": 15571083}},
    }
    installed = {"nrg_cp": {"archive_md5": "7443fe30674ef149aa4c23580044f597"}}
    out = _merge_install_manifest(
        _path(), existing, installed, registry_name="orbitune-commercial-safe-v3"
    )
    assert set(out["sources"].keys()) == {"pdmx", "nrg_cp"}


def test_merge_fails_closed_on_registry_name_mismatch() -> None:
    existing = {
        "registry_name": "orbitune-commercial-safe-v3",
        "sources": {"pdmx": {"record_id": 15571083}},
    }
    installed = {"nrg_cp": {"archive_md5": "x"}}
    with pytest.raises(SystemExit, match="refusing to merge"):
        _merge_install_manifest(
            _path(),
            existing,
            installed,
            registry_name="orbitune-commercial-safe-v4",
        )


def test_merge_fails_closed_on_malformed_sources_field() -> None:
    existing = {"registry_name": "orbitune-commercial-safe-v3", "sources": "not-a-dict"}
    installed = {"nrg_cp": {}}
    with pytest.raises(SystemExit, match="'sources' is not a dict"):
        _merge_install_manifest(
            _path(), existing, installed, registry_name="orbitune-commercial-safe-v3"
        )


def test_merge_fails_closed_on_different_provenance_payload() -> None:
    existing = {
        "registry_name": "orbitune-commercial-safe-v3",
        "sources": {"pdmx": {"record_id": 15571083, "files": ["PDMX.csv"]}},
    }
    installed = {"pdmx": {"record_id": 99999999, "files": ["PDMX.csv"]}}
    with pytest.raises(SystemExit, match="different provenance payload"):
        _merge_install_manifest(
            _path(), existing, installed, registry_name="orbitune-commercial-safe-v3"
        )


def test_merge_allows_identical_reinstallation() -> None:
    existing = {
        "registry_name": "orbitune-commercial-safe-v3",
        "sources": {"pdmx": {"record_id": 15571083, "files": ["PDMX.csv"]}},
    }
    installed = {"pdmx": {"record_id": 15571083, "files": ["PDMX.csv"]}}
    out = _merge_install_manifest(
        _path(), existing, installed, registry_name="orbitune-commercial-safe-v3"
    )
    assert out["sources"]["pdmx"] == {"record_id": 15571083, "files": ["PDMX.csv"]}


def test_merge_handles_no_existing_manifest() -> None:
    out = _merge_install_manifest(
        _path(), None, {"pdmx": {"record_id": 1}}, registry_name="orbitune-commercial-safe-v3"
    )
    assert out["sources"] == {"pdmx": {"record_id": 1}}
    assert out["registry_name"] == "orbitune-commercial-safe-v3"
