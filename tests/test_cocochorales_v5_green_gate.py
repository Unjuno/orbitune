"""Regression tests for the CocoChorales v5 GREEN gate.

The v5 GREEN gate has 9 conjunctive conditions, all of which must be
PASS for a row to be admitted. The static evaluation is fed by the
documented upstream chain; the per-row evaluation is fed by row dicts.

This test pins both the static PASS verdict and a handful of negative
cases so that future edits to ``tools/cocochorales_train_corpus_provenance.py``
cannot silently lower the bar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
sys.path.insert(0, str(REPO_ROOT))

from tools.cocochorales_train_corpus_provenance import (  # type: ignore
    _eval_static_conditions,
    _per_row_verdict,
)


def test_static_evaluation_all_nine_conditions_pass() -> None:
    rows = _eval_static_conditions()
    assert len(rows) == 9
    for r in rows:
        assert r["verdict"].startswith("PASS"), f"{r['id']} failed: {r['evidence']}"


def test_static_evaluation_g8_is_pending_install_check() -> None:
    rows = _eval_static_conditions()
    g8 = next(r for r in rows if r["id"] == "G8")
    # G8 is verified at install time, not at audit time.
    assert g8["verdict"] == "PASS_PENDING_INSTALL_CHECK"


def test_static_evaluation_ids_match_documented_gate() -> None:
    rows = _eval_static_conditions()
    expected_ids = {f"G{i}" for i in range(1, 10)}
    assert {r["id"] for r in rows} == expected_ids


def test_per_row_green_when_all_nine_pass() -> None:
    row = {
        "filename": "string_track001010.tfrecord",
        "license": "cc-by-4.0",
        "license_version": "4.0",
        "composer": "J.S. Bach (style; generated)",
        "work": "Four-part chorale (Coconet-generated)",
        "source_url": "https://magenta.withgoogle.com/datasets/cocochorales",
        "year": 2022,
        "generation_method": "coconet+urmp",
        "parse_status": "ok",
    }
    v = _per_row_verdict(row, known_bwv_buckets=set())
    assert v["green"] is True
    for c in v["conditions"]:
        assert c["verdict"] == "PASS", f"{c['id']} failed: {c['evidence']}"


@pytest.mark.parametrize("bad_license", ["cc-by-nc-4.0", "cc-by-nd-4.0", "cc-by-nc-nd-4.0"])
def test_per_row_rejects_nc_or_nd(bad_license: str) -> None:
    row = {
        "filename": "x.tfrecord",
        "license": bad_license,
        "license_version": "4.0",
        "composer": "J.S. Bach (style; generated)",
        "work": "Four-part chorale (Coconet-generated)",
        "source_url": "https://magenta.withgoogle.com/datasets/cocochorales",
        "year": 2022,
        "generation_method": "coconet+urmp",
        "parse_status": "ok",
    }
    v = _per_row_verdict(row, known_bwv_buckets=set())
    assert v["green"] is False
    # At least one of G1, G4, G5, G6 must FAIL.
    failed = {c["id"] for c in v["conditions"] if c["verdict"] == "FAIL"}
    assert failed.intersection({"G1", "G4", "G5", "G6"})


def test_per_row_rejects_imnsf_marker() -> None:
    row = {
        "filename": "x.tfrecord",
        "license": "imnsf-2026",
        "license_version": "n/a",
        "composer": "?",
        "work": "?",
        "source_url": "",
        "year": None,
        "generation_method": "coconet+urmp",
        "parse_status": "ok",
    }
    v = _per_row_verdict(row, known_bwv_buckets=set())
    assert v["green"] is False
    failed = {c["id"] for c in v["conditions"] if c["verdict"] == "FAIL"}
    assert "G5" in failed
    assert "G7" in failed


def test_per_row_rejects_verbatim_transcription() -> None:
    row = {
        "filename": "x.tfrecord",
        "license": "cc-by-4.0",
        "license_version": "4.0",
        "composer": "J.S. Bach (style; generated)",
        "work": "Four-part chorale (Coconet-generated)",
        "source_url": "https://magenta.withgoogle.com/datasets/cocochorales",
        "year": 2022,
        "generation_method": "verbatim",
        "parse_status": "ok",
    }
    v = _per_row_verdict(row, known_bwv_buckets=set())
    assert v["green"] is False
    failed = {c["id"] for c in v["conditions"] if c["verdict"] == "FAIL"}
    assert "G3" in failed


@pytest.mark.parametrize(
    "bad_filename", ["bad/quote.mid", "bad<pipe>.mid", "x\ry.mid", "x\ny.mid", ""]
)
def test_per_row_rejects_windows_illegal_filename(bad_filename: str) -> None:
    row = {
        "filename": bad_filename,
        "license": "cc-by-4.0",
        "license_version": "4.0",
        "composer": "J.S. Bach (style; generated)",
        "work": "Four-part chorale (Coconet-generated)",
        "source_url": "https://magenta.withgoogle.com/datasets/cocochorales",
        "year": 2022,
        "generation_method": "coconet+urmp",
        "parse_status": "ok",
    }
    v = _per_row_verdict(row, known_bwv_buckets=set())
    assert v["green"] is False
    failed = {c["id"] for c in v["conditions"] if c["verdict"] == "FAIL"}
    assert "G8" in failed


def test_per_row_rejects_parse_failure() -> None:
    row = {
        "filename": "x.tfrecord",
        "license": "cc-by-4.0",
        "license_version": "4.0",
        "composer": "J.S. Bach (style; generated)",
        "work": "Four-part chorale (Coconet-generated)",
        "source_url": "https://magenta.withgoogle.com/datasets/cocochorales",
        "year": 2022,
        "generation_method": "coconet+urmp",
        "parse_status": "failed",
    }
    v = _per_row_verdict(row, known_bwv_buckets=set())
    assert v["green"] is False
    failed = {c["id"] for c in v["conditions"] if c["verdict"] == "FAIL"}
    assert "G9" in failed


def test_per_row_rejects_missing_attribution() -> None:
    row = {
        "filename": "x.tfrecord",
        "license": "cc-by-4.0",
        "license_version": "4.0",
        "composer": "",
        "work": "",
        "source_url": "",
        "year": None,
        "generation_method": "coconet+urmp",
        "parse_status": "ok",
    }
    v = _per_row_verdict(row, known_bwv_buckets=set())
    assert v["green"] is False
    failed = {c["id"] for c in v["conditions"] if c["verdict"] == "FAIL"}
    assert "G7" in failed
