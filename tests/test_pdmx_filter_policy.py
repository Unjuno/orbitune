"""Regression tests for the registry-driven PDMX admission policy.

The PDMX admission gate must be controlled entirely by the ``filters``
list in the source registry. The v3 commercial Base policy is
``["no_license_conflict", "deduplicated", "midi_available"]``; the v4
commercial Base policy is ``["no_license_conflict", "midi_available"]``
(the upstream ``deduplicated`` gate is dropped and replaced by an
Orbitune-level cross-source dedup). These tests pin both policies on
synthetic CSV + tiny MIDI fixtures.
"""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from orbitune.pretrain_corpus import iter_pdmx_midi  # type: ignore


V3_FILTERS = ("no_license_conflict", "deduplicated", "midi_available")
V4_FILTERS = ("no_license_conflict", "midi_available")


def _write_pdmx(root: Path, rows: list[dict[str, str]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fields = [
        "mid",
        "subset:no_license_conflict",
        "subset:deduplicated",
        "rating",
        "n_tracks",
        "license",
    ]
    with (root / "PDMX.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})


def _make_midi(path: Path) -> None:
    # Trivial MThd MIDI header: 4 bytes "MThd" + length 6, format 0, 1 track, division 480.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MThd" + b"\x00\x00\x00\x06" + b"\x00\x00" + b"\x00\x01" + b"\x01\xe0")


def _row(mid: str, *, no_conflict: str = "True", dedup: str = "True") -> dict[str, str]:
    return {
        "mid": mid,
        "subset:no_license_conflict": no_conflict,
        "subset:deduplicated": dedup,
        "rating": "4.0",
        "n_tracks": "3",
        "license": "public-domain",
    }


def _admitted(root: Path, filters) -> list[str]:
    out: list[str] = []
    for path, _meta in iter_pdmx_midi(root, filters=filters):
        out.append(str(path.relative_to(root)).replace("\\", "/"))
    return sorted(out)


def test_v3_admits_canonical_row(tmp_path: Path) -> None:
    _make_midi(tmp_path / "a.mid")
    _write_pdmx(tmp_path, [_row("a.mid")])
    assert _admitted(tmp_path, V3_FILTERS) == ["a.mid"]


def test_v4_admits_row_even_when_upstream_dedup_is_false(tmp_path: Path) -> None:
    _make_midi(tmp_path / "a.mid")
    _write_pdmx(tmp_path, [_row("a.mid", dedup="False")])
    # v4 drops the upstream deduplicated gate: this row must be admitted.
    assert _admitted(tmp_path, V4_FILTERS) == ["a.mid"]
    # v3 still drops it: the upstream dedup gate is on.
    assert _admitted(tmp_path, V3_FILTERS) == []


def test_v3_and_v4_both_reject_license_conflict(tmp_path: Path) -> None:
    _make_midi(tmp_path / "a.mid")
    _write_pdmx(tmp_path, [_row("a.mid", no_conflict="False", dedup="True")])
    assert _admitted(tmp_path, V3_FILTERS) == []
    assert _admitted(tmp_path, V4_FILTERS) == []


def test_v3_and_v4_both_reject_missing_midi(tmp_path: Path) -> None:
    # No file written on disk for b.mid.
    _write_pdmx(tmp_path, [_row("b.mid")])
    assert _admitted(tmp_path, V3_FILTERS) == []
    assert _admitted(tmp_path, V4_FILTERS) == []


@pytest.mark.parametrize("empty_mid", ["", "n/a", "NaN", "None"])
def test_midi_available_gate_rejects_empty_or_nan_mid(tmp_path: Path, empty_mid: str) -> None:
    _write_pdmx(tmp_path, [{"mid": empty_mid, "subset:no_license_conflict": "True", "subset:deduplicated": "True"}])
    assert _admitted(tmp_path, V3_FILTERS) == []
    assert _admitted(tmp_path, V4_FILTERS) == []


def test_audit_census_count_matches_v4_policy(tmp_path: Path) -> None:
    """Frozen audit census for the v4 PDMX relaxed policy.

    Audit produced 229,462,616 active events on the full PDMX v1.0.0
    dataset using the v4 policy. This regression pins the filter
    policy on a synthetic 4-row fixture so the admission gate is
    not silently re-tightened. The exact active-event total is
    audited elsewhere; here we only assert that v4 admits strictly
    more than v3 on the same input.
    """
    rows: list[dict[str, str]] = []
    for i, (nlc, dedup) in enumerate(
        [
            ("True", "True"),
            ("True", "False"),
            ("False", "True"),
            ("True", "True"),
        ]
    ):
        _make_midi(tmp_path / f"row_{i}.mid")
        rows.append(_row(f"row_{i}.mid", no_conflict=nlc, dedup=dedup))
    _write_pdmx(tmp_path, rows)
    v3 = set(_admitted(tmp_path, V3_FILTERS))
    v4 = set(_admitted(tmp_path, V4_FILTERS))
    # v3 admits row_0 and row_3 (nlc=true, dedup=true, midi present).
    assert v3 == {"row_0.mid", "row_3.mid"}
    # v4 also admits row_1: nlc=true, dedup=false, midi present.
    assert v4 == {"row_0.mid", "row_1.mid", "row_3.mid"}
    # v4 strictly relaxes v3.
    assert v3 < v4
