from __future__ import annotations

import gzip
import importlib.util
import json
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "tools" / "rism_exact.py"
SPEC = importlib.util.spec_from_file_location("rism_exact", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _write_rism_fixture(path: Path) -> Path:
    xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<collection xmlns=\"http://www.loc.gov/MARC21/slim\">
  <record>
    <controlfield tag=\"001\">safe-1</controlfield>
    <datafield tag=\"100\"><subfield code=\"a\">Safe Composer</subfield><subfield code=\"d\">1685-1750</subfield></datafield>
    <datafield tag=\"031\"><subfield code=\"g\">G-2</subfield><subfield code=\"o\">4/4</subfield><subfield code=\"p\">4C D E /</subfield></datafield>
  </record>
  <record>
    <controlfield tag=\"001\">dup-1</controlfield>
    <datafield tag=\"100\"><subfield code=\"d\">1700-1770</subfield></datafield>
    <datafield tag=\"031\"><subfield code=\"g\">G-2</subfield><subfield code=\"o\">4/4</subfield><subfield code=\"p\">4C D E /</subfield></datafield>
  </record>
  <record>
    <controlfield tag=\"001\">missing-clef</controlfield>
    <datafield tag=\"100\"><subfield code=\"d\">1700-1770</subfield></datafield>
    <datafield tag=\"031\"><subfield code=\"p\">4F G A /</subfield></datafield>
  </record>
  <record>
    <controlfield tag=\"001\">unsafe-date</controlfield>
    <datafield tag=\"100\"><subfield code=\"d\">1900-</subfield></datafield>
    <datafield tag=\"031\"><subfield code=\"g\">G-2</subfield><subfield code=\"p\">4A B C /</subfield></datafield>
  </record>
</collection>
"""
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(xml)
    return path


def test_iter_admitted_unique_preserves_conservative_policy(tmp_path: Path) -> None:
    archive = _write_rism_fixture(tmp_path / "rism.xml.gz")
    counters: Counter[str] = Counter()
    rows = list(MOD.iter_admitted_unique(archive, pd_death_cutoff=1955, counters=counters))
    assert [row["record_id"] for row in rows] == ["safe-1"]
    assert counters["source_records"] == 4
    assert counters["musical_incipits"] == 4
    assert counters["pd_safe_incipits_pre_dedup"] == 3
    assert counters["pae_duplicates"] == 1
    assert counters["rejected_missing_clef"] == 1
    assert counters["rejected_person_date_policy"] == 1
    assert counters["pae_unique"] == 1


def test_load_baseline_normalized_is_fail_closed(tmp_path: Path) -> None:
    a = "a" * 64
    b = "b" * 64
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"normalized_fingerprint": a}) + "\n" + json.dumps({"normalized_fingerprint": b}) + "\n", encoding="utf-8")
    values, report = MOD.load_baseline_normalized(manifest)
    assert values == {bytes.fromhex(a), bytes.fromhex(b)}
    assert report["rows"] == 2
    assert len(report["sha256"]) == 64

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"normalized_fingerprint": "not-a-sha"}) + "\n", encoding="utf-8")
    try:
        MOD.load_baseline_normalized(bad)
    except ValueError as exc:
        assert "not suitable" in str(exc)
    else:
        raise AssertionError("invalid baseline fingerprint must fail closed")


def test_classify_conversion_result_uses_normalized_not_composition_for_dedup() -> None:
    baseline = {bytes.fromhex("b" * 64)}
    seen: set[bytes] = set()
    counters: Counter[str] = Counter()

    retained = {
        "ok": True,
        "normalized_fingerprint": "a" * 64,
        "composition_fingerprint": "c" * 64,
        "active_events": 10,
        "compound_records": 11,
        "midi_events": 12,
        "verovio_log_lines": 0,
    }
    assert MOD.classify_conversion_result(retained, baseline_normalized=baseline, seen_rism_normalized=seen, counters=counters)

    same_composition_different_normalized = {
        **retained,
        "normalized_fingerprint": "d" * 64,
        "active_events": 7,
    }
    assert MOD.classify_conversion_result(same_composition_different_normalized, baseline_normalized=baseline, seen_rism_normalized=seen, counters=counters)

    intra = {**retained, "active_events": 99}
    assert not MOD.classify_conversion_result(intra, baseline_normalized=baseline, seen_rism_normalized=seen, counters=counters)

    cross = {**retained, "normalized_fingerprint": "b" * 64}
    assert not MOD.classify_conversion_result(cross, baseline_normalized=baseline, seen_rism_normalized=seen, counters=counters)

    failed = {"ok": False, "verovio_log_lines": 2, "verovio_log": "warning"}
    assert not MOD.classify_conversion_result(failed, baseline_normalized=baseline, seen_rism_normalized=seen, counters=counters)

    assert counters["conversion_success"] == 4
    assert counters["conversion_failure"] == 1
    assert counters["normalized_unique"] == 3
    assert counters["intra_source_duplicates"] == 1
    assert counters["cross_v4_duplicates"] == 1
    assert counters["retained_after_cross_dedup"] == 2
    assert counters["exact_active_events_post_dedup"] == 17
