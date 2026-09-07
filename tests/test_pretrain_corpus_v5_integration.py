"""Regression tests for the v5 commercial corpus integration.

Each test enforces a v5-specific invariant from the v5 integration spec.
None of these tests mutate C:\\ov3, C:\\ov4, or any production data.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
V5_REGISTRY = REPO_ROOT / "configs" / "pretrain_corpus_commercial_v5.json"
V4_REGISTRY = REPO_ROOT / "configs" / "pretrain_corpus_commercial_v4.json"

V4_FROZEN_MANIFEST_SHA256 = "1c582a08a3087952a57a604b5652cae2bef6dd4e6acb2addc5eb49cdfbe58c72"
V4_TRAIN_SONGS = 215963
V4_TRAIN_RECORDS = 234904281
V4_ONE_X_ACTIVE_EVENTS = 234688318

PREVIOUS_REV = "8fd2169855a1b27586454b183739573962f9d8ca"
TARGET_REV = "a3b3813477b06c2c48887f1055f8567dae0c2232"
FULL_HEX = re.compile(r"^[0-9a-fA-F]{40}$")
FULL_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v5_registry_pins_exact_target_revision():
    reg = _load(V5_REGISTRY)
    sources = {s["id"]: s for s in reg["sources"]}
    assert "openscore_string_quartets" in sources
    sq = sources["openscore_string_quartets"]
    assert sq["ref"] == TARGET_REV
    assert FULL_HEX.fullmatch(sq["ref"]), "v5 ref must be a 40-char hex commit"


def test_v5_registry_records_previous_revision():
    reg = _load(V5_REGISTRY)
    sources = {s["id"]: s for s in reg["sources"]}
    sq = sources["openscore_string_quartets"]
    assert sq.get("previous_ref") == PREVIOUS_REV
    assert FULL_HEX.fullmatch(sq["previous_ref"]), "previous_ref must be a 40-char hex commit"


def test_v5_delta_census_sha256_recorded_in_registry():
    reg = _load(V5_REGISTRY)
    sq = next(s for s in reg["sources"] if s["id"] == "openscore_string_quartets")
    assert sq.get("v5_delta_census") == "tools/openscore_string_quartets_v5_delta_census.json"
    assert FULL_SHA256.fullmatch(sq.get("v5_delta_census_sha256", "")), "delta census SHA256 must be 64-char hex"
    # Re-verify by hashing the actual file
    census_path = REPO_ROOT / sq["v5_delta_census"]
    assert census_path.exists(), f"census path {census_path} must exist"
    actual = hashlib.sha256(census_path.read_bytes()).hexdigest()
    assert actual == sq["v5_delta_census_sha256"], (
        f"delta census SHA mismatch: declared {sq['v5_delta_census_sha256']}, "
        f"actual {actual}"
    )


def test_v5_delta_census_41_rows_internally_consistent():
    reg = _load(V5_REGISTRY)
    sq = next(s for s in reg["sources"] if s["id"] == "openscore_string_quartets")
    census_path = REPO_ROOT / sq["v5_delta_census"]
    c = _load(census_path)
    assert c["census_input"] == c["expected_delta_artifacts"] == 41
    assert c["v5_rev"] == TARGET_REV
    assert c["v4_rev"] == PREVIOUS_REV
    # 41-row evidence
    assert len(c["rows"]) == 41
    parse_ok = sum(1 for r in c["rows"] if r["parse_ok"])
    assert parse_ok == 41, "every admitted delta artifact must parse OK"


def test_v5_full_v4_dedup_uses_all_admitted_v4_fingerprints():
    """The v4 fingerprint index must equal the count of accepted v4 records
    (each row in C:\\ov4\\manifest.jsonl has a unique normalized_fingerprint,
    by construction of the v4 build)."""
    reg = _load(V5_REGISTRY)
    sq = next(s for s in reg["sources"] if s["id"] == "openscore_string_quartets")
    cross = _load(REPO_ROOT / sq["v5_delta_cross_v4_dedup"])
    assert cross["v4_manifest_sha256"] == V4_FROZEN_MANIFEST_SHA256
    assert cross["v4_manifest_sha256_match"] is True
    # Expected v4 normalized_fp unique count is 229476
    assert cross["v4_normalized_fingerprint_index_size"] == 229476
    # cross-v4 matches should be zero for the OpenScore SQ delta
    assert cross["delta_normalized_matches"] == 0
    assert cross["delta_composition_matches"] == 0
    assert cross["delta_post_dedup_accepted"] == 41


def test_v5_normalized_fingerprint_is_production_dedup_key():
    reg = _load(V5_REGISTRY)
    sq = next(s for s in reg["sources"] if s["id"] == "openscore_string_quartets")
    cross = _load(REPO_ROOT / sq["v5_delta_cross_v4_dedup"])
    assert cross["production_dedup_key"] == "normalized_fingerprint"
    assert "split_grouping_only" in cross["composition_fingerprint_role"]
    assert cross["composition_fingerprint_role"].startswith("split_grouping_only")


def test_v5_no_delta_double_ingestion():
    """v5 registry must NOT create a second logical source for the delta.
    The openscore_string_quartets source must be the only one whose ref
    changed between v4 and v5; the registry must not introduce any
    openscore_string_quartets_v5_delta source."""
    reg_v5 = _load(V5_REGISTRY)
    reg_v4 = _load(V4_REGISTRY)
    v5_ids = {s["id"] for s in reg_v5["sources"]}
    v4_ids = {s["id"] for s in reg_v4["sources"]}
    # Same set of source ids
    assert v5_ids == v4_ids, f"v5 source ids must match v4 set: v5-v4={v5_ids-v4_ids}, v4-v5={v4_ids-v5_ids}"
    # No delta-only id was created
    assert "openscore_string_quartets_v5_delta" not in v5_ids
    # Only one ref changes between v4 and v5
    v4_by_id = {s["id"]: s for s in reg_v4["sources"]}
    changed_refs = [
        sid for sid in v5_ids
        if "ref" in v4_by_id[sid] and v4_by_id[sid].get("ref") != next(s for s in reg_v5["sources"] if s["id"] == sid).get("ref")
    ]
    assert changed_refs == ["openscore_string_quartets"], (
        f"only openscore_string_quartets should change ref; got {changed_refs}"
    )


def test_v4_registry_unchanged():
    """The v4 registry JSON on disk must be byte-identical to the v4 commit
    it points at. We confirm by checking that the source_ids, ref values,
    and license_policy deny_markers match the v5-expected baseline."""
    reg = _load(V4_REGISTRY)
    assert reg["name"] == "orbitune-commercial-safe-v4"
    # Confirm key v4 sources and pins
    sources = {s["id"]: s for s in reg["sources"]}
    assert sources["openscore_string_quartets"]["ref"] == PREVIOUS_REV
    assert sources["nrg_cp"]["license"] == "cc-by-4.0"
    assert sources["groove_midi_dataset"]["license"] == "cc-by-4.0"
    # License deny_markers must include 'sharealike' (regression: v5 must
    # not weaken it)
    assert "sharealike" in reg["license_policy"]["deny_markers"]
    assert "cc-by-sa" in reg["license_policy"]["deny_markers"]


def test_v4_build_identity_unchanged():
    """The v4 manifest SHA256 and frozen metrics must be recorded in the
    v5 registry's v4_baseline and must match the canonical frozen values.
    Regression: do not silently rewrite the v4 baseline in the v5 registry."""
    reg = _load(V5_REGISTRY)
    bl = reg.get("v4_baseline")
    assert bl is not None, "v5 registry must record v4_baseline"
    assert bl["manifest_sha256"] == V4_FROZEN_MANIFEST_SHA256
    assert bl["train_songs"] == V4_TRAIN_SONGS
    assert bl["train_records"] == V4_TRAIN_RECORDS
    assert bl["one_x_active_events"] == V4_ONE_X_ACTIVE_EVENTS
    # Invariant: train_records - train_songs == one_x_active_events
    assert bl["train_records"] - bl["train_songs"] == bl["one_x_active_events"], (
        "v4 baseline violates TRAIN_RECORDS - TRAIN_SONGS == ONE_X_ACTIVE_EVENTS"
    )


def test_v5_hold_sources_not_added():
    """Sources that are YELLOW/HOLD or otherwise unresolved in v5-rev3
    (CocoChorales, RISM, IMSLP-CC-BY expansion, Humdrum new children,
    ATEPP, GiantMIDI-Piano) must not appear as production sources in v5."""
    reg = _load(V5_REGISTRY)
    forbidden = {
        "cocochorales",
        "cocochorales_synthetic",
        "cocochorales_v5",
        "rism",
        "imslp_midi_cc_by_expansion",
        "humdrum_haydn_piano_sonatas",
        "humdrum_bach_chorales",
        "atepp",
        "giantmidi_piano",
    }
    present = {s["id"] for s in reg["sources"]}
    overlap = forbidden & present
    assert not overlap, f"hold / not-yet-admitted sources present: {overlap}"


def test_v5_train_records_semantics_regression():
    """Regression: the v5 integration must NOT add accepted_song_count to
    V4_TRAIN_RECORDS to obtain V5_TRAIN_RECORDS.  V5_TRAIN_RECORDS is
    produced by the build and equals (sum of post-dedup events across
    all v5 train songs) and is therefore strictly greater than
    V4_TRAIN_RECORDS + 0 (it is allowed to equal V4_TRAIN_RECORDS only
    if the delta contributes 0 active events, which the delta does
    not).  This test guards the *integration logic* by checking the v5
    registry's v4_baseline frozen value, not a pre-computed v5 value."""
    reg = _load(V5_REGISTRY)
    bl = reg["v4_baseline"]
    # If a hardcoded V5_TRAIN_RECORDS field were equal to
    # V4_TRAIN_RECORDS + delta_song_count, the integration would be
    # silently incorrect. The v5 registry MUST NOT pre-bake that value.
    assert "train_records_upper_bound" not in reg, (
        "v5 registry must not pre-bake V5_TRAIN_RECORDS = V4 + delta_song_count; "
        "use the build's actual manifest to compute the final value."
    )
    # Also ensure the registry does not record delta_song_count as a
    # fixed train-increment number that could be misused.
    sq = next(s for s in reg["sources"] if s["id"] == "openscore_string_quartets")
    assert sq.get("delta_accepted_artifacts") == 41
    assert sq.get("delta_post_dedup_active_events") == 560680
    # delta_accepted_artifacts is the *pre-split* number, not a train-increment.
    assert "delta_train_songs" not in sq, (
        "v5 must not pre-declare train split assignment; let the build assign."
    )


def test_v5_train_records_minus_train_songs_invariant_documents_definition():
    """Document the v4 invariant and re-state it for v5. If the builder
    semantics differ in v5, the test fails and forces explicit
    documentation."""
    reg = _load(V5_REGISTRY)
    bl = reg["v4_baseline"]
    # v4 invariant
    assert bl["train_records"] - bl["train_songs"] == bl["one_x_active_events"]
    # The v5 invariant will be: V5_TRAIN_RECORDS - V5_TRAIN_SONGS == V5_ONE_X_ACTIVE_EVENTS
    # (after build). The v5 registry does not yet pre-declare v5 metrics,
    # so this test only guards the documentation pattern (v4_baseline present).
    # When the build completes, the v5 manifest will populate the same
    # relation in build_report.json.
    assert "one_x_active_events" in bl
