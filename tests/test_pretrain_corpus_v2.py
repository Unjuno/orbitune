from __future__ import annotations

from pathlib import Path

from orbitune.pretrain_corpus import commercial_safe_sources, load_registry


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "configs" / "pretrain_corpus_commercial_v1.json"
V2 = ROOT / "configs" / "pretrain_corpus_commercial_v2.json"


def test_commercial_v1_remains_immutable_six_source_registry() -> None:
    registry = load_registry(V1)
    assert registry["name"] == "orbitune-commercial-safe-v1"
    assert [source.id for source in commercial_safe_sources(registry)] == [
        "pdmx",
        "openscore_lieder",
        "openscore_string_quartets",
        "openscore_orchestra",
        "mutopia",
        "imslp_midi_cc0",
    ]


def test_commercial_v2_extends_v1_without_reassigning_existing_splits() -> None:
    v1 = load_registry(V1)
    v2 = load_registry(V2)
    v1_ids = [source.id for source in commercial_safe_sources(v1)]
    v2_ids = [source.id for source in commercial_safe_sources(v2)]

    assert v2["name"] == "orbitune-commercial-safe-v2"
    assert v2_ids[:-1] == v1_ids
    assert v2_ids[-1] == "florence_price_art_songs"
    # Keep the existing composition-level split mapping stable. Adding v2 data
    # must not reshuffle the six already-built v1 sources between train/val/test.
    assert v2["split"]["seed"] == v1["split"]["seed"]


def test_florence_price_source_is_pinned_cc0_and_fail_closed_to_upstream_generation() -> None:
    registry = load_registry(V2)
    source = next(item for item in registry["sources"] if item["id"] == "florence_price_art_songs")

    assert source["commercial_safe"] is True
    assert source["license"] == "cc0-1.0"
    assert source["git_url"] == "https://github.com/TT515/Florence_Price_Art_Song_Dataset.git"
    assert source["ref"] == "fa25c98d495e4ab86217dcc341a4b2d9fb714cfa"
    assert len(source["ref"]) == 40
    # The pinned upstream generation already contains MIDI for complete songs.
    # Only Thou'rt My Loved One lacks a MIDI and is converted explicitly; the
    # incomplete-score tree is deliberately not a conversion candidate.
    assert source["score_globs"] == ["price_songs_main/Thou'rt My Loved One/*.mscz"]
    assert "17 songs" in source["provenance_note"]
