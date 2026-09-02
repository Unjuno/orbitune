from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import torch

from orbitune.compound import CompoundEvent, CompoundEventType
from orbitune.compound_base import write_compound_midi
from orbitune.compound_indexed import build_indexed_compound_dataset, load_indexed_compound_corpus
from orbitune.indexed_sampling import IndexedSequentialSongChunkSampler, IndexedTensorSampler
from orbitune.pretrain_corpus import (
    CorpusEntry,
    _mutopia_license_from_text,
    commercial_safe_sources,
    deduplicate_entries,
    iter_pdmx_midi,
    load_registry,
    midi_fingerprints,
    split_for_composition,
    write_manifest,
)


def _write_midi(path: Path, *, transpose: int = 0, channel: int = 0, notes: int = 12) -> Path:
    events = [
        CompoundEvent(CompoundEventType.TEMPO, 0, 0, 120),
        CompoundEvent(CompoundEventType.PROGRAM, 0, channel, 40),
    ]
    for index in range(notes):
        events.append(
            CompoundEvent(
                CompoundEventType.NOTE,
                index * 24,
                channel,
                60 + transpose + index % 5,
                12,
                80,
            )
        )
    write_compound_midi(path, events)
    return path


def _entry(path: Path, *, source: str, tier: str, norm: str, comp: str, tracks: int, quality: float = 1.0) -> CorpusEntry:
    return CorpusEntry(
        source_id=source,
        path=str(path),
        license="cc0-1.0",
        tier=tier,
        quality_weight=quality,
        raw_sha256="raw-" + source,
        normalized_fingerprint=norm,
        composition_fingerprint=comp,
        events=100,
        tracks=tracks,
    )


def test_registry_is_commercial_safe_and_pins_git_sources() -> None:
    payload = load_registry()
    sources = commercial_safe_sources(payload)
    ids = {source.id for source in sources}
    assert ids == {
        "pdmx",
        "openscore_lieder",
        "openscore_string_quartets",
        "openscore_orchestra",
        "mutopia",
        "imslp_midi_cc0",
    }
    for source in sources:
        if source.kind == "git_scores":
            assert len(str(source.raw.get("ref", ""))) == 40


def test_pdmx_requires_no_license_conflict_and_upstream_dedup(tmp_path: Path) -> None:
    good = tmp_path / "mid" / "good.mid"
    conflict = tmp_path / "mid" / "conflict.mid"
    duplicate = tmp_path / "mid" / "duplicate.mid"
    good.parent.mkdir()
    for path in (good, conflict, duplicate):
        path.write_bytes(b"placeholder")
    with (tmp_path / "PDMX.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mid", "subset:no_license_conflict", "subset:deduplicated", "rating", "license", "n_tracks"],
        )
        writer.writeheader()
        writer.writerow({"mid": "mid/good.mid", "subset:no_license_conflict": "True", "subset:deduplicated": "True", "rating": "4.9", "license": "Public Domain", "n_tracks": "4"})
        writer.writerow({"mid": "mid/conflict.mid", "subset:no_license_conflict": "False", "subset:deduplicated": "True", "rating": "5", "license": "Public Domain", "n_tracks": "2"})
        writer.writerow({"mid": "mid/duplicate.mid", "subset:no_license_conflict": "True", "subset:deduplicated": "False", "rating": "5", "license": "Public Domain", "n_tracks": "2"})
    rows = list(iter_pdmx_midi(tmp_path))
    assert [path.name for path, _ in rows] == ["good.mid"]
    assert rows[0][1]["rating"] == 4.9
    assert rows[0][1]["n_tracks"] == 4


def test_mutopia_allowlist_rejects_sharealike_and_noncommercial() -> None:
    assert _mutopia_license_from_text("This work is Public Domain") == "public-domain"
    assert _mutopia_license_from_text('license = "public-domain"') == "public-domain"
    assert _mutopia_license_from_text("Creative Commons Zero 1.0") == "cc0-1.0"
    assert _mutopia_license_from_text('license = "cc0-1.0"') == "cc0-1.0"
    assert _mutopia_license_from_text("Creative Commons Attribution 4.0") == "cc-by-4.0"
    assert _mutopia_license_from_text('license = "cc-by-4.0"') == "cc-by-4.0"
    assert _mutopia_license_from_text("Creative Commons Attribution 3.0") == "cc-by-3.0"
    assert _mutopia_license_from_text('license = "cc-by-3.0"') == "cc-by-3.0"
    assert _mutopia_license_from_text("Creative Commons Attribution-ShareAlike 4.0") is None
    assert _mutopia_license_from_text("Creative Commons Attribution-NonCommercial 4.0") is None


def test_composition_fingerprint_groups_transpositions(tmp_path: Path) -> None:
    a = _write_midi(tmp_path / "a.mid", transpose=0)
    b = _write_midi(tmp_path / "b.mid", transpose=5)
    raw_a, normalized_a, composition_a, events_a, tracks_a = midi_fingerprints(a)
    raw_b, normalized_b, composition_b, events_b, tracks_b = midi_fingerprints(b)
    assert raw_a != raw_b
    assert normalized_a != normalized_b
    assert composition_a == composition_b
    assert events_a == events_b
    assert tracks_a == tracks_b == 1


def test_cross_source_dedup_prefers_quality_anchor(tmp_path: Path) -> None:
    ordinary = _entry(tmp_path / "pdmx.mid", source="pdmx", tier="primary", norm="same", comp="c", tracks=1)
    verified = _entry(tmp_path / "openscore.mid", source="openscore_lieder", tier="quality-anchor", norm="same", comp="c", tracks=1, quality=2.0)
    result = deduplicate_entries([ordinary, verified])
    assert len(result) == 1
    assert result[0].source_id == "openscore_lieder"


def test_split_is_composition_stable() -> None:
    kwargs = {"seed": "test", "validation_fraction": 0.1, "test_fraction": 0.1}
    assert split_for_composition("abc", **kwargs) == split_for_composition("abc", **kwargs)


def test_manifest_balances_track_buckets_without_split_leakage(tmp_path: Path) -> None:
    entries = [
        _entry(tmp_path / "a.mid", source="pdmx", tier="primary", norm="a", comp="c1", tracks=1),
        _entry(tmp_path / "b.mid", source="pdmx", tier="primary", norm="b", comp="c2", tracks=1),
        _entry(tmp_path / "c.mid", source="pdmx", tier="primary", norm="c", comp="c3", tracks=3),
        _entry(tmp_path / "d.mid", source="pdmx", tier="primary", norm="d", comp="c4", tracks=8),
    ]
    manifest = tmp_path / "manifest.jsonl"
    report = write_manifest(
        entries,
        manifest,
        split_config={"seed": "test", "validation_fraction": 0.2, "test_fraction": 0.2},
        track_bucket_targets={"solo": 0.4, "small_ensemble_2_5": 0.5, "large_ensemble_6_plus": 0.1},
    )
    factors = report["track_bucket_factors"]
    assert factors["solo"] == 0.8
    assert factors["small_ensemble_2_5"] == 2.0
    assert factors["large_ensemble_6_plus"] == 0.4
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    by_composition: dict[str, set[str]] = {}
    for row in rows:
        by_composition.setdefault(row["composition_fingerprint"], set()).add(row["split"])
        assert row["sampling_weight"] > 0
    assert all(len(splits) == 1 for splits in by_composition.values())


def test_indexed_corpus_build_load_and_fixed_sampler(tmp_path: Path) -> None:
    midi_a = _write_midi(tmp_path / "a.mid", notes=20)
    midi_b = _write_midi(tmp_path / "b.mid", transpose=3, notes=20)
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"path": str(midi_a), "split": "train", "raw_sha256": "a", "composition_fingerprint": "ca", "source_id": "pdmx", "license": "public-domain", "quality_weight": 1.0, "sampling_weight": 0.0, "tracks": 1, "track_bucket": "solo"},
        {"path": str(midi_b), "split": "train", "raw_sha256": "b", "composition_fingerprint": "cb", "source_id": "openscore_lieder", "license": "cc0-1.0", "quality_weight": 2.0, "sampling_weight": 1.0, "tracks": 3, "track_bucket": "small_ensemble_2_5"},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metadata = build_indexed_compound_dataset(manifest, tmp_path / "indexed", split="train")
    assert metadata["songs"] == 2
    assert len(str(metadata["manifest_sha256"])) == 64
    corpus = load_indexed_compound_corpus(tmp_path / "indexed" / "index.json")
    assert corpus.metadata["manifest_sha256"] == metadata["manifest_sha256"]
    assert len(corpus.songs) == 2
    assert len(corpus.songs[0].records) > 8
    sampler = IndexedTensorSampler(corpus.songs, weighted=True)
    x, y = sampler.sample(4, 8, random.Random(7), torch.device("cpu"))
    assert x.shape == y.shape == (4, 8, 12)
    second = torch.tensor(corpus.songs[1].records[:], dtype=torch.long)
    for sample in x:
        assert any(torch.equal(sample, second[start : start + 8]) for start in range(len(second) - 8))

    original_digest = str(metadata["manifest_sha256"])
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    changed = build_indexed_compound_dataset(manifest, tmp_path / "indexed_changed", split="train")
    assert changed["manifest_sha256"] != original_digest


def test_indexed_tbptt_sampler_state_restores_lane_positions(tmp_path: Path) -> None:
    midi = _write_midi(tmp_path / "song.mid", notes=30)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"path": str(midi), "split": "train", "raw_sha256": "a", "composition_fingerprint": "ca", "source_id": "pdmx", "license": "public-domain", "quality_weight": 1.0, "sampling_weight": 1.0, "tracks": 1, "track_bucket": "solo"}) + "\n",
        encoding="utf-8",
    )
    build_indexed_compound_dataset(manifest, tmp_path / "indexed", split="train")
    corpus = load_indexed_compound_corpus(tmp_path / "indexed" / "index.json")
    rng_a = random.Random(11)
    sampler_a = IndexedSequentialSongChunkSampler(corpus.songs, batch_size=1, seq_len=4, rng=rng_a, weighted=True)
    sampler_a.sample("cpu")
    saved = sampler_a.state_dict()
    rng_state = rng_a.getstate()
    expected = sampler_a.sample("cpu")

    rng_b = random.Random()
    rng_b.setstate(rng_state)
    sampler_b = IndexedSequentialSongChunkSampler(corpus.songs, batch_size=1, seq_len=4, rng=rng_b, weighted=True)
    sampler_b.load_state_dict(saved)
    resumed = sampler_b.sample("cpu")
    assert expected.song_indices == resumed.song_indices
    assert expected.offsets == resumed.offsets
    assert torch.equal(expected.inputs, resumed.inputs)
    assert torch.equal(expected.targets, resumed.targets)
