from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from orbitune.compound import CompoundEvent, CompoundEventType
from orbitune.compound_base import write_compound_midi
from orbitune.compound_indexed import build_indexed_compound_dataset
from orbitune.indexed_sampling import IndexedSequentialSongChunkSampler
from orbitune.pretrain_corpus import load_registry
from scripts.build_pretrain_corpus import _mutopia_primary_scores
from scripts.install_pretrain_corpora import install_hf_midi


def _fake_song(*, sha: str, weight: float = 1.0, records: int = 12):
    return SimpleNamespace(
        sha256=sha,
        composition_fingerprint="composition-" + sha,
        records=[[0] * 12 for _ in range(records)],
        quality_weight=1.0,
        sampling_weight=weight,
        tracks=1,
        source_id="test",
        license="cc0-1.0",
    )


def _write_tiny_midi(path: Path) -> Path:
    events = [CompoundEvent(CompoundEventType.TEMPO, 0, 0, 120)]
    for index in range(12):
        events.append(CompoundEvent(CompoundEventType.NOTE, index * 24, 0, 60 + index % 5, 12, 80))
    write_compound_midi(path, events)
    return path


def test_huggingface_source_is_pinned_to_full_revision() -> None:
    registry = load_registry()
    source = next(item for item in registry["sources"] if item["id"] == "imslp_midi_cc0")
    assert source["revision"] == "6ae7ad248c5a599aef5f095b0694598b266eb13f"


def test_openscore_registry_selects_only_canonical_score_trees() -> None:
    registry = load_registry()
    sources = {item["id"]: item for item in registry["sources"]}
    assert all(pattern.startswith("scores/") for pattern in sources["openscore_lieder"]["score_globs"])
    assert all(pattern.startswith("scores/") for pattern in sources["openscore_string_quartets"]["score_globs"])
    orchestra = sources["openscore_orchestra"]
    assert orchestra["score_globs"] == ["data/**/*.mscz"]
    assert orchestra["exclude_annotations"] is True


def test_huggingface_installer_ingests_all_upstream_splits(tmp_path, monkeypatch) -> None:
    expected_revision = "6ae7ad248c5a599aef5f095b0694598b266eb13f"

    def fake_load_dataset(repo_id: str, *, revision: str):
        assert repo_id == "TiMauzi/imslp-midi-cc0-1.0"
        assert revision == expected_revision
        return {
            "train": [{"midi": b"train", "license": "CC0 1.0"}],
            "validation": [{"midi": b"validation", "license": "Public Domain"}],
            "test": [{"midi": b"test", "license": "CC0 1.0"}],
        }

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    target = tmp_path / "imslp"
    report = install_hf_midi(
        {
            "repo_id": "TiMauzi/imslp-midi-cc0-1.0",
            "revision": expected_revision,
            "midi_column": "midi",
        },
        target,
    )
    assert report["rows"] == 3
    assert report["splits"] == {"test": 1, "train": 1, "validation": 1}
    assert (target / "train-000000.mid").read_bytes() == b"train"
    assert (target / "validation-000000.mid").read_bytes() == b"validation"
    assert (target / "test-000000.mid").read_bytes() == b"test"


def test_mutopia_license_selection_does_not_borrow_sibling_terms(tmp_path) -> None:
    work = tmp_path / "ftp" / "Composer" / "Work"
    work.mkdir(parents=True)
    (work / "unsafe.ly").write_text(
        'mutopiatitle = "Unsafe"\nlicense = "Creative Commons Attribution-ShareAlike 4.0"\n\\midi { }\n',
        encoding="utf-8",
    )
    (work / "sibling.ly").write_text('license = "Public Domain"\n', encoding="utf-8")
    selected = _mutopia_primary_scores(tmp_path)
    assert selected == []


def test_mutopia_license_selection_accepts_local_public_domain(tmp_path) -> None:
    work = tmp_path / "ftp" / "Composer" / "Work"
    work.mkdir(parents=True)
    score = work / "score.ly"
    score.write_text(
        'mutopiatitle = "Safe"\nlicense = "Public Domain"\n\\midi { }\n',
        encoding="utf-8",
    )
    assert _mutopia_primary_scores(tmp_path) == [(score, "public-domain")]


def test_indexed_tbptt_resume_rejects_different_corpus() -> None:
    sampler_a = IndexedSequentialSongChunkSampler(
        [_fake_song(sha="a")],
        batch_size=1,
        seq_len=4,
        rng=random.Random(1),
        weighted=True,
    )
    state = sampler_a.state_dict()
    sampler_b = IndexedSequentialSongChunkSampler(
        [_fake_song(sha="b")],
        batch_size=1,
        seq_len=4,
        rng=random.Random(1),
        weighted=True,
    )
    with pytest.raises(ValueError, match="corpus identity mismatch"):
        sampler_b.load_state_dict(state)


def test_indexed_tbptt_compensates_song_start_weight_for_chunk_count() -> None:
    sampler = IndexedSequentialSongChunkSampler(
        [
            _fake_song(sha="short", weight=1.0, records=9),
            _fake_song(sha="long", weight=1.0, records=17),
        ],
        batch_size=1,
        seq_len=4,
        rng=random.Random(1),
        weighted=True,
    )
    # 9 records -> 2 complete chunks, 17 records -> 4 complete chunks.
    # A start-weight ratio of 2:1 makes the expected emitted chunk mass 1:1.
    assert sampler._complete_chunks(0) == 2
    assert sampler._complete_chunks(1) == 4
    assert sampler._song_start_weight(0) == pytest.approx(0.5)
    assert sampler._song_start_weight(1) == pytest.approx(0.25)


def test_indexed_rebuild_failure_removes_old_commit_marker(tmp_path, monkeypatch) -> None:
    midi = _write_tiny_midi(tmp_path / "song.mid")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "path": str(midi),
                "split": "train",
                "raw_sha256": "song",
                "composition_fingerprint": "composition-song",
                "source_id": "test",
                "license": "cc0-1.0",
                "quality_weight": 1.0,
                "sampling_weight": 1.0,
                "tracks": 1,
                "track_bucket": "solo",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "indexed"
    build_indexed_compound_dataset(manifest, out_dir, split="train")
    assert (out_dir / "index.json").exists()

    original_replace = Path.replace

    def fail_on_song_index(self: Path, target: Path):
        if self.name == "songs.jsonl.tmp":
            raise OSError("simulated publish failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_on_song_index)
    with pytest.raises(OSError, match="simulated publish failure"):
        build_indexed_compound_dataset(manifest, out_dir, split="train")
    assert not (out_dir / "index.json").exists()
