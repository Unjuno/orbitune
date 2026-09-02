from __future__ import annotations

import random
import sys
from types import SimpleNamespace

import pytest

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


def test_huggingface_source_is_pinned_to_full_revision() -> None:
    registry = load_registry()
    source = next(item for item in registry["sources"] if item["id"] == "imslp_midi_cc0")
    assert source["revision"] == "6ae7ad248c5a599aef5f095b0694598b266eb13f"


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
