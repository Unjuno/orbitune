from __future__ import annotations

import shutil

from orbitune.dataset import prepare_corpus, prepare_split_corpus
from orbitune.demo import make_demo_events
from orbitune.midi import write_midi


def test_prepare_corpus(tmp_path):
    source = tmp_path / "midi"
    source.mkdir()
    write_midi(make_demo_events(bars=2, bpm=84), source / "a.mid", bpm=84)
    write_midi(make_demo_events(bars=1, bpm=72), source / "b.mid", bpm=72)

    tokens = tmp_path / "corpus.tokens"
    report_path = tmp_path / "report.json"
    report = prepare_corpus(source, tokens, report_path)
    text = tokens.read_text(encoding="utf-8")

    assert report["files_seen"] == 2
    assert report["files_accepted"] == 2
    assert report["files_rejected"] == 0
    assert report["unique_content_files"] == 2
    assert report["duplicate_files"] == 0
    assert report["song_boundaries"] == 1
    assert report["total_tokens"] > 0
    assert tokens.exists()
    assert report_path.exists()
    assert "BAR" in text
    assert "EOS\nBOS" in text


def test_prepare_split_corpus_splits_by_content_without_overlap(tmp_path):
    source = tmp_path / "midi"
    source.mkdir()
    for index in range(5):
        write_midi(make_demo_events(bars=1 + index % 2, bpm=80 + index), source / f"song-{index}.mid", bpm=80 + index)

    train_path = tmp_path / "train.tokens"
    validation_path = tmp_path / "validation.tokens"
    report_path = tmp_path / "split-report.json"
    report = prepare_split_corpus(
        source,
        train_path,
        validation_path,
        report_path,
        validation_fraction=0.2,
        split_seed="test-seed",
    )

    train_files = {item["path"] for item in report["train"]}
    validation_files = {item["path"] for item in report["validation"]}
    train_hashes = {item["sha256"] for item in report["train"]}
    validation_hashes = {item["sha256"] for item in report["validation"]}
    assert report["train_files"] == 4
    assert report["validation_files"] == 1
    assert report["split_unit"] == "MIDI content SHA-256 group"
    assert train_files.isdisjoint(validation_files)
    assert train_hashes.isdisjoint(validation_hashes)
    assert train_path.exists() and validation_path.exists()
    assert "BAR" in train_path.read_text(encoding="utf-8")
    assert "BAR" in validation_path.read_text(encoding="utf-8")


def test_exact_duplicate_midi_files_stay_in_same_split(tmp_path):
    source = tmp_path / "midi"
    source.mkdir()
    original = source / "original.mid"
    duplicate = source / "duplicate.mid"
    other = source / "other.mid"
    write_midi(make_demo_events(bars=2, bpm=84), original, bpm=84)
    shutil.copyfile(original, duplicate)
    write_midi(make_demo_events(bars=2, bpm=96), other, bpm=96)

    report = prepare_split_corpus(
        source,
        tmp_path / "train.tokens",
        tmp_path / "validation.tokens",
        tmp_path / "report.json",
        validation_fraction=0.5,
        split_seed="duplicate-test",
    )

    assert report["files_accepted"] == 3
    assert report["unique_content_files"] == 2
    assert report["duplicate_files"] == 1
    train_hashes = {item["sha256"] for item in report["train"]}
    validation_hashes = {item["sha256"] for item in report["validation"]}
    assert train_hashes.isdisjoint(validation_hashes)
    duplicate_hash = next(item["sha256"] for item in report["train"] + report["validation"] if item["path"].endswith("original.mid"))
    locations = [
        name
        for name, records in (("train", report["train"]), ("validation", report["validation"]))
        if sum(item["sha256"] == duplicate_hash for item in records) == 2
    ]
    assert len(locations) == 1
