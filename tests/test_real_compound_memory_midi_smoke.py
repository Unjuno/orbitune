from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from orbitune.compound_dataset import prepare_compound_split_corpus


SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_memory_experiment_matched.py"


def _load_module():  # type: ignore[no-untyped-def]
    name = "orbitune_real_compound_memory_midi_smoke_matched"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _vlq(value: int) -> bytes:
    if value < 0:
        raise ValueError("VLQ must be non-negative")
    parts = [value & 0x7F]
    value >>= 7
    while value:
        parts.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(parts))


def _midi_bytes(seed: int, notes: int = 40) -> bytes:
    track = bytearray()
    track += b"\x00\xff\x51\x03" + int(500_000 + seed * 1_000).to_bytes(3, "big")
    track += b"\x00" + bytes([0xC0, seed % 32])
    for index in range(notes):
        if index and index % 10 == 0:
            track += b"\x00" + bytes([0xC0, (seed + index) % 64])
        if index % 13 == 0:
            track += b"\x00" + bytes([0xB0, 64, 127 if (index // 13) % 2 == 0 else 0])
        pitch = 36 + ((seed * 7 + index * 5) % 48)
        velocity = 40 + ((seed * 11 + index * 9) % 70)
        track += _vlq(12 + (index % 4) * 6) + bytes([0x90, pitch, velocity])
        track += _vlq(18 + (index % 3) * 6) + bytes([0x80, pitch, 0])
    track += b"\x00\xff\x2f\x00"
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + (96).to_bytes(2, "big")
    chunk = b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)
    return header + chunk


def test_raw_midi_to_real_memory_harness_smoke(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "midi"
    source.mkdir()
    for seed in range(1, 5):
        (source / f"fixture-{seed}.mid").write_bytes(_midi_bytes(seed))

    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    report_json = tmp_path / "split-report.json"
    report = prepare_compound_split_corpus(
        source,
        train_jsonl,
        validation_jsonl,
        report_json,
        validation_fraction=0.25,
        split_seed="real-memory-ci-smoke",
        min_events=8,
    )
    assert report["files_accepted"] == 4
    assert report["files_rejected"] == 0

    train, validation = module.load_splits(train_jsonl, validation_jsonl)
    assert train and validation
    assert {song.sha256 for song in train}.isdisjoint(
        {song.sha256 for song in validation}
    )
    profile = module.target_profile(train, warmup_events=4)
    assert profile["fast"]["events"] > 0
    assert profile["medium"]["events"] > 0
    assert profile["slow"]["events"] > 0

    args = SimpleNamespace(
        device="cpu",
        train_jsonl=str(train_jsonl),
        validation_jsonl=str(validation_jsonl),
        max_train_songs=2,
        max_validation_songs=1,
        warmup_events=4,
        seed=1,
        mode="multibank_routed",
        memory_epochs=1,
        chunk_size=16,
        memory_lr=1e-3,
        composer_policy="frozen",
        composer_epochs=1,
        composer_lr=1e-3,
        memory_lr_multiplier=0.1,
        checkpoint_out=None,
    )
    result = module.run(args)
    assert result["split"]["train"]["songs"] >= 1
    assert result["split"]["validation"]["songs"] == 1
    assert result["training"]["state_carry"].startswith("composition-local")
    assert result["validation_after_composer"]["events_evaluated"] > 0
    assert result["memory_delta"]["fast_macro_recall"] == 0.0
    assert result["memory_delta"]["medium_macro_recall"] == 0.0
    assert result["memory_delta"]["slow_macro_recall"] == 0.0
    print(
        "REAL_MIDI_SMOKE_RESULT="
        + json.dumps(
            {
                "split": result["split"],
                "before": result["validation_before_composer"],
                "after": result["validation_after_composer"],
                "memory_delta": result["memory_delta"],
            },
            sort_keys=True,
        )
    )
