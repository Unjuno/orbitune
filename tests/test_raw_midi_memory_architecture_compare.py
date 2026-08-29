from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

from orbitune.compound_dataset import prepare_compound_split_corpus


SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_memory_experiment_matched.py"


def _load_module():  # type: ignore[no-untyped-def]
    name = "orbitune_raw_midi_memory_architecture_compare"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _vlq(value: int) -> bytes:
    parts = [value & 0x7F]
    value >>= 7
    while value:
        parts.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(parts))


def _midi(seed: int, notes: int = 72) -> bytes:
    track = bytearray()
    tempo = 360_000 + (seed % 5) * 70_000
    track += b"\x00\xff\x51\x03" + tempo.to_bytes(3, "big")
    track += b"\x00" + bytes([0xC0, (seed * 7) % 96])
    track += b"\x00" + bytes([0xC1, (seed * 13 + 24) % 112])
    for index in range(notes):
        channel = index % 2
        status_on = 0x90 | channel
        status_off = 0x80 | channel
        if index and index % 18 == 0:
            program = (seed * 11 + index * 3 + channel * 17) % 128
            track += b"\x00" + bytes([0xC0 | channel, program])
        if index % 16 == 0:
            track += b"\x00" + bytes([0xB0 | channel, 64, 127 if (index // 16 + seed) % 2 else 0])
        if index and index % 24 == 0:
            # Causal tempo changes broaden medium-state targets.
            next_tempo = 300_000 + ((seed + index) % 6) * 80_000
            track += b"\x00\xff\x51\x03" + next_tempo.to_bytes(3, "big")
        pitch_span = 24 + (seed % 4) * 12
        pitch = 30 + ((seed * 5 + index * (3 + seed % 4)) % pitch_span)
        velocity = 24 + ((seed * 19 + index * 11) % 100)
        gap = 3 + ((index * 7 + seed * 5) % 36)
        duration = 6 + ((index * 5 + seed * 3) % 42)
        track += _vlq(gap) + bytes([status_on, pitch, velocity])
        track += _vlq(duration) + bytes([status_off, pitch, 0])
    track += b"\x00\xff\x2f\x00"
    header = (
        b"MThd"
        + (6).to_bytes(4, "big")
        + (0).to_bytes(2, "big")
        + (1).to_bytes(2, "big")
        + (96).to_bytes(2, "big")
    )
    return header + b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)


def test_raw_midi_parameter_matched_shared_vs_routed_three_seed(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "midi"
    source.mkdir()
    for seed in range(1, 9):
        (source / f"arch-{seed}.mid").write_bytes(_midi(seed))

    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    report_json = tmp_path / "split.json"
    report = prepare_compound_split_corpus(
        source,
        train_jsonl,
        validation_jsonl,
        report_json,
        validation_fraction=0.25,
        split_seed="raw-midi-memory-architecture-compare-v1",
        min_events=32,
    )
    assert report["files_accepted"] == 8
    assert report["files_rejected"] == 0

    rows: dict[str, list[dict[str, float]]] = {
        "shared_matched": [],
        "multibank_routed": [],
    }
    for mode in rows:
        for seed in (1, 2, 3):
            args = SimpleNamespace(
                device="cpu",
                train_jsonl=str(train_jsonl),
                validation_jsonl=str(validation_jsonl),
                max_train_songs=0,
                max_validation_songs=0,
                warmup_events=8,
                seed=seed,
                mode=mode,
                memory_epochs=2,
                chunk_size=32,
                memory_lr=2e-3,
                composer_policy="frozen",
                composer_epochs=0,
                composer_lr=1e-3,
                memory_lr_multiplier=0.1,
                checkpoint_out=None,
            )
            result = module.run(args)
            metric = result["validation_before_composer"]
            rows[mode].append(
                {
                    "fast": metric["fast_macro_recall"],
                    "medium": metric["medium_macro_recall"],
                    "slow": metric["slow_macro_recall"],
                    "event": metric["next_event_type_accuracy"],
                }
            )

    summary: dict[str, dict[str, float]] = {}
    for mode, metrics in rows.items():
        summary[mode] = {
            key: statistics.mean(row[key] for row in metrics)
            for key in ("fast", "medium", "slow", "event")
        }
    shared_params = sum(p.numel() for p in module.SharedMatched().parameters())
    routed_params = sum(p.numel() for p in module.RoutedMultiBank().parameters())
    assert shared_params == routed_params == 157650
    assert all(0.0 <= value <= 1.0 for mode in summary.values() for value in mode.values())
    warnings.warn(
        "RAW_MIDI_ARCH_COMPARE="
        + json.dumps(
            {
                "parameter_count_each": shared_params,
                "files": 8,
                "seeds": [1, 2, 3],
                "summary": summary,
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
