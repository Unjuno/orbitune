from __future__ import annotations

import importlib.util
import json
import random
import statistics
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import torch

from orbitune.compound_dataset import prepare_compound_split_corpus


MEMORY_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_memory_experiment_matched.py"
COMPOSER_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_factorized_composer.py"


def _load(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
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


def _midi(seed: int, notes: int = 80) -> bytes:
    # Repeated motifs create order-sensitive local continuation while program,
    # tempo, velocity and timing variation still exercise the Compound fields.
    motifs = (
        (48, 55, 60, 64, 60, 55),
        (50, 57, 60, 67, 60, 57),
        (52, 59, 62, 65, 62, 59),
        (47, 54, 59, 62, 59, 54),
    )
    track = bytearray()
    tempo = 390_000 + (seed % 4) * 65_000
    track += b"\x00\xff\x51\x03" + tempo.to_bytes(3, "big")
    track += b"\x00" + bytes([0xC0, (seed * 11) % 96])
    track += b"\x00" + bytes([0xC1, (seed * 17 + 16) % 112])
    for index in range(notes):
        channel = index % 2
        motif = motifs[(index // 10 + seed) % len(motifs)]
        pitch = motif[index % len(motif)] + (12 if channel else 0)
        if index and index % 20 == 0:
            track += b"\x00" + bytes(
                [0xC0 | channel, (seed * 13 + index * 5 + channel * 19) % 128]
            )
        if index % 17 == 0:
            track += b"\x00" + bytes(
                [0xB0 | channel, 64, 127 if (index // 17 + seed) % 2 else 0]
            )
        if index and index % 25 == 0:
            next_tempo = 320_000 + ((seed + index) % 5) * 85_000
            track += b"\x00\xff\x51\x03" + next_tempo.to_bytes(3, "big")
        velocity = 36 + ((seed * 17 + index * 13) % 84)
        gap = (6, 12, 18, 24)[(index + seed) % 4]
        duration = (12, 18, 24, 30)[(index // 2 + seed) % 4]
        track += _vlq(gap) + bytes([0x90 | channel, pitch, velocity])
        track += _vlq(duration) + bytes([0x80 | channel, pitch, 0])
    track += b"\x00\xff\x2f\x00"
    header = (
        b"MThd"
        + (6).to_bytes(4, "big")
        + (0).to_bytes(2, "big")
        + (1).to_bytes(2, "big")
        + (96).to_bytes(2, "big")
    )
    return header + b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)


def test_factorized_active_masks_exclude_unused_a4() -> None:
    composer = _load(COMPOSER_SCRIPT, "orbitune_factorized_mask_contract")
    assert 7 not in composer.PREDICTED_FIELDS
    for fields in composer.ACTIVE_FIELDS.values():
        assert 7 not in fields
        assert 0 in fields and 2 in fields and 3 in fields


def test_raw_midi_factorized_no_local_vs_bounded_transformer(tmp_path: Path) -> None:
    memory = _load(MEMORY_SCRIPT, "orbitune_factorized_memory")
    composer = _load(COMPOSER_SCRIPT, "orbitune_factorized_compare")
    source = tmp_path / "midi"
    source.mkdir()
    for seed in range(1, 9):
        (source / f"factorized-{seed}.mid").write_bytes(_midi(seed))

    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    report_json = tmp_path / "split.json"
    prepare_compound_split_corpus(
        source,
        train_jsonl,
        validation_jsonl,
        report_json,
        validation_fraction=0.25,
        split_seed="raw-midi-factorized-composer-v1",
        min_events=32,
    )
    train, validation = memory.load_splits(train_jsonl, validation_jsonl)

    baseline_params = composer.parameter_count(composer.CapacityMatchedNoLocalComposer())
    transformer_params = composer.parameter_count(
        composer.BoundedFactorizedTransformerComposer(local_window=16)
    )
    assert baseline_params == transformer_params

    names = ("no_local", "bounded_transformer")
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in names}
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        random.seed(seed)
        consolidated = memory.RoutedMultiBank()
        memory.train_memory_stage(
            consolidated,
            train,
            epochs=2,
            chunk_size=32,
            warmup_events=8,
            seed=seed,
            device=torch.device("cpu"),
            learning_rate=2e-3,
        )

        for name in names:
            torch.manual_seed(seed + (1000 if name == "no_local" else 2000))
            if name == "no_local":
                model = composer.CapacityMatchedNoLocalComposer()
            else:
                model = composer.BoundedFactorizedTransformerComposer(local_window=16)
            composer.train_factorized_composer(
                consolidated,
                model,
                train,
                epochs=3,
                chunk_size=32,
                seed=seed,
                device=torch.device("cpu"),
                learning_rate=2e-3,
            )
            metric = composer.evaluate_factorized_composer(
                consolidated,
                model,
                validation,
                chunk_size=32,
                device=torch.device("cpu"),
            )
            rows[name].append(asdict(metric))

    def means(items: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: statistics.mean(float(row[key]) for row in items)
            for key in (
                "active_field_nll",
                "active_field_accuracy",
                "exact_event_accuracy",
                "event_type_accuracy",
                "note_pitch_accuracy",
                "note_velocity_accuracy",
                "note_duration_pair_accuracy",
                "delta_pair_accuracy",
            )
        }

    summary = {name: means(items) for name, items in rows.items()}
    warnings.warn(
        "RAW_MIDI_FACTORIZED_COMPOSER="
        + json.dumps(
            {
                "files": 8,
                "seeds": [1, 2, 3],
                "memory_epochs": 2,
                "composer_epochs": 3,
                "local_window": 16,
                "trainable_params_each": baseline_params,
                "summary": summary,
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
