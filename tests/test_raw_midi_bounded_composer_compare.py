from __future__ import annotations

import copy
import importlib.util
import json
import random
import statistics
import sys
import warnings
from pathlib import Path

import torch

from orbitune.compound_dataset import prepare_compound_split_corpus


MEMORY_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_memory_experiment_matched.py"
COMPOSER_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_bounded_composer.py"


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


def _midi(seed: int, notes: int = 72) -> bytes:
    track = bytearray()
    tempo = 360_000 + (seed % 5) * 70_000
    track += b"\x00\xff\x51\x03" + tempo.to_bytes(3, "big")
    track += b"\x00" + bytes([0xC0, (seed * 7) % 96])
    track += b"\x00" + bytes([0xC1, (seed * 13 + 24) % 112])
    for index in range(notes):
        channel = index % 2
        if index and index % 18 == 0:
            track += b"\x00" + bytes(
                [0xC0 | channel, (seed * 11 + index * 3 + channel * 17) % 128]
            )
        if index % 16 == 0:
            track += b"\x00" + bytes(
                [0xB0 | channel, 64, 127 if (index // 16 + seed) % 2 else 0]
            )
        if index and index % 24 == 0:
            next_tempo = 300_000 + ((seed + index) % 6) * 80_000
            track += b"\x00\xff\x51\x03" + next_tempo.to_bytes(3, "big")
        pitch_span = 24 + (seed % 4) * 12
        pitch = 30 + ((seed * 5 + index * (3 + seed % 4)) % pitch_span)
        velocity = 24 + ((seed * 19 + index * 11) % 100)
        gap = 3 + ((index * 7 + seed * 5) % 36)
        duration = 6 + ((index * 5 + seed * 3) % 42)
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


def test_bounded_local_attention_mask_is_causal_and_windowed() -> None:
    composer = _load(COMPOSER_SCRIPT, "orbitune_bounded_composer_mask")
    mask = composer._local_causal_mask(12, 4, torch.device("cpu"))
    for query in range(12):
        for key in range(12):
            expected_blocked = key > query or query - key >= 4
            assert bool(mask[query, key]) is expected_blocked


def test_raw_midi_mlp_vs_bounded_local_transformer_three_seed(tmp_path: Path) -> None:
    memory = _load(MEMORY_SCRIPT, "orbitune_bounded_composer_memory")
    composer = _load(COMPOSER_SCRIPT, "orbitune_bounded_composer_compare")
    source = tmp_path / "midi"
    source.mkdir()
    for seed in range(1, 9):
        (source / f"composer-{seed}.mid").write_bytes(_midi(seed))

    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    report_json = tmp_path / "split.json"
    prepare_compound_split_corpus(
        source,
        train_jsonl,
        validation_jsonl,
        report_json,
        validation_fraction=0.25,
        split_seed="raw-midi-bounded-composer-v1",
        min_events=32,
    )
    train, validation = memory.load_splits(train_jsonl, validation_jsonl)

    mlp_scores: list[float] = []
    transformer_scores: list[float] = []
    memory_before: list[dict[str, float]] = []
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
        before = memory.evaluate(
            consolidated,
            validation,
            chunk_size=32,
            warmup_events=8,
            device=torch.device("cpu"),
        )
        memory_before.append(
            {
                "fast": before.fast_macro_recall,
                "medium": before.medium_macro_recall,
                "slow": before.slow_macro_recall,
            }
        )

        mlp = copy.deepcopy(consolidated)
        memory.train_composer_stage(
            mlp,
            train,
            policy="frozen",
            epochs=2,
            chunk_size=32,
            seed=seed,
            device=torch.device("cpu"),
            composer_lr=2e-3,
            memory_lr_multiplier=0.1,
        )
        mlp_metric = memory.evaluate(
            mlp,
            validation,
            chunk_size=32,
            warmup_events=8,
            device=torch.device("cpu"),
        )
        mlp_scores.append(mlp_metric.next_event_type_accuracy)

        torch.manual_seed(seed + 1000)
        local = composer.BoundedLocalTransformerComposer(local_window=16)
        composer.train_bounded_composer(
            consolidated,
            local,
            train,
            epochs=2,
            chunk_size=32,
            seed=seed,
            device=torch.device("cpu"),
            learning_rate=2e-3,
        )
        transformer_scores.append(
            composer.evaluate_bounded_composer(
                consolidated,
                local,
                validation,
                chunk_size=32,
                device=torch.device("cpu"),
            )
        )

        # Training the separate composer must not mutate consolidated memory.
        after_memory = memory.evaluate(
            consolidated,
            validation,
            chunk_size=32,
            warmup_events=8,
            device=torch.device("cpu"),
        )
        assert after_memory.fast_macro_recall == before.fast_macro_recall
        assert after_memory.medium_macro_recall == before.medium_macro_recall
        assert after_memory.slow_macro_recall == before.slow_macro_recall

    result = {
        "files": 8,
        "seeds": [1, 2, 3],
        "local_window": 16,
        "memory_mean": {
            key: statistics.mean(row[key] for row in memory_before)
            for key in ("fast", "medium", "slow")
        },
        "mlp_event_accuracy_mean": statistics.mean(mlp_scores),
        "bounded_transformer_event_accuracy_mean": statistics.mean(transformer_scores),
        "mlp_event_accuracy_by_seed": mlp_scores,
        "bounded_transformer_event_accuracy_by_seed": transformer_scores,
        "bounded_transformer_trainable_params": composer.composer_parameter_count(
            composer.BoundedLocalTransformerComposer(local_window=16)
        ),
    }
    warnings.warn(
        "RAW_MIDI_BOUNDED_COMPOSER=" + json.dumps(result, sort_keys=True),
        RuntimeWarning,
        stacklevel=1,
    )
