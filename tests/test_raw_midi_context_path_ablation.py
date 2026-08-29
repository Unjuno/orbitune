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


def test_raw_midi_memory_vs_local_context_path_ablation(tmp_path: Path) -> None:
    memory = _load(MEMORY_SCRIPT, "orbitune_context_ablation_memory")
    composer = _load(COMPOSER_SCRIPT, "orbitune_context_ablation_composer")

    class MemoryAblatedComposer(composer.BoundedFactorizedTransformerComposer):
        def condition_memory(  # type: ignore[no-untyped-def]
            self, hidden: torch.Tensor, memory_tokens: torch.Tensor
        ) -> torch.Tensor:
            del memory_tokens
            # Keep the same post-memory feed-forward path and retain all
            # memory-attention parameters in the module so every arm has the
            # same allocated parameter count. Only contextual memory is ablated.
            return hidden + self.post_ff(self.post_norm(hidden))

    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"context-{fixture_seed}.mid").write_bytes(_midi(fixture_seed))

    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    prepare_compound_split_corpus(
        source,
        train_jsonl,
        validation_jsonl,
        tmp_path / "split.json",
        validation_fraction=0.25,
        split_seed="raw-midi-factorized-composer-v1",
        min_events=32,
    )
    train, validation = memory.load_splits(train_jsonl, validation_jsonl)

    configs = {
        "current_only": {"memory": False, "attention_window": 1},
        "memory_only": {"memory": True, "attention_window": 1},
        "local_only": {"memory": False, "attention_window": 16},
        "local_plus_memory": {"memory": True, "attention_window": 16},
    }
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in configs}
    parameter_counts: dict[str, int] = {}

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

        for name, config in configs.items():
            torch.manual_seed(seed + 9000)
            random.seed(seed + 9000)
            model_type = (
                composer.BoundedFactorizedTransformerComposer
                if config["memory"]
                else MemoryAblatedComposer
            )
            model = model_type(local_window=16)
            # Changing the effective attention/history window after allocation
            # preserves the exact parameter count and isolates context access.
            model.local_window = int(config["attention_window"])
            parameter_counts[name] = composer.parameter_count(model)
            composer.train_factorized_composer(
                consolidated,
                model,
                train,
                epochs=12,
                chunk_size=32,
                seed=seed,
                device=torch.device("cpu"),
                learning_rate=1e-3,
            )
            metric = composer.evaluate_factorized_composer(
                consolidated,
                model,
                validation,
                chunk_size=32,
                device=torch.device("cpu"),
            )
            rows[name].append(asdict(metric))

    assert set(parameter_counts.values()) == {280_088}
    metric_names = (
        "active_field_nll",
        "active_field_accuracy",
        "exact_event_accuracy",
        "event_type_accuracy",
        "note_pitch_accuracy",
        "note_velocity_accuracy",
        "note_duration_pair_accuracy",
        "delta_pair_accuracy",
    )

    def summarize(items: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: statistics.mean(float(row[key]) for row in items)
            for key in metric_names
        }

    warnings.warn(
        "RAW_MIDI_CONTEXT_PATH_ABLATION="
        + json.dumps(
            {
                "files": 8,
                "seeds": [1, 2, 3],
                "memory_epochs": 2,
                "composer_epochs": 12,
                "composer_learning_rate": 1e-3,
                "allocated_params_each": 280_088,
                "configs": configs,
                "summary": {name: summarize(rows[name]) for name in configs},
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
