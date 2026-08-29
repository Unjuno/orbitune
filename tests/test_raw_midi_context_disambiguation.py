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


def _note(track: bytearray, pitch: int, *, velocity: int = 80) -> None:
    track += _vlq(12) + bytes([0x90, pitch, velocity])
    track += _vlq(12) + bytes([0x80, pitch, 0])


def _midi(seed: int, patterns: int = 20) -> bytes:
    # At every branch point the current NOTE event is exactly pitch 64 with
    # identical channel/timing/velocity/duration. The next pitch is 67 or 69
    # and can only be inferred from a marker three NOTE events earlier.
    rng = random.Random(10_000 + seed)
    branches = [rng.randrange(2) for _ in range(patterns)]
    # Keep each file roughly balanced without making branch depend on position.
    if sum(branches) < patterns // 3:
        branches[: patterns // 2] = [index % 2 for index in range(patterns // 2)]
    if sum(branches) > 2 * patterns // 3:
        branches[: patterns // 2] = [(index + 1) % 2 for index in range(patterns // 2)]

    track = bytearray()
    track += b"\x00\xff\x51\x03" + int(500_000).to_bytes(3, "big")
    track += b"\x00" + bytes([0xC0, 0])
    for branch in branches:
        marker = 40 if branch == 0 else 52
        target = 67 if branch == 0 else 69
        _note(track, marker)
        _note(track, 60)
        _note(track, 62)
        _note(track, 64)
        _note(track, target)
    track += b"\x00\xff\x2f\x00"
    header = (
        b"MThd"
        + (6).to_bytes(4, "big")
        + (0).to_bytes(2, "big")
        + (1).to_bytes(2, "big")
        + (96).to_bytes(2, "big")
    )
    return header + b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)


def _branch_pitch_accuracy(
    composer_module,  # type: ignore[no-untyped-def]
    memory_model,  # type: ignore[no-untyped-def]
    model,  # type: ignore[no-untyped-def]
    songs,  # type: ignore[no-untyped-def]
) -> tuple[float, int]:
    device = torch.device("cpu")
    memory_model.eval()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for song in songs:
            if len(song.records) < 2:
                continue
            state = None
            history = None
            final_input = len(song.records) - 1
            for start in range(0, final_input, 32):
                stop = min(final_input, start + 32)
                records = song.records[start:stop].unsqueeze(0).to(device)
                targets = song.records[start + 1 : stop + 1].unsqueeze(0).to(device)
                memory_tokens, next_state = composer_module.bounded.routed_memory_reads(
                    memory_model, records, state
                )
                logits, history = model.forward_chunk(
                    records,
                    memory_tokens,
                    history,
                    start_index=start,
                )
                current_is_branch = records[..., 0].eq(0) & records[..., 4].eq(64)
                target_is_branch = targets[..., 0].eq(0) & (
                    targets[..., 4].eq(67) | targets[..., 4].eq(69)
                )
                mask = current_is_branch & target_is_branch
                if mask.any():
                    prediction = logits[4].argmax(-1)
                    correct += int(prediction[mask].eq(targets[..., 4][mask]).sum())
                    total += int(mask.sum())
                state = composer_module.memory_base._detach_state(next_state)
    return (correct / total if total else 0.0), total


def test_raw_midi_context_disambiguation_requires_history(tmp_path: Path) -> None:
    memory = _load(MEMORY_SCRIPT, "orbitune_disambiguation_memory")
    composer = _load(COMPOSER_SCRIPT, "orbitune_disambiguation_composer")

    class MemoryAblatedComposer(composer.BoundedFactorizedTransformerComposer):
        def condition_memory(  # type: ignore[no-untyped-def]
            self, hidden: torch.Tensor, memory_tokens: torch.Tensor
        ) -> torch.Tensor:
            del memory_tokens
            return hidden + self.post_ff(self.post_norm(hidden))

    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"context-branch-{fixture_seed}.mid").write_bytes(_midi(fixture_seed))

    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    prepare_compound_split_corpus(
        source,
        train_jsonl,
        validation_jsonl,
        tmp_path / "split.json",
        validation_fraction=0.25,
        split_seed="raw-midi-context-disambiguation-v1",
        min_events=32,
    )
    train, validation = memory.load_splits(train_jsonl, validation_jsonl)

    configs = {
        "current_only": {"memory": False, "attention_window": 1},
        "memory_only": {"memory": True, "attention_window": 1},
        "local_only": {"memory": False, "attention_window": 4},
        "local_plus_memory": {"memory": True, "attention_window": 4},
    }
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in configs}
    branch_rows: dict[str, list[float]] = {name: [] for name in configs}
    branch_counts: list[int] = []

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
            torch.manual_seed(seed + 12_000)
            random.seed(seed + 12_000)
            model_type = (
                composer.BoundedFactorizedTransformerComposer
                if config["memory"]
                else MemoryAblatedComposer
            )
            model = model_type(local_window=16)
            model.local_window = int(config["attention_window"])
            assert composer.parameter_count(model) == 280_088
            composer.train_factorized_composer(
                consolidated,
                model,
                train,
                epochs=8,
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
            branch_accuracy, branch_count = _branch_pitch_accuracy(
                composer, consolidated, model, validation
            )
            rows[name].append(asdict(metric))
            branch_rows[name].append(branch_accuracy)
            if name == "current_only":
                branch_counts.append(branch_count)

    assert min(branch_counts) > 0
    metric_names = (
        "active_field_nll",
        "active_field_accuracy",
        "exact_event_accuracy",
        "event_type_accuracy",
        "note_pitch_accuracy",
        "note_duration_pair_accuracy",
        "delta_pair_accuracy",
    )

    def summarize(items: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: statistics.mean(float(row[key]) for row in items)
            for key in metric_names
        }

    warnings.warn(
        "RAW_MIDI_CONTEXT_DISAMBIGUATION="
        + json.dumps(
            {
                "files": 8,
                "seeds": [1, 2, 3],
                "memory_epochs": 2,
                "composer_epochs": 8,
                "composer_learning_rate": 1e-3,
                "allocated_params_each": 280_088,
                "branch_points_each_seed": branch_counts,
                "configs": configs,
                "branch_pitch_accuracy": {
                    name: {
                        "mean": statistics.mean(values),
                        "by_seed": values,
                    }
                    for name, values in branch_rows.items()
                },
                "full_event_metrics": {
                    name: summarize(rows[name]) for name in configs
                },
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
