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
import torch.nn as nn
import torch.nn.functional as F

from orbitune.compound_dataset import prepare_compound_split_corpus


MEMORY_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_memory_experiment_matched.py"


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


def _note(track: bytearray, pitch: int) -> None:
    track += _vlq(12) + bytes([0x90, pitch, 80])
    track += _vlq(12) + bytes([0x80, pitch, 0])


def _midi(seed: int, patterns: int = 24) -> bytes:
    rng = random.Random(30_000 + seed)
    branches = [rng.randrange(2) for _ in range(patterns)]
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


def _branch_labels(records: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = (
        records[..., 0].eq(0)
        & records[..., 4].eq(64)
        & targets[..., 0].eq(0)
        & (targets[..., 4].eq(67) | targets[..., 4].eq(69))
    )
    labels = targets[..., 4].eq(69).long()
    return mask, labels


def _fast_reads(model, records: torch.Tensor, state):  # type: ignore[no-untyped-def]
    hidden = model.embedding(records)
    return model.fast_memory.forward_chunk(hidden, state)


def _train_probe_only(memory_model, probe, songs, *, epochs: int, seed: int) -> None:  # type: ignore[no-untyped-def]
    for parameter in memory_model.parameters():
        parameter.requires_grad_(False)
    memory_model.eval()
    probe.train()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3)
    rng = random.Random(seed + 5011)
    for _ in range(epochs):
        order = list(songs)
        rng.shuffle(order)
        for song in order:
            state = None
            final_input = len(song.records) - 1
            for start in range(0, final_input, 32):
                stop = min(final_input, start + 32)
                records = song.records[start:stop].unsqueeze(0)
                targets = song.records[start + 1 : stop + 1].unsqueeze(0)
                with torch.no_grad():
                    reads, next_state = _fast_reads(memory_model, records, state)
                mask, labels = _branch_labels(records, targets)
                if mask.any():
                    loss = F.cross_entropy(probe(reads)[mask], labels[mask])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                state = memory_model.fast_memory.forward_chunk.__globals__["chunkwise_discounted_scan"] and (
                    next_state[0].detach(), next_state[1].detach()
                )


def _train_fast_memory_aux(memory_model, probe, songs, *, epochs: int, seed: int) -> None:  # type: ignore[no-untyped-def]
    for parameter in memory_model.parameters():
        parameter.requires_grad_(False)
    for parameter in memory_model.embedding.parameters():
        parameter.requires_grad_(True)
    for parameter in memory_model.fast_memory.parameters():
        parameter.requires_grad_(True)
    for parameter in probe.parameters():
        parameter.requires_grad_(True)
    memory_model.train()
    probe.train()
    optimizer = torch.optim.AdamW(
        [
            *memory_model.embedding.parameters(),
            *memory_model.fast_memory.parameters(),
            *probe.parameters(),
        ],
        lr=2e-3,
    )
    rng = random.Random(seed + 6011)
    for _ in range(epochs):
        order = list(songs)
        rng.shuffle(order)
        for song in order:
            state = None
            final_input = len(song.records) - 1
            for start in range(0, final_input, 32):
                stop = min(final_input, start + 32)
                records = song.records[start:stop].unsqueeze(0)
                targets = song.records[start + 1 : stop + 1].unsqueeze(0)
                reads, next_state = _fast_reads(memory_model, records, state)
                mask, labels = _branch_labels(records, targets)
                if mask.any():
                    loss = F.cross_entropy(probe(reads)[mask], labels[mask])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [
                            *memory_model.embedding.parameters(),
                            *memory_model.fast_memory.parameters(),
                            *probe.parameters(),
                        ],
                        1.0,
                    )
                    optimizer.step()
                state = next_state[0].detach(), next_state[1].detach()


def _evaluate(memory_model, probe, songs) -> tuple[float, int]:  # type: ignore[no-untyped-def]
    memory_model.eval()
    probe.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for song in songs:
            state = None
            final_input = len(song.records) - 1
            for start in range(0, final_input, 32):
                stop = min(final_input, start + 32)
                records = song.records[start:stop].unsqueeze(0)
                targets = song.records[start + 1 : stop + 1].unsqueeze(0)
                reads, next_state = _fast_reads(memory_model, records, state)
                mask, labels = _branch_labels(records, targets)
                if mask.any():
                    prediction = probe(reads).argmax(-1)
                    correct += int(prediction[mask].eq(labels[mask]).sum())
                    total += int(mask.sum())
                state = next_state[0].detach(), next_state[1].detach()
    return (correct / total if total else 0.0), total


def test_explicit_marker_objective_teaches_fast_recurrent_memory(tmp_path: Path) -> None:
    memory = _load(MEMORY_SCRIPT, "orbitune_marker_objective_memory")
    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"marker-objective-{fixture_seed}.mid").write_bytes(_midi(fixture_seed))

    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    prepare_compound_split_corpus(
        source,
        train_jsonl,
        validation_jsonl,
        tmp_path / "split.json",
        validation_fraction=0.25,
        split_seed="raw-midi-marker-objective-v1",
        min_events=32,
    )
    train, validation = memory.load_splits(train_jsonl, validation_jsonl)

    probe_only_rows: list[float] = []
    auxiliary_rows: list[float] = []
    generic_before_rows: list[dict[str, float]] = []
    generic_after_rows: list[dict[str, float]] = []
    counts: list[int] = []

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
        generic_before_rows.append(
            {
                "fast": before.fast_macro_recall,
                "medium": before.medium_macro_recall,
                "slow": before.slow_macro_recall,
            }
        )

        frozen = copy.deepcopy(consolidated)
        torch.manual_seed(seed + 31_000)
        probe = nn.Linear(memory.D_MODEL, 2)
        _train_probe_only(frozen, probe, train, epochs=12, seed=seed)
        accuracy, count = _evaluate(frozen, probe, validation)
        probe_only_rows.append(accuracy)
        counts.append(count)

        adapted = copy.deepcopy(consolidated)
        torch.manual_seed(seed + 41_000)
        aux_probe = nn.Linear(memory.D_MODEL, 2)
        _train_fast_memory_aux(adapted, aux_probe, train, epochs=12, seed=seed)
        aux_accuracy, _ = _evaluate(adapted, aux_probe, validation)
        auxiliary_rows.append(aux_accuracy)
        after = memory.evaluate(
            adapted,
            validation,
            chunk_size=32,
            warmup_events=8,
            device=torch.device("cpu"),
        )
        generic_after_rows.append(
            {
                "fast": after.fast_macro_recall,
                "medium": after.medium_macro_recall,
                "slow": after.slow_macro_recall,
            }
        )

    def group_mean(rows: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: statistics.mean(row[key] for row in rows)
            for key in ("fast", "medium", "slow")
        }

    warnings.warn(
        "RAW_MIDI_MEMORY_MARKER_OBJECTIVE="
        + json.dumps(
            {
                "files": 8,
                "seeds": [1, 2, 3],
                "branch_points_each_seed": counts,
                "generic_memory_epochs": 2,
                "probe_epochs": 12,
                "auxiliary_epochs": 12,
                "frozen_probe_branch_accuracy": {
                    "by_seed": probe_only_rows,
                    "mean": statistics.mean(probe_only_rows),
                },
                "fast_memory_aux_branch_accuracy": {
                    "by_seed": auxiliary_rows,
                    "mean": statistics.mean(auxiliary_rows),
                },
                "generic_memory_before": group_mean(generic_before_rows),
                "generic_memory_after_branch_aux": group_mean(generic_after_rows),
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
