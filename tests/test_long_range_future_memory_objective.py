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
BOUNDED_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_bounded_composer.py"


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


def _midi(seed: int, patterns: int = 12) -> bytes:
    # Each pattern has exactly 32 NOTE events so the branch point has the same
    # local-position phase. The marker is 30 NOTE events before the ambiguous
    # current event and is therefore invisible to a 16-event local window.
    branches = [0, 1] * (patterns // 2)
    random.Random(20_000 + seed).shuffle(branches)
    track = bytearray()
    track += b"\x00\xff\x51\x03" + int(500_000).to_bytes(3, "big")
    track += b"\x00" + bytes([0xC0, 0])
    fillers = (60, 62, 65)
    for branch in branches:
        _note(track, 40 if branch == 0 else 52)
        for index in range(29):
            _note(track, fillers[index % len(fillers)])
        _note(track, 64)
        _note(track, 67 if branch == 0 else 69)
    track += b"\x00\xff\x2f\x00"
    header = (
        b"MThd"
        + (6).to_bytes(4, "big")
        + (0).to_bytes(2, "big")
        + (1).to_bytes(2, "big")
        + (96).to_bytes(2, "big")
    )
    return header + b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)


def _branch_examples(song):  # type: ignore[no-untyped-def]
    records = song.records
    examples: list[tuple[int, int]] = []
    for index in range(len(records) - 1):
        if int(records[index, 0]) != 0 or int(records[index, 4]) != 64:
            continue
        target_pitch = int(records[index + 1, 4])
        if int(records[index + 1, 0]) == 0 and target_pitch in (67, 69):
            examples.append((index, 0 if target_pitch == 67 else 1))
    return examples


def _assert_local_context_is_ambiguous(songs, window: int = 16) -> None:  # type: ignore[no-untyped-def]
    fingerprints: dict[int, set[tuple[tuple[int, ...], ...]]] = {0: set(), 1: set()}
    for song in songs:
        for index, label in _branch_examples(song):
            assert index + 1 >= 30
            start = index - window + 1
            context = tuple(tuple(int(v) for v in row) for row in song.records[start : index + 1])
            fingerprints[label].add(context)
    # Exact same bounded local contexts occur for both continuation classes.
    assert fingerprints[0]
    assert fingerprints[1]
    assert fingerprints[0] == fingerprints[1]


def _branch_features(bounded, memory_model, songs):  # type: ignore[no-untyped-def]
    features: list[torch.Tensor] = []
    labels: list[int] = []
    memory_model.eval()
    with torch.no_grad():
        for song in songs:
            records = song.records.unsqueeze(0)
            tokens, _ = bounded.routed_memory_reads(memory_model, records, None)
            for index, label in _branch_examples(song):
                features.append(tokens[0, index].reshape(-1).cpu())
                labels.append(label)
    return torch.stack(features), torch.tensor(labels, dtype=torch.long)


class Probe(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _fit_fresh_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    seed: int,
) -> float:
    torch.manual_seed(seed + 30_000)
    probe = Probe(train_features.shape[-1])
    optimizer = torch.optim.AdamW(probe.parameters(), lr=3e-3)
    for _ in range(250):
        logits = probe(train_features)
        loss = F.cross_entropy(logits, train_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        prediction = probe(validation_features).argmax(-1)
    return float(prediction.eq(validation_labels).float().mean())


def _future_consolidate(
    bounded,
    memory_model,
    songs,
    *,
    seed: int,
    epochs: int = 20,
) -> None:  # type: ignore[no-untyped-def]
    torch.manual_seed(seed + 40_000)
    head = Probe(3 * 48)
    parameters = [*memory_model.parameters(), *head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    rng = random.Random(seed + 40_000)
    memory_model.train()
    head.train()
    for _ in range(epochs):
        order = list(songs)
        rng.shuffle(order)
        for song in order:
            records = song.records.unsqueeze(0)
            tokens, _ = bounded.routed_memory_reads(memory_model, records, None)
            indices_and_labels = _branch_examples(song)
            indices = torch.tensor([item[0] for item in indices_and_labels], dtype=torch.long)
            labels = torch.tensor([item[1] for item in indices_and_labels], dtype=torch.long)
            branch_features = tokens[0, indices].reshape(len(indices_and_labels), -1)
            loss = F.cross_entropy(head(branch_features), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()


def test_future_prediction_objective_makes_long_range_memory_decodable(tmp_path: Path) -> None:
    memory = _load(MEMORY_SCRIPT, "orbitune_long_range_memory")
    bounded = _load(BOUNDED_SCRIPT, "orbitune_long_range_bounded")
    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"long-range-{fixture_seed}.mid").write_bytes(_midi(fixture_seed))

    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    prepare_compound_split_corpus(
        source,
        train_jsonl,
        validation_jsonl,
        tmp_path / "split.json",
        validation_fraction=0.25,
        split_seed="long-range-future-memory-v1",
        min_events=64,
    )
    train, validation = memory.load_splits(train_jsonl, validation_jsonl)
    _assert_local_context_is_ambiguous([*train, *validation], window=16)
    assert sum(len(_branch_examples(song)) for song in validation) == 24

    rows = {"aggregate_only": [], "future_consolidated": []}
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        random.seed(seed)
        aggregate = memory.RoutedMultiBank()
        memory.train_memory_stage(
            aggregate,
            train,
            epochs=2,
            chunk_size=1024,
            warmup_events=8,
            seed=seed,
            device=torch.device("cpu"),
            learning_rate=2e-3,
        )

        future = copy.deepcopy(aggregate)
        _future_consolidate(bounded, future, train, seed=seed, epochs=20)

        for name, model in (("aggregate_only", aggregate), ("future_consolidated", future)):
            train_features, train_labels = _branch_features(bounded, model, train)
            validation_features, validation_labels = _branch_features(bounded, model, validation)
            assert int(torch.bincount(train_labels, minlength=2).min()) > 0
            assert int(torch.bincount(validation_labels, minlength=2).min()) > 0
            accuracy = _fit_fresh_probe(
                train_features,
                train_labels,
                validation_features,
                validation_labels,
                seed=seed,
            )
            rows[name].append(accuracy)

    warnings.warn(
        "LONG_RANGE_FUTURE_MEMORY="
        + json.dumps(
            {
                "marker_distance_note_events": 30,
                "local_window_audited": 16,
                "validation_branch_points_each_seed": 24,
                "aggregate_memory_epochs": 2,
                "future_consolidation_epochs": 20,
                "seeds": [1, 2, 3],
                "fresh_probe_accuracy": {
                    name: {"mean": statistics.mean(values), "by_seed": values}
                    for name, values in rows.items()
                },
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
