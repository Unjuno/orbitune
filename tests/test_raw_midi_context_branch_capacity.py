from __future__ import annotations

import importlib.util
import json
import random
import statistics
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

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


def _note(track: bytearray, pitch: int) -> None:
    track += _vlq(12) + bytes([0x90, pitch, 80])
    track += _vlq(12) + bytes([0x80, pitch, 0])


def _midi(seed: int, patterns: int = 24) -> bytes:
    rng = random.Random(20_000 + seed)
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


def _branch_mask(records: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    current = records[..., 0].eq(0) & records[..., 4].eq(64)
    nxt = targets[..., 0].eq(0) & (targets[..., 4].eq(67) | targets[..., 4].eq(69))
    return current & nxt


def _marker_distances(songs) -> list[int]:  # type: ignore[no-untyped-def]
    distances: list[int] = []
    for song in songs:
        records = song.records
        for index in range(len(records) - 1):
            current = records[index]
            target = records[index + 1]
            if not (
                int(current[0]) == 0
                and int(current[4]) == 64
                and int(target[0]) == 0
                and int(target[4]) in (67, 69)
            ):
                continue
            found = None
            for distance in range(1, min(12, index + 1)):
                previous = records[index - distance]
                if int(previous[0]) == 0 and int(previous[4]) in (40, 52):
                    found = distance
                    break
            assert found is not None
            distances.append(found)
    return distances


def _train_branch_only(
    composer_module,  # type: ignore[no-untyped-def]
    memory_model,  # type: ignore[no-untyped-def]
    model,  # type: ignore[no-untyped-def]
    songs,  # type: ignore[no-untyped-def]
    *,
    epochs: int,
    seed: int,
) -> None:
    device = torch.device("cpu")
    for parameter in memory_model.parameters():
        parameter.requires_grad_(False)
    memory_model.eval()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    rng = random.Random(seed + 4117)
    for _ in range(epochs):
        order = list(songs)
        rng.shuffle(order)
        for song in order:
            state = None
            history = None
            final_input = len(song.records) - 1
            for start in range(0, final_input, 32):
                stop = min(final_input, start + 32)
                records = song.records[start:stop].unsqueeze(0)
                targets = song.records[start + 1 : stop + 1].unsqueeze(0)
                with torch.no_grad():
                    memory_tokens, next_state = composer_module.bounded.routed_memory_reads(
                        memory_model, records, state
                    )
                logits, history = model.forward_chunk(
                    records, memory_tokens, history, start_index=start
                )
                mask = _branch_mask(records, targets)
                if mask.any():
                    loss = F.cross_entropy(logits[4][mask], targets[..., 4][mask])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                state = composer_module.memory_base._detach_state(next_state)


def _evaluate_branch(
    composer_module,  # type: ignore[no-untyped-def]
    memory_model,  # type: ignore[no-untyped-def]
    model,  # type: ignore[no-untyped-def]
    songs,  # type: ignore[no-untyped-def]
) -> tuple[float, int]:
    model.eval()
    memory_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for song in songs:
            state = None
            history = None
            final_input = len(song.records) - 1
            for start in range(0, final_input, 32):
                stop = min(final_input, start + 32)
                records = song.records[start:stop].unsqueeze(0)
                targets = song.records[start + 1 : stop + 1].unsqueeze(0)
                memory_tokens, next_state = composer_module.bounded.routed_memory_reads(
                    memory_model, records, state
                )
                logits, history = model.forward_chunk(
                    records, memory_tokens, history, start_index=start
                )
                mask = _branch_mask(records, targets)
                if mask.any():
                    prediction = logits[4].argmax(-1)
                    correct += int(prediction[mask].eq(targets[..., 4][mask]).sum())
                    total += int(mask.sum())
                state = composer_module.memory_base._detach_state(next_state)
    return (correct / total if total else 0.0), total


def test_branch_specific_loss_isolates_local_context_capacity(tmp_path: Path) -> None:
    memory = _load(MEMORY_SCRIPT, "orbitune_branch_capacity_memory")
    composer = _load(COMPOSER_SCRIPT, "orbitune_branch_capacity_composer")

    class MemoryAblatedComposer(composer.BoundedFactorizedTransformerComposer):
        def condition_memory(self, hidden, memory_tokens):  # type: ignore[no-untyped-def]
            del memory_tokens
            return hidden + self.post_ff(self.post_norm(hidden))

    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"branch-capacity-{fixture_seed}.mid").write_bytes(_midi(fixture_seed))

    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    prepare_compound_split_corpus(
        source,
        train_jsonl,
        validation_jsonl,
        tmp_path / "split.json",
        validation_fraction=0.25,
        split_seed="raw-midi-context-branch-capacity-v1",
        min_events=32,
    )
    train, validation = memory.load_splits(train_jsonl, validation_jsonl)
    marker_distances = _marker_distances(validation)
    assert marker_distances and set(marker_distances) == {3}

    configs = {
        "current_only": {"memory": False, "window": 1},
        "memory_only": {"memory": True, "window": 1},
        "local_w3": {"memory": False, "window": 3},
        "local_w4": {"memory": False, "window": 4},
        "local_w6": {"memory": False, "window": 6},
    }
    rows: dict[str, list[float]] = {name: [] for name in configs}
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

        for name, config in configs.items():
            torch.manual_seed(seed + 21_000)
            random.seed(seed + 21_000)
            model_type = (
                composer.BoundedFactorizedTransformerComposer
                if config["memory"]
                else MemoryAblatedComposer
            )
            model = model_type(local_window=16)
            model.local_window = int(config["window"])
            assert composer.parameter_count(model) == 280_088
            _train_branch_only(
                composer,
                consolidated,
                model,
                train,
                epochs=16,
                seed=seed,
            )
            accuracy, count = _evaluate_branch(composer, consolidated, model, validation)
            rows[name].append(accuracy)
            if name == "current_only":
                counts.append(count)

    warnings.warn(
        "RAW_MIDI_CONTEXT_BRANCH_CAPACITY="
        + json.dumps(
            {
                "files": 8,
                "seeds": [1, 2, 3],
                "branch_points_each_seed": counts,
                "marker_distance_compound_events": sorted(set(marker_distances)),
                "branch_only_epochs": 16,
                "configs": configs,
                "branch_pitch_accuracy": {
                    name: {"by_seed": values, "mean": statistics.mean(values)}
                    for name, values in rows.items()
                },
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
