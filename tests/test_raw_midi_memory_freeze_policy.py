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


SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_memory_experiment_matched.py"


def _load_module():  # type: ignore[no-untyped-def]
    name = "orbitune_raw_midi_memory_freeze_policy"
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


def _metrics(metric) -> dict[str, float]:  # type: ignore[no-untyped-def]
    return {
        "fast": metric.fast_macro_recall,
        "medium": metric.medium_macro_recall,
        "slow": metric.slow_macro_recall,
        "event": metric.next_event_type_accuracy,
    }


def test_raw_midi_freeze_low_lr_joint_three_seed(tmp_path: Path) -> None:
    module = _load_module()
    source = tmp_path / "midi"
    source.mkdir()
    for seed in range(1, 9):
        (source / f"policy-{seed}.mid").write_bytes(_midi(seed))

    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    report_json = tmp_path / "split.json"
    prepare_compound_split_corpus(
        source,
        train_jsonl,
        validation_jsonl,
        report_json,
        validation_fraction=0.25,
        split_seed="raw-midi-freeze-policy-v1",
        min_events=32,
    )
    train, validation = module.load_splits(train_jsonl, validation_jsonl)

    policies = ("frozen", "low_lr", "joint")
    rows: dict[str, list[dict[str, float]]] = {policy: [] for policy in policies}
    before_rows: list[dict[str, float]] = []
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        random.seed(seed)
        consolidated = module.RoutedMultiBank()
        module.train_memory_stage(
            consolidated,
            train,
            epochs=2,
            chunk_size=32,
            warmup_events=8,
            seed=seed,
            device=torch.device("cpu"),
            learning_rate=2e-3,
        )
        before = module.evaluate(
            consolidated,
            validation,
            chunk_size=32,
            warmup_events=8,
            device=torch.device("cpu"),
        )
        before_rows.append(_metrics(before))

        for policy in policies:
            model = copy.deepcopy(consolidated)
            module.train_composer_stage(
                model,
                train,
                policy=policy,
                epochs=2,
                chunk_size=32,
                seed=seed,
                device=torch.device("cpu"),
                composer_lr=2e-3,
                memory_lr_multiplier=0.1,
            )
            after = module.evaluate(
                model,
                validation,
                chunk_size=32,
                warmup_events=8,
                device=torch.device("cpu"),
            )
            row = _metrics(after)
            row.update(
                {
                    "fast_delta": row["fast"] - before.fast_macro_recall,
                    "medium_delta": row["medium"] - before.medium_macro_recall,
                    "slow_delta": row["slow"] - before.slow_macro_recall,
                }
            )
            rows[policy].append(row)

    def mean_rows(items: list[dict[str, float]]) -> dict[str, float]:
        keys = items[0].keys()
        return {key: statistics.mean(row[key] for row in items) for key in keys}

    summary = {
        "before": mean_rows(before_rows),
        **{policy: mean_rows(rows[policy]) for policy in policies},
    }
    # Frozen policy is a hard invariant: composer-only training cannot change
    # the consolidated memory objective outputs.
    assert abs(summary["frozen"]["fast_delta"]) < 1e-12
    assert abs(summary["frozen"]["medium_delta"]) < 1e-12
    assert abs(summary["frozen"]["slow_delta"]) < 1e-12
    warnings.warn(
        "RAW_MIDI_FREEZE_POLICY="
        + json.dumps(
            {
                "files": 8,
                "seeds": [1, 2, 3],
                "memory_epochs": 2,
                "composer_epochs": 2,
                "summary": summary,
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
