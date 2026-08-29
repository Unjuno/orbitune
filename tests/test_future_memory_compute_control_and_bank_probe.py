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


LONG_HELPER = Path(__file__).parent / "test_long_range_future_memory_objective.py"
ARCH_HELPER = Path(__file__).parent / "test_raw_midi_memory_architecture_compare.py"
RETENTION_HELPER = Path(__file__).parent / "test_future_memory_retention_tradeoff.py"


def _load(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bank_probe_accuracy(long_helper, bounded, model, train, validation, seed: int):  # type: ignore[no-untyped-def]
    train_features, train_labels = long_helper._branch_features(bounded, model, train)
    validation_features, validation_labels = long_helper._branch_features(
        bounded, model, validation
    )
    width = train_features.shape[-1] // 3
    slices = {
        "fast": slice(0, width),
        "medium": slice(width, 2 * width),
        "slow": slice(2 * width, 3 * width),
        "all": slice(0, 3 * width),
    }
    return {
        name: long_helper._fit_fresh_probe(
            train_features[:, part],
            train_labels,
            validation_features[:, part],
            validation_labels,
            seed=seed + offset,
        )
        for offset, (name, part) in enumerate(slices.items())
    }


def test_future_memory_compute_control_and_bank_localization(tmp_path: Path) -> None:
    long_helper = _load(LONG_HELPER, "orbitune_compute_long_helper")
    arch_helper = _load(ARCH_HELPER, "orbitune_compute_arch_helper")
    retention_helper = _load(RETENTION_HELPER, "orbitune_compute_retention_helper")
    memory = long_helper._load(long_helper.MEMORY_SCRIPT, "orbitune_compute_memory")
    bounded = long_helper._load(long_helper.BOUNDED_SCRIPT, "orbitune_compute_bounded")

    retention_source = tmp_path / "retention-midi"
    retention_source.mkdir()
    for fixture_seed in range(1, 9):
        (retention_source / f"retention-{fixture_seed}.mid").write_bytes(
            arch_helper._midi(fixture_seed)
        )
    retention_train_jsonl = tmp_path / "retention-train.jsonl"
    retention_validation_jsonl = tmp_path / "retention-validation.jsonl"
    prepare_compound_split_corpus(
        retention_source,
        retention_train_jsonl,
        retention_validation_jsonl,
        tmp_path / "retention-split.json",
        validation_fraction=0.25,
        split_seed="raw-midi-memory-architecture-compare-v1",
        min_events=32,
    )
    retention_train, retention_validation = memory.load_splits(
        retention_train_jsonl, retention_validation_jsonl
    )

    future_source = tmp_path / "future-midi"
    future_source.mkdir()
    for fixture_seed in range(1, 9):
        (future_source / f"future-{fixture_seed}.mid").write_bytes(
            long_helper._midi(fixture_seed)
        )
    future_train_jsonl = tmp_path / "future-train.jsonl"
    future_validation_jsonl = tmp_path / "future-validation.jsonl"
    prepare_compound_split_corpus(
        future_source,
        future_train_jsonl,
        future_validation_jsonl,
        tmp_path / "future-split.json",
        validation_fraction=0.25,
        split_seed="long-range-future-memory-v1",
        min_events=64,
    )
    future_train, future_validation = memory.load_splits(
        future_train_jsonl, future_validation_jsonl
    )

    conditions = ("base", "aggregate_extra", "future_plus_rehearsal")
    retention_rows: dict[str, list[dict[str, float]]] = {
        name: [] for name in conditions
    }
    retrieval_rows: dict[str, list[float]] = {name: [] for name in conditions}
    bank_rows: dict[str, list[dict[str, float]]] = {name: [] for name in conditions}

    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        random.seed(seed)
        base = memory.RoutedMultiBank()
        memory.train_memory_stage(
            base,
            retention_train,
            epochs=2,
            chunk_size=32,
            warmup_events=8,
            seed=seed,
            device=torch.device("cpu"),
            learning_rate=2e-3,
        )
        aggregate_extra = copy.deepcopy(base)
        future_rehearsal = copy.deepcopy(base)

        # Matched control: exactly the same number and LR of additional
        # descriptive-rehearsal epochs as the multi-objective arm, without the
        # future-prediction updates.
        for epoch in range(20):
            memory.train_memory_stage(
                aggregate_extra,
                retention_train,
                epochs=1,
                chunk_size=32,
                warmup_events=8,
                seed=seed + epoch,
                device=torch.device("cpu"),
                learning_rate=2e-4,
            )
        retention_helper._future_with_rehearsal(
            long_helper,
            bounded,
            memory,
            future_rehearsal,
            future_train,
            retention_train,
            seed=seed,
            epochs=20,
        )

        for name, model in (
            ("base", base),
            ("aggregate_extra", aggregate_extra),
            ("future_plus_rehearsal", future_rehearsal),
        ):
            retention_rows[name].append(
                retention_helper._memory_metrics(memory, model, retention_validation)
            )
            probes = _bank_probe_accuracy(
                long_helper,
                bounded,
                model,
                future_train,
                future_validation,
                seed,
            )
            bank_rows[name].append(probes)
            retrieval_rows[name].append(probes["all"])

    def mean_memory(rows: list[dict[str, float]]) -> dict[str, float]:
        return {
            tier: statistics.mean(row[tier] for row in rows)
            for tier in ("fast", "medium", "slow")
        }

    def mean_banks(rows: list[dict[str, float]]) -> dict[str, float]:
        return {
            bank: statistics.mean(row[bank] for row in rows)
            for bank in ("fast", "medium", "slow", "all")
        }

    warnings.warn(
        "FUTURE_MEMORY_COMPUTE_CONTROL="
        + json.dumps(
            {
                "seeds": [1, 2, 3],
                "marker_distance_note_events": 30,
                "extra_aggregate_epochs": 20,
                "extra_aggregate_lr": 2e-4,
                "future_epochs": 20,
                "fresh_probe_accuracy_all_banks": {
                    name: statistics.mean(values)
                    for name, values in retrieval_rows.items()
                },
                "fresh_probe_accuracy_by_bank": {
                    name: mean_banks(rows) for name, rows in bank_rows.items()
                },
                "retention_memory_macro_recall": {
                    name: mean_memory(rows) for name, rows in retention_rows.items()
                },
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
