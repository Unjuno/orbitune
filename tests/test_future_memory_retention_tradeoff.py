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
import torch.nn.functional as F

from orbitune.compound_dataset import prepare_compound_split_corpus


LONG_HELPER = Path(__file__).parent / "test_long_range_future_memory_objective.py"
ARCH_HELPER = Path(__file__).parent / "test_raw_midi_memory_architecture_compare.py"


def _load(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _future_epoch(long_helper, bounded, memory_model, head, songs, optimizer, rng) -> None:  # type: ignore[no-untyped-def]
    order = list(songs)
    rng.shuffle(order)
    memory_model.train()
    head.train()
    for song in order:
        records = song.records.unsqueeze(0)
        tokens, _ = bounded.routed_memory_reads(memory_model, records, None)
        examples = long_helper._branch_examples(song)
        indices = torch.tensor([item[0] for item in examples], dtype=torch.long)
        labels = torch.tensor([item[1] for item in examples], dtype=torch.long)
        branch_features = tokens[0, indices].reshape(len(examples), -1)
        loss = F.cross_entropy(head(branch_features), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*memory_model.parameters(), *head.parameters()], 1.0
        )
        optimizer.step()


def _future_with_rehearsal(
    long_helper,
    bounded,
    memory,
    model,
    future_train,
    retention_train,
    *,
    seed: int,
    epochs: int = 20,
) -> None:  # type: ignore[no-untyped-def]
    torch.manual_seed(seed + 50_000)
    head = long_helper.Probe(3 * 48)
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *head.parameters()], lr=1e-3
    )
    rng = random.Random(seed + 50_000)
    for epoch in range(epochs):
        _future_epoch(long_helper, bounded, model, head, future_train, optimizer, rng)
        # One low-LR rehearsal epoch preserves the descriptive memory targets
        # while the predictive objective teaches generation-critical retrieval.
        memory.train_memory_stage(
            model,
            retention_train,
            epochs=1,
            chunk_size=32,
            warmup_events=8,
            seed=seed + epoch,
            device=torch.device("cpu"),
            learning_rate=2e-4,
        )


def _memory_metrics(memory, model, songs):  # type: ignore[no-untyped-def]
    metric = memory.evaluate(
        model,
        songs,
        chunk_size=32,
        warmup_events=8,
        device=torch.device("cpu"),
    )
    return {
        "fast": metric.fast_macro_recall,
        "medium": metric.medium_macro_recall,
        "slow": metric.slow_macro_recall,
    }


def test_predictive_memory_objective_retention_tradeoff(tmp_path: Path) -> None:
    long_helper = _load(LONG_HELPER, "orbitune_retention_long_helper")
    arch_helper = _load(ARCH_HELPER, "orbitune_retention_arch_helper")
    memory = long_helper._load(
        long_helper.MEMORY_SCRIPT, "orbitune_retention_memory"
    )
    bounded = long_helper._load(
        long_helper.BOUNDED_SCRIPT, "orbitune_retention_bounded"
    )

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
    long_helper._assert_local_context_is_ambiguous(
        [*future_train, *future_validation], window=16
    )

    conditions = ("aggregate_only", "future_only", "future_plus_rehearsal")
    retrieval: dict[str, list[float]] = {name: [] for name in conditions}
    retained: dict[str, list[dict[str, float]]] = {name: [] for name in conditions}

    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        random.seed(seed)
        aggregate = memory.RoutedMultiBank()
        memory.train_memory_stage(
            aggregate,
            retention_train,
            epochs=2,
            chunk_size=32,
            warmup_events=8,
            seed=seed,
            device=torch.device("cpu"),
            learning_rate=2e-3,
        )
        future_only = copy.deepcopy(aggregate)
        future_rehearsal = copy.deepcopy(aggregate)

        long_helper._future_consolidate(
            bounded, future_only, future_train, seed=seed, epochs=20
        )
        _future_with_rehearsal(
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
            ("aggregate_only", aggregate),
            ("future_only", future_only),
            ("future_plus_rehearsal", future_rehearsal),
        ):
            train_features, train_labels = long_helper._branch_features(
                bounded, model, future_train
            )
            validation_features, validation_labels = long_helper._branch_features(
                bounded, model, future_validation
            )
            retrieval[name].append(
                long_helper._fit_fresh_probe(
                    train_features,
                    train_labels,
                    validation_features,
                    validation_labels,
                    seed=seed,
                )
            )
            retained[name].append(_memory_metrics(memory, model, retention_validation))

    def summarize_memory(rows: list[dict[str, float]]) -> dict[str, float]:
        return {
            tier: statistics.mean(row[tier] for row in rows)
            for tier in ("fast", "medium", "slow")
        }

    warnings.warn(
        "FUTURE_MEMORY_RETENTION="
        + json.dumps(
            {
                "marker_distance_note_events": 30,
                "future_epochs": 20,
                "rehearsal_lr": 2e-4,
                "seeds": [1, 2, 3],
                "fresh_probe_accuracy": {
                    name: {
                        "mean": statistics.mean(values),
                        "by_seed": values,
                    }
                    for name, values in retrieval.items()
                },
                "retention_memory_macro_recall": {
                    name: summarize_memory(rows) for name, rows in retained.items()
                },
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
