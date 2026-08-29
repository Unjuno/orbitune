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
BRANCH_HELPER = Path(__file__).parent / "test_raw_midi_context_branch_capacity.py"
MEMORY_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_memory_experiment_matched.py"
BOUNDED_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_bounded_composer.py"
COMPOSER_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_factorized_composer.py"


def _load(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_future_consolidated_memory_is_usable_by_frozen_memory_composer(tmp_path: Path) -> None:
    long = _load(LONG_HELPER, "orbitune_long_interface_helper")
    branch = _load(BRANCH_HELPER, "orbitune_long_interface_branch")
    memory = _load(MEMORY_SCRIPT, "orbitune_long_interface_memory")
    bounded = _load(BOUNDED_SCRIPT, "orbitune_long_interface_bounded")
    composer = _load(COMPOSER_SCRIPT, "orbitune_long_interface_composer")

    class MemoryAblatedComposer(composer.BoundedFactorizedTransformerComposer):
        def condition_memory(self, hidden, memory_tokens):  # type: ignore[no-untyped-def]
            del memory_tokens
            return hidden + self.post_ff(self.post_norm(hidden))

    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"long-interface-{fixture_seed}.mid").write_bytes(long._midi(fixture_seed))

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
    long._assert_local_context_is_ambiguous([*train, *validation], window=16)

    names = (
        "local_only",
        "aggregate_plus_local",
        "future_plus_local",
        "future_only",
    )
    rows: dict[str, list[float]] = {name: [] for name in names}
    counts: list[int] = []

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
        long._future_consolidate(bounded, future, train, seed=seed, epochs=20)

        configs = {
            "local_only": (aggregate, MemoryAblatedComposer, 16),
            "aggregate_plus_local": (
                aggregate,
                composer.BoundedFactorizedTransformerComposer,
                16,
            ),
            "future_plus_local": (
                future,
                composer.BoundedFactorizedTransformerComposer,
                16,
            ),
            "future_only": (
                future,
                composer.BoundedFactorizedTransformerComposer,
                1,
            ),
        }
        for name, (memory_model, model_type, window) in configs.items():
            torch.manual_seed(seed + 51_000)
            random.seed(seed + 51_000)
            model = model_type(local_window=16)
            model.local_window = window
            assert composer.parameter_count(model) == 280_088
            branch._train_branch_only(
                composer,
                memory_model,
                model,
                train,
                epochs=16,
                seed=seed,
            )
            accuracy, count = branch._evaluate_branch(
                composer,
                memory_model,
                model,
                validation,
            )
            rows[name].append(accuracy)
            if name == "local_only":
                counts.append(count)

    assert counts and min(counts) == 24
    warnings.warn(
        "LONG_RANGE_MEMORY_COMPOSER_INTERFACE="
        + json.dumps(
            {
                "files": 8,
                "seeds": [1, 2, 3],
                "marker_distance_note_events": 30,
                "local_window": 16,
                "validation_branch_points_each_seed": counts,
                "aggregate_memory_epochs": 2,
                "future_consolidation_epochs": 20,
                "composer_branch_epochs": 16,
                "memory_frozen_during_composer": True,
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
