from __future__ import annotations

import importlib.util
import json
import random
import statistics
import sys
import warnings
from pathlib import Path

import torch

from orbitune.compound_dataset import prepare_compound_split_corpus


HELPER_SCRIPT = Path(__file__).parent / "test_raw_midi_context_disambiguation.py"


def _load(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_local_context_branch_convergence_three_seed(tmp_path: Path) -> None:
    helper = _load(HELPER_SCRIPT, "orbitune_context_convergence_helper")
    memory = helper._load(helper.MEMORY_SCRIPT, "orbitune_context_convergence_memory")
    composer = helper._load(helper.COMPOSER_SCRIPT, "orbitune_context_convergence_composer")

    class MemoryAblatedComposer(composer.BoundedFactorizedTransformerComposer):
        def condition_memory(  # type: ignore[no-untyped-def]
            self, hidden: torch.Tensor, memory_tokens: torch.Tensor
        ) -> torch.Tensor:
            del memory_tokens
            return hidden + self.post_ff(self.post_norm(hidden))

    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"context-convergence-{fixture_seed}.mid").write_bytes(
            helper._midi(fixture_seed)
        )

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

    epochs_to_test = (8, 16, 24)
    results: dict[int, list[float]] = {epochs: [] for epochs in epochs_to_test}
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

        for epochs in epochs_to_test:
            torch.manual_seed(seed + 12_000)
            random.seed(seed + 12_000)
            model = MemoryAblatedComposer(local_window=16)
            model.local_window = 4
            assert composer.parameter_count(model) == 280_088
            composer.train_factorized_composer(
                consolidated,
                model,
                train,
                epochs=epochs,
                chunk_size=32,
                seed=seed,
                device=torch.device("cpu"),
                learning_rate=1e-3,
            )
            branch_accuracy, branch_count = helper._branch_pitch_accuracy(
                composer, consolidated, model, validation
            )
            assert branch_count == 40
            results[epochs].append(branch_accuracy)

    warnings.warn(
        "CONTEXT_DISAMBIGUATION_CONVERGENCE="
        + json.dumps(
            {
                "seeds": [1, 2, 3],
                "attention_window": 4,
                "memory_context": False,
                "learning_rate": 1e-3,
                "branch_points_each_seed": 40,
                "branch_pitch_accuracy": {
                    str(epochs): {
                        "mean": statistics.mean(values),
                        "by_seed": values,
                    }
                    for epochs, values in results.items()
                },
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
