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
import torch.nn as nn
import torch.nn.functional as F

from orbitune.compound_dataset import prepare_compound_split_corpus


HELPER_SCRIPT = Path(__file__).parent / "test_raw_midi_factorized_composer_converged_compare.py"


def _load(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_matched_transformer_w4_three_seed(tmp_path: Path) -> None:
    helper = _load(HELPER_SCRIPT, "orbitune_exact_w4_helper")
    memory = _load(helper.MEMORY_SCRIPT, "orbitune_exact_w4_memory")
    composer = _load(helper.COMPOSER_SCRIPT, "orbitune_exact_w4_composer")

    class ExactMatchedTransformerW4(composer.BoundedFactorizedTransformerComposer):
        def __init__(self) -> None:
            super().__init__(local_window=4)
            # W4 removes 12*48=576 position parameters relative to W16.
            # Put exactly 576 useful parameters back on the hidden-state loss path.
            self.match_down = nn.Linear(composer.D_MODEL, 6, bias=False)  # 288
            self.match_up = nn.Linear(6, composer.D_MODEL, bias=False)  # 288

        def condition_memory(
            self, hidden: torch.Tensor, memory_tokens: torch.Tensor
        ) -> torch.Tensor:
            hidden = hidden + self.match_up(F.gelu(self.match_down(hidden)))
            return super().condition_memory(hidden, memory_tokens)

    model_probe = ExactMatchedTransformerW4()
    assert composer.parameter_count(model_probe) == 280_088
    assert model_probe.local_position.num_embeddings == 4

    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"exact-w4-{fixture_seed}.mid").write_bytes(helper._midi(fixture_seed))

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

    rows: list[dict[str, float]] = []
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
        torch.manual_seed(seed + 7000)
        random.seed(seed + 7000)
        model = ExactMatchedTransformerW4()
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
        rows.append(
            asdict(
                composer.evaluate_factorized_composer(
                    consolidated,
                    model,
                    validation,
                    chunk_size=32,
                    device=torch.device("cpu"),
                )
            )
        )

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
    summary = {
        key: {
            "mean": statistics.mean(float(row[key]) for row in rows),
            "by_seed": [float(row[key]) for row in rows],
        }
        for key in metric_names
    }
    assert all(int(row["fields_scored"]) == 1404 for row in rows)
    assert all(int(row["events_scored"]) == 186 for row in rows)
    warnings.warn(
        "RAW_MIDI_EXACT_MATCHED_TRANSFORMER_W4="
        + json.dumps(
            {
                "files": 8,
                "seeds": [1, 2, 3],
                "memory_epochs": 2,
                "composer_epochs": 12,
                "composer_learning_rate": 1e-3,
                "local_window": 4,
                "trainable_params": 280_088,
                "all_params_on_loss_path": True,
                "summary": summary,
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
