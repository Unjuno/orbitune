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


def test_raw_midi_factorized_candidate_composers_three_seed(tmp_path: Path) -> None:
    helper = _load(HELPER_SCRIPT, "orbitune_candidate_helper")
    memory = _load(helper.MEMORY_SCRIPT, "orbitune_candidate_memory")
    composer = _load(helper.COMPOSER_SCRIPT, "orbitune_candidate_composer")

    class WindowedGatedMLPComposer(composer._MemoryConditionedBase):
        def __init__(self, *, local_window: int = 4, heads: int = 4) -> None:
            if local_window != 4:
                raise ValueError("candidate is exactly parameter-matched for local_window=4")
            super().__init__(heads=heads)
            self.local_window = local_window
            width = local_window * composer.D_MODEL
            self.local_norm = nn.LayerNorm(width)
            self.local_a = nn.Linear(width, 55)
            self.local_b = nn.Linear(width, 55)
            self.local_out = nn.Linear(55, composer.D_MODEL)
            self.local_calibration = nn.Parameter(torch.zeros(82))

        def forward_chunk(
            self,
            records: torch.Tensor,
            memory_tokens: torch.Tensor,
            history_records: torch.Tensor | None,
            *,
            start_index: int,
        ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
            del start_index
            if history_records is None:
                full_records = records
                history_length = 0
            else:
                full_records = torch.cat([history_records, records], dim=1)
                history_length = history_records.shape[1]

            embedded = self.embedding(full_records)
            batch, steps, width = embedded.shape
            padding = embedded.new_zeros(batch, self.local_window - 1, width)
            padded = torch.cat([padding, embedded], dim=1)
            windowed = torch.stack(
                [padded[:, offset : offset + steps] for offset in range(self.local_window)],
                dim=2,
            ).reshape(batch, steps, self.local_window * width)
            normalized = self.local_norm(windowed)
            hidden = self.local_out(
                F.silu(self.local_a(normalized)) * self.local_b(normalized)
            )
            hidden = hidden + self.local_calibration.mean() * torch.tanh(hidden)
            current = hidden[:, history_length:]
            current = self.condition_memory(current, memory_tokens)
            logits = self.factorized_heads(self.output_norm(current))
            keep = min(self.local_window - 1, full_records.shape[1])
            history = full_records[:, -keep:].detach() if keep else full_records[:, :0]
            return logits, history

    class GRUComposer(composer._MemoryConditionedBase):
        def __init__(self, *, heads: int = 4) -> None:
            super().__init__(heads=heads)
            self.gru = nn.GRU(composer.D_MODEL, composer.D_MODEL, batch_first=True)
            self.local_norm = nn.LayerNorm(composer.D_MODEL)
            self.local_ff = nn.Sequential(
                nn.Linear(composer.D_MODEL, 104),
                nn.GELU(),
                nn.Linear(104, composer.D_MODEL),
            )
            self.local_calibration = nn.Parameter(torch.zeros(40))

        def forward_chunk(
            self,
            records: torch.Tensor,
            memory_tokens: torch.Tensor,
            history_records: torch.Tensor | None,
            *,
            start_index: int,
        ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
            del start_index
            initial_state = history_records
            hidden, next_state = self.gru(self.embedding(records), initial_state)
            hidden = self.local_norm(hidden)
            hidden = hidden + self.local_ff(hidden)
            hidden = hidden + self.local_calibration.mean() * torch.tanh(hidden)
            hidden = self.condition_memory(hidden, memory_tokens)
            logits = self.factorized_heads(self.output_norm(hidden))
            return logits, next_state.detach()

    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"candidate-{fixture_seed}.mid").write_bytes(helper._midi(fixture_seed))

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

    def bounded_transformer_w4():  # type: ignore[no-untyped-def]
        model = composer.BoundedFactorizedTransformerComposer(local_window=16)
        model.local_window = 4
        return model

    model_types = {
        "no_local": composer.CapacityMatchedNoLocalComposer,
        "bounded_transformer_w4": bounded_transformer_w4,
        "windowed_gated_mlp_w4": WindowedGatedMLPComposer,
        "gru_fixed_state": GRUComposer,
    }
    baseline_params = composer.parameter_count(composer.CapacityMatchedNoLocalComposer())
    assert baseline_params == 280_088
    for model_type in model_types.values():
        assert composer.parameter_count(model_type()) == baseline_params

    rows: dict[str, list[dict[str, float]]] = {name: [] for name in model_types}
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

        for name, model_type in model_types.items():
            torch.manual_seed(seed + 7000)
            random.seed(seed + 7000)
            model = model_type()
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
            metric = composer.evaluate_factorized_composer(
                consolidated,
                model,
                validation,
                chunk_size=32,
                device=torch.device("cpu"),
            )
            rows[name].append(asdict(metric))

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

    def summarize(items: list[dict[str, float]]) -> dict[str, object]:
        return {
            key: {
                "mean": statistics.mean(float(row[key]) for row in items),
                "by_seed": [float(row[key]) for row in items],
            }
            for key in metric_names
        }

    for name in model_types:
        assert len({int(row["fields_scored"]) for row in rows[name]}) == 1
        assert len({int(row["events_scored"]) for row in rows[name]}) == 1
        assert int(rows[name][0]["fields_scored"]) == 1404
        assert int(rows[name][0]["events_scored"]) == 186

    warnings.warn(
        "RAW_MIDI_FACTORIZED_CANDIDATES="
        + json.dumps(
            {
                "files": 8,
                "seeds": [1, 2, 3],
                "memory_epochs": 2,
                "composer_epochs": 12,
                "composer_learning_rate": 1e-3,
                "trainable_params_each": baseline_params,
                "local_window": 4,
                "gru_state": "fixed-size hidden state carried across chunks",
                "summary": {name: summarize(rows[name]) for name in model_types},
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
