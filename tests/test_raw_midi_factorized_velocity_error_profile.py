from __future__ import annotations

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


HELPER_SCRIPT = Path(__file__).parent / "test_raw_midi_factorized_composer_converged_compare.py"


def _load(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_raw_midi_factorized_velocity_error_profile_three_seed(tmp_path: Path) -> None:
    helper = _load(HELPER_SCRIPT, "orbitune_velocity_profile_helper")
    memory = _load(helper.MEMORY_SCRIPT, "orbitune_velocity_profile_memory")
    composer = _load(helper.COMPOSER_SCRIPT, "orbitune_velocity_profile_composer")

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
            hidden = self.local_out(F.silu(self.local_a(normalized)) * self.local_b(normalized))
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
            hidden, next_state = self.gru(self.embedding(records), history_records)
            hidden = self.local_norm(hidden)
            hidden = hidden + self.local_ff(hidden)
            hidden = hidden + self.local_calibration.mean() * torch.tanh(hidden)
            hidden = self.condition_memory(hidden, memory_tokens)
            logits = self.factorized_heads(self.output_norm(hidden))
            return logits, next_state.detach()

    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"profile-{fixture_seed}.mid").write_bytes(helper._midi(fixture_seed))
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

    model_types = {
        "no_local": composer.CapacityMatchedNoLocalComposer,
        "windowed_gated_mlp_w4": WindowedGatedMLPComposer,
        "gru_fixed_state": GRUComposer,
    }
    for model_type in model_types.values():
        assert composer.parameter_count(model_type()) == 280_088

    def profile(model: nn.Module, consolidated: nn.Module) -> dict[str, float]:
        model.eval()
        consolidated.eval()
        vel_count = 0
        argmax_abs = 0.0
        expected_abs = 0.0
        within4 = 0
        within8 = 0
        exact_no_vel = 0
        events = 0
        with torch.no_grad():
            for song in validation:
                state = None
                history = None
                final_input = len(song.records) - 1
                for start in range(0, final_input, 32):
                    stop = min(final_input, start + 32)
                    records = song.records[start:stop].unsqueeze(0)
                    targets = song.records[start + 1 : stop + 1].unsqueeze(0)
                    memory_tokens, next_state = composer.bounded.routed_memory_reads(
                        consolidated, records, state
                    )
                    logits, history = model.forward_chunk(
                        records, memory_tokens, history, start_index=start
                    )
                    preds = {idx: value.argmax(-1) for idx, value in logits.items()}
                    exact = torch.ones_like(targets[..., 0], dtype=torch.bool)
                    for idx in composer.PREDICTED_FIELDS:
                        if idx == 6:
                            continue
                        mask = composer.active_mask(targets, idx)
                        exact &= (~mask) | preds[idx].eq(targets[..., idx])
                    exact_no_vel += int(exact.sum())
                    events += targets.shape[0] * targets.shape[1]

                    note = targets[..., 0].eq(0)
                    if note.any():
                        target_v = targets[..., 6][note].float()
                        pred_v = preds[6][note].float()
                        probs = logits[6][note].softmax(-1)
                        classes = torch.arange(logits[6].shape[-1], dtype=probs.dtype)
                        expected_v = (probs * classes).sum(-1)
                        err = (pred_v - target_v).abs()
                        argmax_abs += float(err.sum())
                        expected_abs += float((expected_v - target_v).abs().sum())
                        within4 += int(err.le(4).sum())
                        within8 += int(err.le(8).sum())
                        vel_count += int(note.sum())
                    state = composer.memory_base._detach_state(next_state)
        return {
            "velocity_argmax_mae": argmax_abs / max(1, vel_count),
            "velocity_expected_mae": expected_abs / max(1, vel_count),
            "velocity_within4_accuracy": within4 / max(1, vel_count),
            "velocity_within8_accuracy": within8 / max(1, vel_count),
            "exact_event_without_velocity_accuracy": exact_no_vel / max(1, events),
        }

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
            standard = composer.evaluate_factorized_composer(
                consolidated,
                model,
                validation,
                chunk_size=32,
                device=torch.device("cpu"),
            )
            item = {
                "note_velocity_accuracy": standard.note_velocity_accuracy,
                "exact_event_accuracy": standard.exact_event_accuracy,
                **profile(model, consolidated),
            }
            rows[name].append(item)

    def summarize(items: list[dict[str, float]]) -> dict[str, object]:
        keys = items[0].keys()
        return {
            key: {
                "mean": statistics.mean(row[key] for row in items),
                "by_seed": [row[key] for row in items],
            }
            for key in keys
        }

    warnings.warn(
        "RAW_MIDI_VELOCITY_ERROR_PROFILE="
        + json.dumps(
            {
                "files": 8,
                "seeds": [1, 2, 3],
                "memory_epochs": 2,
                "composer_epochs": 12,
                "composer_learning_rate": 1e-3,
                "trainable_params_each": 280_088,
                "summary": {name: summarize(items) for name, items in rows.items()},
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
