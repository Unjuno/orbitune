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


def _velocity_targets(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = values.float() / 127.0
    coarse = torch.clamp((normalized * 8.0).long(), min=0, max=7)
    lo = coarse.float() / 8.0
    local = (normalized - lo) * 8.0
    residual = torch.clamp(torch.round(local * 7.0).long(), min=0, max=7)
    return coarse, residual


def _decode_velocity(coarse: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    normalized = coarse.float() / 8.0 + residual.float() / 7.0 / 8.0
    return torch.clamp(torch.round(normalized * 127.0), min=0, max=127).long()


def test_raw_midi_factorized_note_velocity_representation_three_seed(tmp_path: Path) -> None:
    helper = _load(HELPER_SCRIPT, "orbitune_velocity_repr_helper")
    memory = _load(helper.MEMORY_SCRIPT, "orbitune_velocity_repr_memory")
    composer = _load(helper.COMPOSER_SCRIPT, "orbitune_velocity_repr_composer")

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

        def local_hidden(
            self,
            records: torch.Tensor,
            memory_tokens: torch.Tensor,
            history_records: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
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
            current = self.output_norm(current)
            keep = min(self.local_window - 1, full_records.shape[1])
            history = full_records[:, -keep:].detach() if keep else full_records[:, :0]
            return current, history

        def forward_chunk(
            self,
            records: torch.Tensor,
            memory_tokens: torch.Tensor,
            history_records: torch.Tensor | None,
            *,
            start_index: int,
        ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
            del start_index
            current, history = self.local_hidden(records, memory_tokens, history_records)
            return self.factorized_heads(current), history

    class FactorizedVelocityMLP(WindowedGatedMLPComposer):
        def __init__(self) -> None:
            super().__init__()
            # Replace the 48->128 raw NOTE velocity head (6,272 params) with two
            # velocity-specific 48->hidden->8 heads plus 43 calibration params.
            # 3,143 + 3,086 + 43 = 6,272, so the total remains exactly 280,088.
            del self.factorized_heads.heads["6"]
            self.velocity_coarse = nn.Sequential(
                nn.Linear(composer.D_MODEL, 55),
                nn.GELU(),
                nn.Linear(55, 8),
            )
            self.velocity_residual = nn.Sequential(
                nn.Linear(composer.D_MODEL, 54),
                nn.GELU(),
                nn.Linear(54, 8),
            )
            self.velocity_calibration = nn.Parameter(torch.zeros(43))

        def forward_factorized_chunk(
            self,
            records: torch.Tensor,
            memory_tokens: torch.Tensor,
            history_records: torch.Tensor | None,
            *,
            start_index: int,
        ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
            del start_index
            current, history = self.local_hidden(records, memory_tokens, history_records)
            logits = self.factorized_heads(current)
            calibration = self.velocity_calibration.mean() * torch.tanh(current)
            velocity_hidden = current + calibration
            return (
                logits,
                self.velocity_coarse(velocity_hidden),
                self.velocity_residual(velocity_hidden),
                history,
            )

    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"velocity-repr-{fixture_seed}.mid").write_bytes(helper._midi(fixture_seed))
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

    raw_params = composer.parameter_count(WindowedGatedMLPComposer())
    factorized_params = composer.parameter_count(FactorizedVelocityMLP())
    assert raw_params == factorized_params == 280_088

    def train_factorized_velocity(
        consolidated: nn.Module,
        model: FactorizedVelocityMLP,
        *,
        seed: int,
    ) -> None:
        for parameter in consolidated.parameters():
            parameter.requires_grad_(False)
        consolidated.eval()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        rng = random.Random(seed + 2903)
        for _ in range(12):
            order = list(train)
            rng.shuffle(order)
            for song in order:
                if len(song.records) < 2:
                    continue
                state = None
                history = None
                final_input = len(song.records) - 1
                for start in range(0, final_input, 32):
                    stop = min(final_input, start + 32)
                    records = song.records[start:stop].unsqueeze(0)
                    targets = song.records[start + 1 : stop + 1].unsqueeze(0)
                    with torch.no_grad():
                        memory_tokens, next_state = composer.bounded.routed_memory_reads(
                            consolidated, records, state
                        )
                    logits, vel_c, vel_r, history = model.forward_factorized_chunk(
                        records, memory_tokens, history, start_index=start
                    )
                    losses: list[torch.Tensor] = []
                    for index in composer.PREDICTED_FIELDS:
                        mask = composer.active_mask(targets, index)
                        if not mask.any():
                            continue
                        if index == 6:
                            values = targets[..., 6][mask]
                            coarse, residual = _velocity_targets(values)
                            velocity_loss = 0.5 * (
                                F.cross_entropy(vel_c[mask], coarse)
                                + F.cross_entropy(vel_r[mask], residual)
                            )
                            losses.append(velocity_loss)
                        else:
                            values = targets[..., index][mask]
                            losses.append(F.cross_entropy(logits[index][mask], values))
                    loss = torch.stack(losses).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    state = composer.memory_base._detach_state(next_state)

    def evaluate_factorized_velocity(
        consolidated: nn.Module,
        model: FactorizedVelocityMLP,
    ) -> dict[str, float]:
        consolidated.eval()
        model.eval()
        nonvel_correct = 0
        nonvel_total = 0
        nonvel_nll = 0.0
        velocity_exact = 0
        velocity_count = 0
        velocity_abs = 0.0
        velocity_within4 = 0
        velocity_within8 = 0
        exact_no_velocity = 0
        exact_with_velocity = 0
        events = 0
        pitch_correct = 0
        notes = 0
        duration_correct = 0
        delta_correct = 0
        with torch.no_grad():
            for song in validation:
                if len(song.records) < 2:
                    continue
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
                    logits, vel_c, vel_r, history = model.forward_factorized_chunk(
                        records, memory_tokens, history, start_index=start
                    )
                    predictions = {idx: value.argmax(-1) for idx, value in logits.items()}
                    vel_pred = _decode_velocity(vel_c.argmax(-1), vel_r.argmax(-1))
                    exact_nonvel = torch.ones_like(targets[..., 0], dtype=torch.bool)
                    for index in composer.PREDICTED_FIELDS:
                        if index == 6:
                            continue
                        mask = composer.active_mask(targets, index)
                        if not mask.any():
                            continue
                        values = targets[..., index]
                        log_probs = F.log_softmax(logits[index], dim=-1)
                        chosen = log_probs.gather(-1, values.unsqueeze(-1)).squeeze(-1)
                        nonvel_nll += float((-chosen[mask]).sum())
                        nonvel_correct += int(predictions[index][mask].eq(values[mask]).sum())
                        nonvel_total += int(mask.sum())
                        exact_nonvel &= (~mask) | predictions[index].eq(values)
                    note = targets[..., 0].eq(0)
                    if note.any():
                        target_v = targets[..., 6]
                        err = (vel_pred[note] - target_v[note]).abs()
                        velocity_exact += int(vel_pred[note].eq(target_v[note]).sum())
                        velocity_abs += float(err.sum())
                        velocity_within4 += int(err.le(4).sum())
                        velocity_within8 += int(err.le(8).sum())
                        velocity_count += int(note.sum())
                        pitch_correct += int(predictions[4][note].eq(targets[..., 4][note]).sum())
                        duration_correct += int(
                            (
                                predictions[8][note].eq(targets[..., 8][note])
                                & predictions[9][note].eq(targets[..., 9][note])
                            ).sum()
                        )
                        notes += int(note.sum())
                    exact_no_velocity += int(exact_nonvel.sum())
                    exact_with = exact_nonvel & ((~note) | vel_pred.eq(targets[..., 6]))
                    exact_with_velocity += int(exact_with.sum())
                    delta_correct += int(
                        (
                            predictions[2].eq(targets[..., 2])
                            & predictions[3].eq(targets[..., 3])
                        ).sum()
                    )
                    events += targets.shape[0] * targets.shape[1]
                    state = composer.memory_base._detach_state(next_state)
        return {
            "nonvelocity_active_accuracy": nonvel_correct / max(1, nonvel_total),
            "nonvelocity_active_nll": nonvel_nll / max(1, nonvel_total),
            "note_pitch_accuracy": pitch_correct / max(1, notes),
            "note_duration_pair_accuracy": duration_correct / max(1, notes),
            "delta_pair_accuracy": delta_correct / max(1, events),
            "velocity_exact_accuracy": velocity_exact / max(1, velocity_count),
            "velocity_mae": velocity_abs / max(1, velocity_count),
            "velocity_within4_accuracy": velocity_within4 / max(1, velocity_count),
            "velocity_within8_accuracy": velocity_within8 / max(1, velocity_count),
            "exact_event_without_velocity_accuracy": exact_no_velocity / max(1, events),
            "exact_event_with_velocity_accuracy": exact_with_velocity / max(1, events),
        }

    def evaluate_raw(
        consolidated: nn.Module,
        model: WindowedGatedMLPComposer,
    ) -> dict[str, float]:
        consolidated.eval()
        model.eval()
        nonvel_correct = 0
        nonvel_total = 0
        nonvel_nll = 0.0
        velocity_exact = 0
        velocity_count = 0
        velocity_abs = 0.0
        velocity_within4 = 0
        velocity_within8 = 0
        exact_no_velocity = 0
        exact_with_velocity = 0
        events = 0
        pitch_correct = 0
        notes = 0
        duration_correct = 0
        delta_correct = 0
        with torch.no_grad():
            for song in validation:
                if len(song.records) < 2:
                    continue
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
                    predictions = {idx: value.argmax(-1) for idx, value in logits.items()}
                    exact_nonvel = torch.ones_like(targets[..., 0], dtype=torch.bool)
                    for index in composer.PREDICTED_FIELDS:
                        if index == 6:
                            continue
                        mask = composer.active_mask(targets, index)
                        if not mask.any():
                            continue
                        values = targets[..., index]
                        log_probs = F.log_softmax(logits[index], dim=-1)
                        chosen = log_probs.gather(-1, values.unsqueeze(-1)).squeeze(-1)
                        nonvel_nll += float((-chosen[mask]).sum())
                        nonvel_correct += int(predictions[index][mask].eq(values[mask]).sum())
                        nonvel_total += int(mask.sum())
                        exact_nonvel &= (~mask) | predictions[index].eq(values)
                    note = targets[..., 0].eq(0)
                    if note.any():
                        target_v = targets[..., 6]
                        pred_v = predictions[6]
                        err = (pred_v[note] - target_v[note]).abs()
                        velocity_exact += int(pred_v[note].eq(target_v[note]).sum())
                        velocity_abs += float(err.sum())
                        velocity_within4 += int(err.le(4).sum())
                        velocity_within8 += int(err.le(8).sum())
                        velocity_count += int(note.sum())
                        pitch_correct += int(predictions[4][note].eq(targets[..., 4][note]).sum())
                        duration_correct += int(
                            (
                                predictions[8][note].eq(targets[..., 8][note])
                                & predictions[9][note].eq(targets[..., 9][note])
                            ).sum()
                        )
                        notes += int(note.sum())
                    exact_no_velocity += int(exact_nonvel.sum())
                    exact_with = exact_nonvel & ((~note) | predictions[6].eq(targets[..., 6]))
                    exact_with_velocity += int(exact_with.sum())
                    delta_correct += int(
                        (
                            predictions[2].eq(targets[..., 2])
                            & predictions[3].eq(targets[..., 3])
                        ).sum()
                    )
                    events += targets.shape[0] * targets.shape[1]
                    state = composer.memory_base._detach_state(next_state)
        return {
            "nonvelocity_active_accuracy": nonvel_correct / max(1, nonvel_total),
            "nonvelocity_active_nll": nonvel_nll / max(1, nonvel_total),
            "note_pitch_accuracy": pitch_correct / max(1, notes),
            "note_duration_pair_accuracy": duration_correct / max(1, notes),
            "delta_pair_accuracy": delta_correct / max(1, events),
            "velocity_exact_accuracy": velocity_exact / max(1, velocity_count),
            "velocity_mae": velocity_abs / max(1, velocity_count),
            "velocity_within4_accuracy": velocity_within4 / max(1, velocity_count),
            "velocity_within8_accuracy": velocity_within8 / max(1, velocity_count),
            "exact_event_without_velocity_accuracy": exact_no_velocity / max(1, events),
            "exact_event_with_velocity_accuracy": exact_with_velocity / max(1, events),
        }

    rows: dict[str, list[dict[str, float]]] = {"raw_128": [], "factorized_8x8": []}
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
        raw_model = WindowedGatedMLPComposer()
        composer.train_factorized_composer(
            consolidated,
            raw_model,
            train,
            epochs=12,
            chunk_size=32,
            seed=seed,
            device=torch.device("cpu"),
            learning_rate=1e-3,
        )
        rows["raw_128"].append(evaluate_raw(consolidated, raw_model))

        torch.manual_seed(seed + 7000)
        random.seed(seed + 7000)
        factorized_model = FactorizedVelocityMLP()
        train_factorized_velocity(consolidated, factorized_model, seed=seed)
        rows["factorized_8x8"].append(
            evaluate_factorized_velocity(consolidated, factorized_model)
        )

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
        "RAW_MIDI_NOTE_VELOCITY_REPRESENTATION="
        + json.dumps(
            {
                "files": 8,
                "seeds": [1, 2, 3],
                "memory_epochs": 2,
                "composer_epochs": 12,
                "composer_learning_rate": 1e-3,
                "local_window": 4,
                "trainable_params_each": 280_088,
                "velocity_factorization": {"coarse_levels": 8, "residual_levels": 8},
                "summary": {name: summarize(items) for name, items in rows.items()},
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
