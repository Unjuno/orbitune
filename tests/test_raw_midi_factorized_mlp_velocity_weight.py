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


def test_raw_midi_factorized_mlp_velocity_weight_three_seed(tmp_path: Path) -> None:
    helper = _load(HELPER_SCRIPT, "orbitune_velocity_helper")
    memory = _load(helper.MEMORY_SCRIPT, "orbitune_velocity_memory")
    composer = _load(helper.COMPOSER_SCRIPT, "orbitune_velocity_composer")

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

    def weighted_loss(
        logits: dict[int, torch.Tensor], targets: torch.Tensor, *, velocity_weight: float
    ) -> torch.Tensor:
        weighted: list[torch.Tensor] = []
        weights: list[float] = []
        for index in composer.PREDICTED_FIELDS:
            mask = composer.active_mask(targets, index)
            if not mask.any():
                continue
            values = targets[..., index][mask]
            card = composer.FIELD_CARDS[index]
            loss = F.cross_entropy(logits[index][mask], values)
            weight = velocity_weight if index == 6 else 1.0
            weighted.append(loss * weight)
            weights.append(weight)
        if not weighted:
            raise ValueError("batch has no active Compound fields")
        return torch.stack(weighted).sum() / sum(weights)

    def train_weighted(
        memory_model: nn.Module,
        model: nn.Module,
        songs: list[object],
        *,
        seed: int,
        velocity_weight: float,
    ) -> None:
        for parameter in memory_model.parameters():
            parameter.requires_grad_(False)
        memory_model.eval()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        rng = random.Random(seed + 2903)
        for _ in range(12):
            order = list(songs)
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
                            memory_model, records, state
                        )
                    logits, history = model.forward_chunk(
                        records, memory_tokens, history, start_index=start
                    )
                    loss = weighted_loss(logits, targets, velocity_weight=velocity_weight)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    state = composer.memory_base._detach_state(next_state)

    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"velocity-{fixture_seed}.mid").write_bytes(helper._midi(fixture_seed))

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

    baseline_params = composer.parameter_count(WindowedGatedMLPComposer())
    assert baseline_params == 280_088

    weights = (1.0, 2.0, 4.0)
    rows: dict[str, list[dict[str, float]]] = {str(weight): [] for weight in weights}
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
        for weight in weights:
            torch.manual_seed(seed + 7000)
            random.seed(seed + 7000)
            model = WindowedGatedMLPComposer()
            train_weighted(consolidated, model, train, seed=seed, velocity_weight=weight)
            metric = composer.evaluate_factorized_composer(
                consolidated,
                model,
                validation,
                chunk_size=32,
                device=torch.device("cpu"),
            )
            rows[str(weight)].append(asdict(metric))

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

    warnings.warn(
        "RAW_MIDI_MLP_VELOCITY_WEIGHT="
        + json.dumps(
            {
                "files": 8,
                "seeds": [1, 2, 3],
                "memory_epochs": 2,
                "composer_epochs": 12,
                "composer_learning_rate": 1e-3,
                "local_window": 4,
                "trainable_params": baseline_params,
                "velocity_weights": list(weights),
                "summary": {key: summarize(value) for key, value in rows.items()},
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
