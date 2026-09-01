from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
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


def test_raw_midi_mlp_vs_transformer_long_convergence_cpu(tmp_path: Path) -> None:
    helper = _load(HELPER_SCRIPT, "orbitune_long_convergence_helper")
    memory = _load(helper.MEMORY_SCRIPT, "orbitune_long_convergence_memory")
    composer = _load(helper.COMPOSER_SCRIPT, "orbitune_long_convergence_composer")

    class WindowedGatedMLPComposer(composer._MemoryConditionedBase):
        def __init__(self) -> None:
            super().__init__(heads=4)
            self.local_window = 4
            width = self.local_window * composer.D_MODEL
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

    class ExactMatchedTransformerW4(composer.BoundedFactorizedTransformerComposer):
        def __init__(self) -> None:
            super().__init__(local_window=4)
            self.match_down = nn.Linear(composer.D_MODEL, 6, bias=False)
            self.match_up = nn.Linear(6, composer.D_MODEL, bias=False)

        def condition_memory(
            self, hidden: torch.Tensor, memory_tokens: torch.Tensor
        ) -> torch.Tensor:
            hidden = hidden + self.match_up(F.gelu(self.match_down(hidden)))
            return super().condition_memory(hidden, memory_tokens)

    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"long-convergence-{fixture_seed}.mid").write_bytes(helper._midi(fixture_seed))

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

    seed = 1
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
    for parameter in consolidated.parameters():
        parameter.requires_grad_(False)
    consolidated.eval()

    model_types = {
        "windowed_gated_mlp_w4": WindowedGatedMLPComposer,
        "transformer_w4_exact": ExactMatchedTransformerW4,
    }
    for model_type in model_types.values():
        assert composer.parameter_count(model_type()) == 280_088

    checkpoints = (12, 24, 48, 96)
    rows: dict[str, dict[str, dict[str, float]]] = {}
    seconds: dict[str, float] = {}

    for name, model_type in model_types.items():
        torch.manual_seed(seed + 7000)
        random.seed(seed + 7000)
        model = model_type().to("cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        rng = random.Random(seed + 2903)
        rows[name] = {}
        started = time.perf_counter()
        for epoch in range(1, checkpoints[-1] + 1):
            model.train()
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
                    logits, history = model.forward_chunk(
                        records, memory_tokens, history, start_index=start
                    )
                    loss = composer.factorized_loss(logits, targets)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    state = composer.memory_base._detach_state(next_state)
            if epoch in checkpoints:
                metric = composer.evaluate_factorized_composer(
                    consolidated,
                    model,
                    validation,
                    chunk_size=32,
                    device=torch.device("cpu"),
                )
                rows[name][str(epoch)] = asdict(metric)
        seconds[name] = time.perf_counter() - started

    warnings.warn(
        "RAW_MIDI_LONG_CONVERGENCE="
        + json.dumps(
            {
                "seed": seed,
                "memory_epochs": 2,
                "composer_learning_rate": 1e-3,
                "checkpoints": list(checkpoints),
                "trainable_params_each": 280_088,
                "local_window": 4,
                "continuous_optimizer_state": True,
                "rows": rows,
                "wall_seconds": seconds,
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
