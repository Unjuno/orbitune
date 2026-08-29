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


HELPER = Path(__file__).parent / "test_committed_memory_hot_cold_complementarity.py"
MEMORY_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_memory_experiment_matched.py"
BOUNDED_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_bounded_composer.py"


def _load(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class OrderedComposer(nn.Module):
    """Same parameters; only cross-memory/local-attention order changes."""

    def __init__(self, factor_embedding, *, memory_first: bool) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.memory_first = memory_first
        self.embedding = factor_embedding()
        self.position = nn.Embedding(4, 48)
        self.local_norm = nn.LayerNorm(48)
        self.local_attention = nn.MultiheadAttention(48, 4, dropout=0.0, batch_first=True)
        self.local_ff_norm = nn.LayerNorm(48)
        self.local_ff = nn.Sequential(nn.Linear(48, 144), nn.GELU(), nn.Linear(144, 48))
        self.memory_norm = nn.LayerNorm(48)
        self.memory_attention = nn.MultiheadAttention(48, 4, dropout=0.0, batch_first=True)
        self.output_norm = nn.LayerNorm(48)
        self.head = nn.Linear(48, 4)

    def _local(self, hidden: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(4, device=hidden.device)
        hidden = hidden + self.position(pos)[None]
        normed = self.local_norm(hidden)
        causal = torch.triu(
            torch.ones(4, 4, dtype=torch.bool, device=hidden.device), diagonal=1
        )
        attended, _ = self.local_attention(
            normed, normed, normed, attn_mask=causal, need_weights=False
        )
        hidden = hidden + attended
        return hidden + self.local_ff(self.local_ff_norm(hidden))

    def _memory_sequence(
        self, hidden: torch.Tensor, memory_tokens: torch.Tensor
    ) -> torch.Tensor:
        query = self.memory_norm(hidden)
        read, _ = self.memory_attention(
            query, memory_tokens, memory_tokens, need_weights=False
        )
        return hidden + read

    def _memory_query(
        self, query: torch.Tensor, memory_tokens: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.memory_norm(query).unsqueeze(1)
        read, _ = self.memory_attention(
            normalized, memory_tokens, memory_tokens, need_weights=False
        )
        return query + read[:, 0]

    def forward(
        self, local_records: torch.Tensor, memory_tokens: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.embedding(local_records)
        if self.memory_first:
            # User-proposed ordering: consolidated cold memory conditions the
            # exact hot sequence before the bounded Transformer composes it.
            hidden = self._memory_sequence(hidden, memory_tokens)
            hidden = self._local(hidden)
            query = hidden[:, -1]
        else:
            hidden = self._local(hidden)
            query = self._memory_query(hidden[:, -1], memory_tokens)
        return self.head(self.output_norm(query))


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _fit_steps(
    helper,  # type: ignore[no-untyped-def]
    memory_base,  # type: ignore[no-untyped-def]
    train_data,
    validation_data,
    *,
    seed: int,
    memory_first: bool,
    steps: int,
) -> float:
    torch.manual_seed(seed + 81_000)
    model = OrderedComposer(memory_base.FactorEmbedding, memory_first=memory_first)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    train_local, train_memory, train_targets = train_data
    val_local, val_memory, val_targets = validation_data
    model.train()
    for _ in range(steps):
        logits = model(train_local, train_memory)
        loss = F.cross_entropy(logits, train_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = model(val_local, val_memory).argmax(-1)
    return float(prediction.eq(val_targets).float().mean())


def test_memory_before_local_vs_local_before_memory_convergence(tmp_path: Path) -> None:
    helper = _load(HELPER, "orbitune_order_hot_cold_helper")
    memory = _load(MEMORY_SCRIPT, "orbitune_order_memory")
    bounded = _load(BOUNDED_SCRIPT, "orbitune_order_bounded")

    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"order-{fixture_seed}.mid").write_bytes(helper._midi(fixture_seed))

    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    prepare_compound_split_corpus(
        source,
        train_jsonl,
        validation_jsonl,
        tmp_path / "split.json",
        validation_fraction=0.25,
        split_seed="committed-hot-cold-v1",
        min_events=64,
    )
    train, validation = memory.load_splits(train_jsonl, validation_jsonl)
    helper._assert_partition_contract([*train, *validation])

    reference_before = OrderedComposer(memory.base.FactorEmbedding, memory_first=True)
    reference_after = OrderedComposer(memory.base.FactorEmbedding, memory_first=False)
    assert _parameter_count(reference_before) == _parameter_count(reference_after)

    steps_to_test = (25, 50, 100, 250)
    rows = {
        "memory_before_local": {steps: [] for steps in steps_to_test},
        "local_before_memory": {steps: [] for steps in steps_to_test},
    }

    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        random.seed(seed)
        memory_model = memory.RoutedMultiBank()
        helper._future_consolidate_committed(
            bounded, memory_model, train, seed=seed, epochs=20
        )
        train_data = helper._dataset(bounded, memory_model, train)
        validation_data = helper._dataset(bounded, memory_model, validation)

        for label, memory_first in (
            ("memory_before_local", True),
            ("local_before_memory", False),
        ):
            for steps in steps_to_test:
                rows[label][steps].append(
                    _fit_steps(
                        helper,
                        memory.base,
                        train_data,
                        validation_data,
                        seed=seed,
                        memory_first=memory_first,
                        steps=steps,
                    )
                )

    warnings.warn(
        "MEMORY_CONDITIONING_ORDER="
        + json.dumps(
            {
                "files": 8,
                "seeds": [1, 2, 3],
                "hot_window": 4,
                "long_marker_distance": 30,
                "local_marker_distance": 3,
                "future_consolidation_epochs": 20,
                "parameter_count_each": _parameter_count(reference_before),
                "four_way_accuracy_by_training_steps": {
                    label: {
                        str(steps): {
                            "by_seed": values,
                            "mean": statistics.mean(values),
                        }
                        for steps, values in step_rows.items()
                    }
                    for label, step_rows in rows.items()
                },
            },
            sort_keys=True,
        ),
        RuntimeWarning,
        stacklevel=1,
    )
