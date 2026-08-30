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


class ConsolidatedInterface(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(3 * 48)
        self.to_token = nn.Sequential(
            nn.Linear(3 * 48, 64),
            nn.GELU(),
            nn.Linear(64, 48),
        )
        self.classifier = nn.Linear(48, 2)

    def token(self, memory_slots: torch.Tensor) -> torch.Tensor:
        flat = memory_slots.reshape(memory_slots.shape[0], -1)
        return self.to_token(self.norm(flat))

    def forward(self, memory_slots: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.token(memory_slots))


def _train_interface_and_memory(helper, bounded, model, interface, songs, *, seed: int) -> None:  # type: ignore[no-untyped-def]
    torch.manual_seed(seed + 91_000)
    parameters = [*model.parameters(), *interface.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    rng = random.Random(seed + 91_000)
    model.train()
    interface.train()
    for _ in range(20):
        order = list(songs)
        rng.shuffle(order)
        for song in order:
            _, memory_slots, long_labels, _ = helper._committed_features(
                bounded, model, song
            )
            logits = interface(memory_slots)
            loss = F.cross_entropy(logits, long_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()


def _interface_accuracy(helper, bounded, model, interface, songs) -> float:  # type: ignore[no-untyped-def]
    model.eval()
    interface.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for song in songs:
            _, memory_slots, long_labels, _ = helper._committed_features(
                bounded, model, song
            )
            prediction = interface(memory_slots).argmax(-1)
            correct += int(prediction.eq(long_labels).sum())
            total += int(long_labels.numel())
    return correct / total if total else 0.0


def _dataset(helper, bounded, model, interface, songs, *, consolidated: bool):  # type: ignore[no-untyped-def]
    model.eval()
    interface.eval()
    rows = []
    with torch.no_grad():
        for song in songs:
            local, raw_slots, _, targets = helper._committed_features(
                bounded, model, song
            )
            if consolidated:
                slots = interface.token(raw_slots).unsqueeze(1)
            else:
                slots = raw_slots
            rows.append((local, slots, targets))
    return (
        torch.cat([row[0] for row in rows]),
        torch.cat([row[1] for row in rows]),
        torch.cat([row[2] for row in rows]),
    )


def _fit(helper, memory_base, train_data, validation_data, *, seed: int, use_local: bool, steps: int) -> float:  # type: ignore[no-untyped-def]
    torch.manual_seed(seed + 92_000)
    composer = helper.TwoScaleComposer(memory_base.FactorEmbedding)
    optimizer = torch.optim.AdamW(composer.parameters(), lr=2e-3)
    train_local, train_memory, train_targets = train_data
    val_local, val_memory, val_targets = validation_data
    composer.train()
    for _ in range(steps):
        logits = composer(
            train_local,
            train_memory,
            use_local=use_local,
            use_memory=True,
        )
        loss = F.cross_entropy(logits, train_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    composer.eval()
    with torch.no_grad():
        prediction = composer(
            val_local,
            val_memory,
            use_local=use_local,
            use_memory=True,
        ).argmax(-1)
    return float(prediction.eq(val_targets).float().mean())


def test_frozen_consolidated_token_reduces_memory_redecoding_burden(tmp_path: Path) -> None:
    helper = _load(HELPER, "orbitune_consolidated_token_helper")
    memory = _load(MEMORY_SCRIPT, "orbitune_consolidated_token_memory")
    bounded = _load(BOUNDED_SCRIPT, "orbitune_consolidated_token_bounded")

    source = tmp_path / "midi"
    source.mkdir()
    for fixture_seed in range(1, 9):
        (source / f"consolidated-token-{fixture_seed}.mid").write_bytes(
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
        split_seed="committed-hot-cold-v1",
        min_events=64,
    )
    train, validation = memory.load_splits(train_jsonl, validation_jsonl)
    helper._assert_partition_contract([*train, *validation])

    steps_to_test = (25, 50, 100, 250)
    rows = {
        "raw_memory_only": {steps: [] for steps in steps_to_test},
        "consolidated_token_only": {steps: [] for steps in steps_to_test},
        "raw_memory_plus_local": {steps: [] for steps in steps_to_test},
        "consolidated_token_plus_local": {steps: [] for steps in steps_to_test},
    }
    interface_accuracy: list[float] = []

    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        random.seed(seed)
        memory_model = memory.RoutedMultiBank()
        interface = ConsolidatedInterface()
        _train_interface_and_memory(
            helper, bounded, memory_model, interface, train, seed=seed
        )
        interface_accuracy.append(
            _interface_accuracy(helper, bounded, memory_model, interface, validation)
        )

        raw_train = _dataset(
            helper, bounded, memory_model, interface, train, consolidated=False
        )
        raw_validation = _dataset(
            helper, bounded, memory_model, interface, validation, consolidated=False
        )
        token_train = _dataset(
            helper, bounded, memory_model, interface, train, consolidated=True
        )
        token_validation = _dataset(
            helper, bounded, memory_model, interface, validation, consolidated=True
        )

        for steps in steps_to_test:
            rows["raw_memory_only"][steps].append(
                _fit(
                    helper,
                    memory.base,
                    raw_train,
                    raw_validation,
                    seed=seed,
                    use_local=False,
                    steps=steps,
                )
            )
            rows["consolidated_token_only"][steps].append(
                _fit(
                    helper,
                    memory.base,
                    token_train,
                    token_validation,
                    seed=seed,
                    use_local=False,
                    steps=steps,
                )
            )
            rows["raw_memory_plus_local"][steps].append(
                _fit(
                    helper,
                    memory.base,
                    raw_train,
                    raw_validation,
                    seed=seed,
                    use_local=True,
                    steps=steps,
                )
            )
            rows["consolidated_token_plus_local"][steps].append(
                _fit(
                    helper,
                    memory.base,
                    token_train,
                    token_validation,
                    seed=seed,
                    use_local=True,
                    steps=steps,
                )
            )

    warnings.warn(
        "CONSOLIDATED_MEMORY_TOKEN_INTERFACE="
        + json.dumps(
            {
                "files": 8,
                "seeds": [1, 2, 3],
                "hot_window": 4,
                "long_marker_distance": 30,
                "local_marker_distance": 3,
                "interface_consolidation_epochs": 20,
                "interface_long_bit_accuracy": {
                    "by_seed": interface_accuracy,
                    "mean": statistics.mean(interface_accuracy),
                },
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
