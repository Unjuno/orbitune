from __future__ import annotations

import importlib.util
import json
import random
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import torch

from orbitune.compound_dataset import prepare_compound_split_corpus


MEMORY_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_memory_experiment_matched.py"
COMPOSER_SCRIPT = Path(__file__).parents[1] / "experiments" / "real_compound_factorized_composer.py"


def _load(path: Path, name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _vlq(value: int) -> bytes:
    parts = [value & 0x7F]
    value >>= 7
    while value:
        parts.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(parts))


def _midi(seed: int, notes: int = 64) -> bytes:
    motifs = (
        (48, 55, 60, 64, 60, 55),
        (50, 57, 60, 67, 60, 57),
        (52, 59, 62, 65, 62, 59),
        (47, 54, 59, 62, 59, 54),
    )
    track = bytearray()
    track += b"\x00\xff\x51\x03" + int(400_000 + seed * 45_000).to_bytes(3, "big")
    track += b"\x00" + bytes([0xC0, (seed * 11) % 96])
    track += b"\x00" + bytes([0xC1, (seed * 17 + 16) % 112])
    for index in range(notes):
        channel = index % 2
        motif = motifs[(index // 8 + seed) % len(motifs)]
        pitch = motif[index % len(motif)] + (12 if channel else 0)
        if index and index % 18 == 0:
            track += b"\x00" + bytes([0xC0 | channel, (seed * 13 + index * 5) % 128])
        if index % 15 == 0:
            track += b"\x00" + bytes([0xB0 | channel, 64, 127 if (index // 15 + seed) % 2 else 0])
        if index and index % 22 == 0:
            tempo = 330_000 + ((seed + index) % 5) * 80_000
            track += b"\x00\xff\x51\x03" + tempo.to_bytes(3, "big")
        velocity = 36 + ((seed * 17 + index * 13) % 84)
        gap = (6, 12, 18, 24)[(index + seed) % 4]
        duration = (12, 18, 24, 30)[(index // 2 + seed) % 4]
        track += _vlq(gap) + bytes([0x90 | channel, pitch, velocity])
        track += _vlq(duration) + bytes([0x80 | channel, pitch, 0])
    track += b"\x00\xff\x2f\x00"
    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + (96).to_bytes(2, "big")
    return header + b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)


def test_factorized_transformer_lr_epoch_convergence_sweep(tmp_path: Path) -> None:
    memory = _load(MEMORY_SCRIPT, "orbitune_transformer_sweep_memory")
    composer = _load(COMPOSER_SCRIPT, "orbitune_transformer_sweep_composer")
    source = tmp_path / "midi"
    source.mkdir()
    for seed in range(1, 7):
        (source / f"sweep-{seed}.mid").write_bytes(_midi(seed))
    train_jsonl = tmp_path / "train.jsonl"
    validation_jsonl = tmp_path / "validation.jsonl"
    prepare_compound_split_corpus(
        source,
        train_jsonl,
        validation_jsonl,
        tmp_path / "split.json",
        validation_fraction=0.25,
        split_seed="factorized-transformer-convergence-v1",
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

    configs = (
        (3, 2e-3),
        (6, 2e-3),
        (6, 1e-3),
        (12, 1e-3),
        (12, 5e-4),
    )
    results: list[dict[str, object]] = []
    for index, (epochs, learning_rate) in enumerate(configs):
        torch.manual_seed(5000 + index)
        model = composer.BoundedFactorizedTransformerComposer(local_window=16)
        composer.train_factorized_composer(
            consolidated,
            model,
            train,
            epochs=epochs,
            chunk_size=32,
            seed=seed,
            device=torch.device("cpu"),
            learning_rate=learning_rate,
        )
        metric = composer.evaluate_factorized_composer(
            consolidated,
            model,
            validation,
            chunk_size=32,
            device=torch.device("cpu"),
        )
        results.append(
            {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "metrics": asdict(metric),
            }
        )

    warnings.warn(
        "FACTORIZED_TRANSFORMER_CONVERGENCE="
        + json.dumps({"seed": seed, "configs": results}, sort_keys=True),
        RuntimeWarning,
        stacklevel=1,
    )
