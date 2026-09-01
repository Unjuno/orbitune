from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import torch

from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_training import CompoundSong


SCRIPT = Path(__file__).parents[1] / "scripts" / "compound_cuda_train.py"
spec = importlib.util.spec_from_file_location("compound_cuda_train", SCRIPT)
assert spec is not None and spec.loader is not None
cuda_train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cuda_train)


def _tiny() -> CompoundBaseConfig:
    return CompoundBaseConfig(
        d_model=32,
        n_head=4,
        local_layers=1,
        medium_layers=1,
        global_layers=1,
        intra_layers=1,
        ff_mult=2,
        dropout=0.0,
        local_window=8,
        medium_stride=2,
        medium_window=8,
        global_stride=2,
        global_window=8,
    )


def _rows() -> list[tuple[int, ...]]:
    return [
        (4, 0, 0, 0, 120, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 2, 60, 0, 82, 0, 1, 4, 0, 0),
        (1, 0, 0, 0, 1, 64, 0, 0, 0, 0, 4, 3),
        (2, 0, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0),
        (9, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0),
        (6, 0, 0, 0, 512, 0, 0, 0, 0, 0, 4, 3),
        (0, 0, 1, 4, 67, 0, 91, 0, 1, 6, 0, 0),
        (5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
    ]


def test_fast_loss_matches_reference_loss() -> None:
    torch.manual_seed(7)
    model = CompoundHierarchicalGPT(_tiny()).eval()
    records = torch.tensor([_rows()], dtype=torch.long)
    inputs, targets = records[:, :-1], records[:, 1:]
    reference, _ = model(inputs, targets)
    optimized, _ = cuda_train.fast_loss(model, inputs, targets)
    assert torch.allclose(reference, optimized, atol=1e-6, rtol=1e-6)


def test_tensor_sampler_keeps_shift_alignment() -> None:
    song = CompoundSong(
        path="fixture.mid",
        sha256="fixture",
        tokenizer_abi="fixture",
        records=tuple(_rows()),
    )
    sampler = cuda_train.TensorSampler([song])
    inputs, targets = sampler.sample(2, 4, random.Random(1), torch.device("cpu"))
    assert inputs.shape == (2, 4, 12)
    assert targets.shape == (2, 4, 12)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_parser_exposes_hardware_tune_train() -> None:
    parser = cuda_train.build_parser()
    assert parser.parse_args(["hardware"]).command == "hardware"
    assert parser.parse_args(["tune", "--train-jsonl", "train.jsonl"]).command == "tune"
    assert parser.parse_args([
        "train", "--train-jsonl", "train.jsonl", "--checkpoint", "base.pt"
    ]).command == "train"
