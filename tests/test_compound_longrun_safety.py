from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import torch
import torch.nn as nn

from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_longrun import build_longrun_checkpoint, restore_longrun_rng, safe_backward_step


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compound_longrun_train.py"
SPEC = importlib.util.spec_from_file_location("compound_longrun_train_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
longrun = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(longrun)


def _tiny_model() -> CompoundHierarchicalGPT:
    return CompoundHierarchicalGPT(
        CompoundBaseConfig(
            d_model=32,
            n_head=2,
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
    )


def test_longrun_checkpoint_keeps_global_and_sampler_rng_separate() -> None:
    random.seed(123)
    sampler_rng = random.Random(456)
    model = _tiny_model()
    expected_global = random.getstate()
    expected_sampler = sampler_rng.getstate()

    payload = build_longrun_checkpoint(
        model=model,
        optimizer=None,
        scaler=None,
        step=7,
        events_seen=700,
        runtime={"n_head": 2, "seq_len": 8, "batch_size": 1, "precision": "fp32", "causal_fastpath": True},
        sampler_rng=sampler_rng,
    )
    assert payload["python_rng_state"] == expected_global
    assert payload["sampler_rng_state"] == expected_sampler

    expected_global_next = random.random()
    expected_sampler_next = sampler_rng.random()
    random.seed(999)
    sampler_rng.seed(999)
    restore_longrun_rng(payload, sampler_rng)
    assert random.random() == expected_global_next
    assert sampler_rng.random() == expected_sampler_next


def test_safe_backward_step_mutates_only_after_finite_checks() -> None:
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.ones(1, 2)).square().sum()
    result = safe_backward_step(loss=loss, model=model, optimizer=optimizer, scaler=None, grad_clip=1.0)
    assert result.stepped
    assert result.failure is None
    assert not torch.equal(before, model.weight.detach())


def test_nonfinite_loss_skips_optimizer_mutation() -> None:
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.ones(1, 2)).sum() * torch.tensor(float("nan"))
    result = safe_backward_step(loss=loss, model=model, optimizer=optimizer, scaler=None, grad_clip=1.0)
    assert not result.stepped
    assert result.failure == "non_finite_loss"
    torch.testing.assert_close(before, model.weight.detach())


class _FiniteForwardNanBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full_like(grad_output, float("nan"))


def test_nonfinite_gradient_skips_optimizer_mutation() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    model = nn.ParameterList([parameter])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = _FiniteForwardNanBackward.apply(parameter).sum()
    result = safe_backward_step(loss=loss, model=model, optimizer=optimizer, scaler=None, grad_clip=1.0)
    assert not result.stepped
    assert result.failure == "non_finite_gradient"
    torch.testing.assert_close(before, parameter.detach())


def test_production_parser_requires_explicit_fixed_window_acknowledgement_at_runtime() -> None:
    parser = longrun.build_parser()
    args = parser.parse_args([
        "--train-jsonl", "train.jsonl",
        "--validation-jsonl", "val.jsonl",
        "--checkpoint", "run.pt",
        "--steps", "10",
    ])
    assert args.allow_fixed_window_training is False
