from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import torch

from orbitune.compound_training import (
    build_compound_checkpoint,
    restore_cuda_rng_state,
)


@dataclass(frozen=True, slots=True)
class OptimizerStepResult:
    loss_value: float
    grad_norm: float | None
    stepped: bool
    failure: str | None = None


def build_longrun_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    step: int,
    events_seen: int,
    runtime: dict[str, object],
    sampler_rng: random.Random,
    **kwargs: Any,
) -> dict[str, object]:
    """Build a Compound checkpoint with distinct global and sampler RNG states.

    ``build_compound_checkpoint`` predates the dedicated sampler RNG and used
    the supplied RNG for ``python_rng_state``.  The production long-run path
    must preserve both streams independently or a resumed run samples a
    different next window.
    """
    payload = build_compound_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=step,
        events_seen=events_seen,
        runtime=runtime,
        rng=sampler_rng,
        **kwargs,
    )
    payload["python_rng_state"] = random.getstate()
    payload["sampler_rng_state"] = sampler_rng.getstate()
    return payload


def restore_longrun_rng(payload: dict[str, object], sampler_rng: random.Random) -> None:
    """Restore every RNG stream used by the production trainer."""
    torch_state = payload.get("torch_rng_state")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.detach().to(device="cpu", dtype=torch.uint8))

    python_state = payload.get("python_rng_state")
    if python_state is not None:
        random.setstate(python_state)

    sampler_state = payload.get("sampler_rng_state")
    if sampler_state is not None:
        sampler_rng.setstate(sampler_state)

    cuda_state = payload.get("cuda_rng_state_all")
    if cuda_state is not None:
        restore_cuda_rng_state(cuda_state)


def safe_backward_step(
    *,
    loss: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    grad_clip: float,
) -> OptimizerStepResult:
    """Validate loss/gradients before mutating optimizer state.

    The legacy CUDA helper performed ``optimizer.step`` before the caller
    inspected finiteness.  This routine makes the ordering explicit:

        finite loss -> backward -> unscale -> finite grad norm -> clip -> step

    A non-finite loss or gradient leaves model and optimizer parameters
    untouched for the step.
    """
    loss_value = float(loss.detach().float().cpu())
    if not math.isfinite(loss_value):
        optimizer.zero_grad(set_to_none=True)
        return OptimizerStepResult(loss_value, None, False, "non_finite_loss")

    use_scaler = scaler is not None and scaler.is_enabled()
    if use_scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
        parameters,
        grad_clip,
        error_if_nonfinite=False,
    )
    grad_norm = float(grad_norm_tensor.detach().float().cpu())
    if not math.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        if use_scaler:
            # unscale_ populated GradScaler's inf/NaN bookkeeping. Updating
            # without scaler.step reduces the scale while skipping mutation.
            scaler.update()
        return OptimizerStepResult(loss_value, grad_norm, False, "non_finite_gradient")

    if use_scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return OptimizerStepResult(loss_value, grad_norm, True, None)
