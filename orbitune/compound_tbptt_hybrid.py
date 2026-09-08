"""Hybrid TBPTT: chunk-vectorized local (A3) + exact per-position hierarchies.

Background: time-batching medium/global via a single causal forward is exact
only for single-layer stacks. With multi-layer stacks, layer-2+ mixes
intermediate rows that were computed under different visibility than the
reference's prefix-limited recompute, once history eviction starts. Measured:
global_history divergence ~0.13 after ~2000 positions with everything else at
1e-6. The causal-gather path is therefore NOT semantics-preserving
post-eviction and is excluded from the recommended candidate (kept in git
history as a rejected experiment).

This hybrid keeps the proven wins:
- A3 unfold windows for local (bit-identical integer construction),
- per-position lane-batched medium/global/memory/fusion via the validated
  advance_all_lanes (same per-position histories as the reference, batched
  across lanes only),
- identical StreamState layout, reset isolation, detach, save/load.
"""
from __future__ import annotations

import torch

from orbitune.compound_base import CompoundHierarchicalGPT, StreamState
from orbitune.compound_tbptt_batched import advance_all_lanes
from orbitune.compound_tbptt_chunkvec import _local_windows_a3


def encode_tbptt_hybrid(
    model: CompoundHierarchicalGPT,
    records: torch.Tensor,
    states: list[StreamState],
    *,
    reset_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[StreamState]]:
    """Same contract as encode_tbptt_chunk."""
    if records.ndim != 3 or records.shape[-1] != 12:
        raise ValueError("records must have shape [batch, time, 12]")
    batch, steps, _ = records.shape
    if len(states) != batch:
        raise ValueError("one StreamState is required per batch lane")
    if reset_mask is None:
        reset_mask = torch.zeros(batch, dtype=torch.bool, device=records.device)
    if reset_mask.numel() != batch:
        raise ValueError("reset_mask must have one entry per batch lane")
    for lane in range(batch):
        if bool(reset_mask[lane].item()):
            states[lane] = model.initial_stream_state()
    device = next(model.parameters()).device
    local_h, _ = _local_windows_a3(model, records, states, device)
    per_step: list[torch.Tensor] = []
    for step in range(steps):
        per_step.append(advance_all_lanes(model, records[:, step], states,
                                          local_hiddens=local_h[:, step]))
    return torch.stack(per_step, dim=1), states


def tbptt_loss_hybrid(
    model: CompoundHierarchicalGPT,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    states: list[StreamState],
    *,
    reset_mask: torch.Tensor | None = None,
    event_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float], list[StreamState]]:
    contexts, states = encode_tbptt_hybrid(model, inputs, states, reset_mask=reset_mask)
    loss, parts = model.decoder.loss(contexts, targets, event_weight=event_weight)
    return loss, parts, states
