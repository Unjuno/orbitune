"""Lane-batched state-carry TBPTT step (perf candidate, branch perf/tbptt-lane-batching).

Same intended semantics as orbitune.compound_tbptt.encode_tbptt_chunk, but the
per-(time, lane) Python loop is restructured to one batched tensor operation
per submodule per time step:

    reference:  for t: for lane: advance(...)
    candidate:  for t: advance_all_lanes(...)

Time dependence stays sequential. Per-lane song state (records, buffers,
histories, memory, steps) is kept in the SAME StreamState layout, updated with
the same values up to floating-point batching order. Resets still touch only
their own lane. Detach boundaries, sampler, save/load formats are untouched.

Deliberate known difference (see module docstring in handover): with
dropout > 0 in train mode, one batched submodule call consumes a different
dropout-mask stream than B sequential calls, so the train trajectory is a
different-but-valid regime (CASE B per task spec), NOT bit-identical.
With dropout disabled (eval) the computation is mathematically identical.
"""
from __future__ import annotations

import torch

from orbitune.compound_base import CompoundHierarchicalGPT, StreamState


def _front_pad_stack(items: list[list[torch.Tensor]], dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length per-lane lists with front (oldest-side) padding.

    Returns (padded [B, Tmax, dim], lengths [B]). Newest element is always at
    index Tmax-1 so the last position can be read uniformly. Padding value is
    arbitrary because callers mask padded keys with -inf bias.
    """
    batch = len(items)
    lengths = torch.tensor([len(v) for v in items], dtype=torch.long)
    tmax = int(lengths.max().item()) if batch else 0
    ref = None
    for v in items:
        if v:
            ref = v[0]
            break
    assert ref is not None and tmax > 0
    out = torch.zeros(batch, tmax, dim, device=ref.device, dtype=ref.dtype)
    for b, v in enumerate(items):
        if v:
            stacked = torch.stack(v)
            out[b, tmax - len(v):] = stacked.to(device=out.device, dtype=out.dtype)
    return out, lengths


def _padded_causal_bias(lengths: torch.Tensor, tmax: int, device: torch.device,
                        *, window: int | None) -> torch.Tensor:
    """Float bias [B, 1, T, T]: -inf for future keys, window violations, pad keys.

    NOTE: the explicit singleton head dim is required. A [B, T, T] float mask
    is silently misinterpreted by SDPA on torch 2.5.1 (verified: wrong
    values); [B, 1, T, T] is bit-exact vs per-sample 2D masks.
    """
    batch = lengths.numel()
    row = torch.arange(tmax, device=device)[:, None]
    col = torch.arange(tmax, device=device)[None, :]
    allowed = col <= row
    if window is not None:
        allowed = allowed & ((row - col) < window)
    bias = torch.zeros(batch, 1, tmax, tmax, device=device, dtype=torch.float32)
    bias = bias.masked_fill(~allowed, float("-inf"))
    # pad keys: first (tmax - length) columns are padding for each lane
    pad = torch.arange(tmax, device=device)[None, :] < (tmax - lengths.to(device))[:, None]
    bias = bias.masked_fill(pad[:, None, None, :], float("-inf"))
    return bias


def _batched_stack_forward(model: CompoundHierarchicalGPT, stack: torch.nn.Module,
                           items: list[list[torch.Tensor]], dim: int,
                           active: list[bool], device: torch.device,
                           *, window: int | None) -> list[torch.Tensor | None]:
    """Run one batched transformer forward for the active lanes.

    Returns per-lane last-position outputs (or None for inactive lanes).
    """
    out: list[torch.Tensor | None] = [None] * len(items)
    idx = [b for b, a in enumerate(active) if a]
    if not idx:
        return out
    sub = [items[b] for b in idx]
    padded, lengths = _front_pad_stack(sub, dim)
    bias = _padded_causal_bias(lengths, padded.shape[1], device, window=window)
    hidden = stack(padded, bias)[:, -1]
    for k, b in enumerate(idx):
        out[b] = hidden[k]
    return out


def advance_all_lanes(
    model: CompoundHierarchicalGPT,
    cur_records: torch.Tensor,
    states: list[StreamState],
    *,
    local_hiddens: torch.Tensor | None = None,
) -> torch.Tensor:
    """One time step for every lane; returns stacked contexts [B, D].

    When local_hiddens [B, D] is supplied (e.g. chunk-vectorized A3 windows),
    the local Transformer forward is skipped, but raw-record append/evict
    still runs so lane state stays identical.
    """
    device = next(model.parameters()).device
    batch = len(states)
    d_model = model.config.d_model

    # 1. local records: append + evict per lane (same rule as reference).
    for b in range(batch):
        raw = cur_records[b].to(device=device, dtype=torch.long).reshape(12)
        states[b].local_records.append(raw)
        if len(states[b].local_records) > model.config.local_window:
            states[b].local_records.pop(0)
    if local_hiddens is None:
        lengths = torch.tensor([len(s.local_records) for s in states], dtype=torch.long)
        tmax = int(lengths.max().item())
        pad_rec = torch.zeros(batch, tmax, 12, device=device, dtype=torch.long)
        for b in range(batch):
            recs = states[b].local_records
            pad_rec[b, tmax - len(recs):] = torch.stack(recs)
        local_event = model.embedding(pad_rec)
        local_out = model.local(
            local_event,
            _padded_causal_bias(lengths, tmax, device, window=model.config.local_window),
        )[:, -1]
        local_hiddens = [local_out[b] for b in range(batch)]
    else:
        local_hiddens = [local_hiddens[b] for b in range(batch)]

    # 2. recurrent memory, batched across lanes (None -> zeros per lane).
    event_emb = model.embedding(cur_records.to(device=device)[:, None, :])[:, 0]
    mem_states = []
    mem_is_none = []
    for b in range(batch):
        mem = states[b].memory
        mem_is_none.append(mem is None)
        mem_states.append(mem)
    stacked_mem: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    if not all(mem_is_none):
        parts: list[list[torch.Tensor]] = [[], [], []]
        for mem in mem_states:
            for k in range(3):
                parts[k].append(mem[k] if mem is not None
                                else event_emb.new_zeros(d_model))
        stacked_mem = (torch.stack(parts[0]), torch.stack(parts[1]), torch.stack(parts[2]))
    memory_read, new_mem = model.memory.step(event_emb, stacked_mem)
    for b in range(batch):
        states[b].memory = (new_mem[0][b], new_mem[1][b], new_mem[2][b])
    memory_reads = [memory_read[b] for b in range(batch)]

    # 3. medium summaries for lanes whose stride boundary fires.
    for b in range(batch):
        states[b].medium_buffer.append(local_hiddens[b])
    firing = [len(s.medium_buffer) >= model.config.medium_stride for s in states]
    if any(firing):
        # buffers have equal length (stride) for all firing lanes
        stacked_bufs = torch.stack([torch.stack(states[b].medium_buffer) for b in range(batch) if firing[b]])
        summaries = stacked_bufs.mean(dim=1)
        k = 0
        for b in range(batch):
            if not firing[b]:
                continue
            states[b].medium_buffer.clear()
            states[b].medium_history.append(summaries[k])
            k += 1
            if len(states[b].medium_history) > model.config.medium_window:
                states[b].medium_history.pop(0)
        # medium_out for firing lanes (batched; mirrors reference completion call)
        outs = _batched_stack_forward(model, model.medium,
                                      [states[b].medium_history for b in range(batch)],
                                      d_model, firing, device, window=None)
        for b in range(batch):
            if firing[b]:
                assert outs[b] is not None
                states[b].global_buffer.append(outs[b])

    # 4. global summaries for lanes whose stride boundary fires.
    gfiring = [len(s.global_buffer) >= model.config.global_stride for s in states]
    if any(gfiring):
        stacked_g = torch.stack([torch.stack(states[b].global_buffer) for b in range(batch) if gfiring[b]])
        gsummaries = stacked_g.mean(dim=1)
        k = 0
        for b in range(batch):
            if not gfiring[b]:
                continue
            states[b].global_buffer.clear()
            states[b].global_history.append(gsummaries[k])
            k += 1
            if len(states[b].global_history) > model.config.global_window:
                states[b].global_history.pop(0)

    # 5. medium/global contexts for lanes with nonempty history.
    mactive = [len(s.medium_history) > 0 for s in states]
    mouts = _batched_stack_forward(model, model.medium,
                                   [states[b].medium_history for b in range(batch)],
                                   d_model, mactive, device, window=None)
    gactive = [len(s.global_history) > 0 for s in states]
    gouts = _batched_stack_forward(model, model.global_stack,
                                   [states[b].global_history for b in range(batch)],
                                   d_model, gactive, device, window=None)
    medium_contexts = [mouts[b] if mouts[b] is not None
                       else local_hiddens[b].new_zeros(d_model) for b in range(batch)]
    global_contexts = [gouts[b] if gouts[b] is not None
                       else local_hiddens[b].new_zeros(d_model) for b in range(batch)]

    for b in range(batch):
        states[b].steps += 1
    fused = model.fusion(torch.stack([
        torch.cat([local_hiddens[b], medium_contexts[b], global_contexts[b], memory_reads[b]], dim=-1)
        for b in range(batch)
    ], dim=0))
    return fused


def encode_tbptt_chunk_batched(
    model: CompoundHierarchicalGPT,
    records: torch.Tensor,
    states: list[StreamState],
    *,
    reset_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[StreamState]]:
    """Lane-batched counterpart of encode_tbptt_chunk (same contract)."""
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

    per_step: list[torch.Tensor] = []
    for step in range(steps):
        per_step.append(advance_all_lanes(model, records[:, step], states))
    return torch.stack(per_step, dim=1), states


def tbptt_loss_batched(
    model: CompoundHierarchicalGPT,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    states: list[StreamState],
    *,
    reset_mask: torch.Tensor | None = None,
    event_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float], list[StreamState]]:
    contexts, states = encode_tbptt_chunk_batched(model, inputs, states, reset_mask=reset_mask)
    loss, parts = model.decoder.loss(contexts, targets, event_weight=event_weight)
    return loss, parts, states
