"""Chunk-time-vectorized state-carry TBPTT (phase 2, branch perf/tbptt-chunkvec).

Reference semantics (orbitune.compound_tbptt) are preserved; only the
EXECUTION GRANULARITY changes:

- LOCAL (phase A): reference runs one Transformer forward per (lane, position)
  over that position's exact local window. This module materializes every
  position's exact window and evaluates them in one (A1) or a few bucketed
  (A2) Transformer calls, taking each window's last-position output.
- MEDIUM/GLOBAL (phases B/C): reference completes summaries sequentially and
  re-runs the hierarchy Transformer per position. This module computes the
  exact completion schedule first, then runs ONE causal (+sliding-window,
  eviction-equivalent) Transformer forward per hierarchy over
  carried_history + new_summaries, gathering each event's context at its
  visible prefix length. No event observes a not-yet-completed summary.
- MEMORY stays sequential (genuine recurrence), lane-batched as before.
- Decoder, sampler, StreamState layout, reset isolation, detach boundaries,
  save/load formats: unchanged.

Known regime note: with dropout > 0, batched/causal forwards consume
different mask streams than sequential calls (CASE B / STOCHASTIC_SEMANTICS
changed). With dropout disabled the computation is mathematically identical
(CASE A up to floating-point order); verified in hierarchy_parity.
"""
from __future__ import annotations

import torch

from orbitune.compound_base import CompoundHierarchicalGPT, StreamState
from orbitune.compound_tbptt_batched import _padded_causal_bias


def _causal_window_endpad_bias(lengths: torch.Tensor, lmax: int, device: torch.device,
                               *, window: int) -> torch.Tensor:
    """Bias [B,1,L,L] for left-aligned sequences with end padding.

    -inf for future keys, keys outside the sliding window, and pad keys.
    Pad QUERY rows are pointed at key 0 so no all--inf row produces NaN
    (their outputs are discarded by callers).
    """
    batch = lengths.numel()
    row = torch.arange(lmax, device=device)[:, None]
    col = torch.arange(lmax, device=device)[None, :]
    ok = (col <= row) & ((row - col) < window)
    valid_key = col < lengths.to(device)[None, :]
    ok = ok & valid_key
    bias = torch.zeros(batch, 1, lmax, lmax, device=device, dtype=torch.float32)
    bias = bias.masked_fill(~ok, float("-inf"))
    pad_query = torch.arange(lmax, device=device)[None, :] >= lengths.to(device)[:, None]
    fix = torch.full_like(bias, float("-inf"))
    fix[:, :, :, 0] = 0.0
    bias = torch.where(pad_query[:, None, :, None].expand_as(bias), fix, bias)
    return bias


def _local_windows_a1(model: CompoundHierarchicalGPT, records: torch.Tensor,
                      states: list[StreamState], device: torch.device,
                      ) -> tuple[torch.Tensor, list[int]]:
    """A1: all [B*T] exact windows front-padded to W in ONE embedding+forward."""
    batch, steps, _ = records.shape
    window = model.config.local_window
    rows: list[torch.Tensor] = []
    lengths: list[int] = []
    for b in range(batch):
        carried = states[b].local_records
        for t in range(steps):
            win = carried + [records[b, s].to(device=device, dtype=torch.long).reshape(12)
                             for s in range(t + 1)]
            win = win[-window:]
            lengths.append(len(win))
            rows.append(win)
    tmax = window
    pad = torch.zeros(len(rows), tmax, 12, device=device, dtype=torch.long)
    for i, win in enumerate(rows):
        stacked = torch.stack(win)
        pad[i, tmax - len(win):] = stacked
    bias = _padded_causal_bias(torch.tensor(lengths, dtype=torch.long), tmax,
                               device, window=window)
    hidden = model.local(model.embedding(pad), bias)[:, -1]
    return hidden.view(batch, steps, -1), lengths


def _local_windows_a2(model: CompoundHierarchicalGPT, records: torch.Tensor,
                      states: list[StreamState], device: torch.device,
                      ) -> tuple[torch.Tensor, dict[int, int]]:
    """A2: bucket (lane, position) rows by exact window length; one forward per bucket."""
    from orbitune.compound_base import _causal_bias

    batch, steps, _ = records.shape
    window = model.config.local_window
    buckets: dict[int, list[tuple[int, int, list[torch.Tensor]]]] = {}
    for b in range(batch):
        carried = states[b].local_records
        for t in range(steps):
            win = carried + [records[b, s].to(device=device, dtype=torch.long).reshape(12)
                             for s in range(t + 1)]
            win = win[-window:]
            buckets.setdefault(len(win), []).append((b, t, win))
    out = torch.empty(batch, steps, model.config.d_model, device=device,
                      dtype=next(model.parameters()).dtype)
    # NOTE: dtype above follows params (fp32); autocast casts inside modules.
    counts: dict[int, int] = {}
    for length, items in buckets.items():
        stacked = torch.stack([torch.stack(win) for _, _, win in items])
        bias = _causal_bias(length, device, window=window)
        hidden = model.local(model.embedding(stacked), bias)[:, -1]
        for (b, t, _), h in zip(items, hidden):
            out[b, t] = h
        counts[length] = len(items)
    return out, counts


def _summaries_from_buffer(buf: list[torch.Tensor], new_items: list[torch.Tensor],
                           stride: int,
                           new_positions: list[int] | None = None) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
    """Exact greedy grouping: returns (summaries, leftover, completion_positions).

    completion_positions are CHUNK positions (0-indexed into the current chunk)
    at which each summary completes, i.e. the first event that observes it.
    new_positions maps each entry of new_items to its chunk position; it is the
    identity when new_items are per-position hiddens (medium), and the medium
    completion positions when new_items are medium completion outputs (global).
    Groups fully inside the carried buffer cannot occur post-boundary
    (carried buffers are always shorter than stride), but are mapped safely.
    """
    full = list(buf) + list(new_items)
    total = len(full)
    n_complete = total // stride
    if new_positions is None:
        new_positions = list(range(len(new_items)))
    summaries: list[torch.Tensor] = []
    completions: list[int] = []
    for g in range(n_complete):
        group = full[g * stride:(g + 1) * stride]
        summaries.append(torch.stack(group).mean(dim=0))
        # chunk position of the last consumed item:
        full_idx = (g + 1) * stride - 1
        if full_idx < len(buf):
            pos = -1  # fully within carried buffer: visible from chunk start
        else:
            pos = new_positions[full_idx - len(buf)]
        completions.append(max(0, pos))
    leftover = full[n_complete * stride:]
    return summaries, leftover, completions


def _hierarchy_contexts(model: CompoundHierarchicalGPT, stack: torch.nn.Module,
                        carried: list[torch.Tensor],
                        new_summaries: list[torch.Tensor],
                        completions: list[int], steps: int,
                        zero_like: torch.Tensor, device: torch.device,
                        *, window: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """ONE causal (+sliding-window, eviction-equivalent) forward; gather per-event contexts.

    Returns (contexts [steps, D], completion_outputs [one per new summary]).
    Event t observes summaries with completion_position <= t (plus carried).
    """
    dtype = zero_like.dtype
    h0 = len(carried)
    seq = carried + new_summaries
    if not seq:
        return zero_like[None, :].expand(steps, -1).clone(), []
    padded = torch.stack(seq)[None].to(device=device, dtype=dtype)
    bias = _causal_window_endpad_bias(torch.tensor([len(seq)]), len(seq), device, window=window)
    out = stack(padded, bias)[0]
    comp_outs = [out[h0 + s] for s in range(len(new_summaries))]
    # visible summary count per event t (new only); context index into out:
    ctx = []
    for t in range(steps):
        s = 0
        for c in completions:
            if c <= t:
                s += 1
            else:
                break
        idx = h0 + s - 1
        if idx < 0:
            ctx.append(zero_like)
        else:
            ctx.append(out[idx])
    return torch.stack(ctx, dim=0), comp_outs


def encode_tbptt_chunkvec(
    model: CompoundHierarchicalGPT,
    records: torch.Tensor,
    states: list[StreamState],
    *,
    reset_mask: torch.Tensor | None = None,
    local_mode: str = "a1",
) -> tuple[torch.Tensor, list[StreamState]]:
    """Chunk-vectorized counterpart of encode_tbptt_chunk (same contract)."""
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
    d_model = model.config.d_model
    cfg = model.config

    # ---- Phase A: local hiddens for every (lane, position) ----
    if local_mode == "a1":
        local_h, _ = _local_windows_a1(model, records, states, device)
    elif local_mode == "a2":
        local_h, _ = _local_windows_a2(model, records, states, device)
    else:
        raise ValueError("local_mode must be 'a1' or 'a2'")
    # local_h: [B, T, D]

    # ---- Memory stays sequential (lane-batched per step, as before) ----
    event_embs = model.embedding(records.to(device=device))
    mem_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    have = [s.memory for s in states]
    if all(m is None for m in have):
        stacked_mem = None
    else:
        parts: list[list[torch.Tensor]] = [[], [], []]
        for mem in have:
            for k in range(3):
                parts[k].append(mem[k] if mem is not None
                                else event_embs.new_zeros(d_model))
        stacked_mem = (torch.stack(parts[0]), torch.stack(parts[1]), torch.stack(parts[2]))
    mem_reads = torch.empty(batch, steps, d_model, device=device,
                            dtype=event_embs.dtype)
    cur = stacked_mem
    for t in range(steps):
        read, cur = model.memory.step(event_embs[:, t], cur)
        mem_reads[:, t] = read
    assert cur is not None
    for b in range(batch):
        states[b].memory = (cur[0][b], cur[1][b], cur[2][b])

    # ---- Phase B: medium summaries + contexts ----
    # NOTE: reference appends local_hidden per position into medium_buffer;
    # leftover slices below reference the chunk's local_h rows (attached),
    # exactly like reference buffer tensors.
    med_ctx = torch.empty(batch, steps, d_model, device=device, dtype=local_h.dtype)
    med_comp_outs: list[list[torch.Tensor]] = []
    med_completions: list[list[int]] = []
    new_med_histories: list[list[torch.Tensor]] = []
    for b in range(batch):
        st = states[b]
        new_items = [local_h[b, t] for t in range(steps)]
        summaries, leftover, completions = _summaries_from_buffer(
            st.medium_buffer, new_items, cfg.medium_stride)
        st.medium_buffer = leftover
        med_completions.append(completions)
        zero = local_h[b, 0].new_zeros(d_model)
        ctx_b, comp_outs = _hierarchy_contexts(
            model, model.medium, st.medium_history, summaries, completions,
            steps, zero, device, window=cfg.medium_window)
        med_ctx[b] = ctx_b.to(med_ctx.dtype)
        med_comp_outs.append(comp_outs)
        new_history = (st.medium_history + summaries)[-cfg.medium_window:]
        new_med_histories.append(new_history)

    # ---- Phase C: global summaries + contexts ----
    glob_ctx = torch.empty(batch, steps, d_model, device=device, dtype=local_h.dtype)
    for b in range(batch):
        st = states[b]
        summaries, leftover, completions = _summaries_from_buffer(
            st.global_buffer, med_comp_outs[b], cfg.global_stride,
            new_positions=med_completions[b])
        st.global_buffer = leftover
        zero = local_h[b, 0].new_zeros(d_model)
        ctx_b, _ = _hierarchy_contexts(
            model, model.global_stack, st.global_history, summaries, completions,
            steps, zero, device, window=cfg.global_window)
        glob_ctx[b] = ctx_b.to(glob_ctx.dtype)
        st.global_history = (st.global_history + summaries)[-cfg.global_window:]

    # ---- fusion (single call) + state bookkeeping ----
    fused = model.fusion(torch.cat([local_h, med_ctx, glob_ctx, mem_reads], dim=-1))
    for b in range(batch):
        st = states[b]
        st.medium_history = new_med_histories[b]
        carried = st.local_records
        new_recs = [records[b, t].to(device=device, dtype=torch.long).reshape(12)
                    for t in range(steps)]
        st.local_records = (carried + new_recs)[-cfg.local_window:]
        st.steps += steps
    return fused, states


def tbptt_loss_chunkvec(
    model: CompoundHierarchicalGPT,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    states: list[StreamState],
    *,
    reset_mask: torch.Tensor | None = None,
    event_weight: torch.Tensor | None = None,
    local_mode: str = "a1",
) -> tuple[torch.Tensor, dict[str, float], list[StreamState]]:
    contexts, states = encode_tbptt_chunkvec(model, inputs, states,
                                             reset_mask=reset_mask,
                                             local_mode=local_mode)
    loss, parts = model.decoder.loss(contexts, targets, event_weight=event_weight)
    return loss, parts, states
