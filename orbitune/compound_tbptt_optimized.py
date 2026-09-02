from __future__ import annotations

from collections import defaultdict

import torch

from orbitune.compound_base import CompoundHierarchicalGPT, StreamState, TransformerStack, _causal_bias


def _group_last_hidden(
    stack: TransformerStack,
    sequences: list[list[torch.Tensor]],
    *,
    device: torch.device,
    width: int,
) -> list[torch.Tensor]:
    """Evaluate variable-length hidden-state histories in length-matched batches.

    Grouping by exact sequence length preserves the same causal-position
    semantics as the legacy one-lane-at-a-time path while turning N tiny GPU
    launches into at most one launch per distinct history length.
    """
    outputs: list[torch.Tensor | None] = [None] * len(sequences)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, sequence in enumerate(sequences):
        if sequence:
            groups[len(sequence)].append(index)

    for length, indices in groups.items():
        batch = torch.stack([torch.stack(sequences[index], dim=0) for index in indices], dim=0)
        hidden = stack(batch, _causal_bias(length, device))[:, -1]
        for row, index in enumerate(indices):
            outputs[index] = hidden[row]

    return [
        value if value is not None else torch.zeros(width, device=device, dtype=next(stack.parameters()).dtype)
        for value in outputs
    ]


def _group_local_hidden(
    model: CompoundHierarchicalGPT,
    states: list[StreamState],
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor | None] = [None] * len(states)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        groups[len(state.local_records)].append(index)

    for length, indices in groups.items():
        raw = torch.stack(
            [torch.stack(states[index].local_records, dim=0) for index in indices],
            dim=0,
        )
        event = model.embedding(raw)
        hidden = model.local(
            event,
            _causal_bias(length, device, window=model.config.local_window),
        )[:, -1]
        for row, index in enumerate(indices):
            outputs[index] = hidden[row]

    if any(value is None for value in outputs):
        raise RuntimeError("local-state batching produced an empty lane")
    return [value for value in outputs if value is not None]


def _batched_memory_step(
    model: CompoundHierarchicalGPT,
    event_emb: torch.Tensor,
    states: list[StreamState],
) -> torch.Tensor:
    batch, width = event_emb.shape
    zeros = event_emb.new_zeros(1, width)
    banks: list[list[torch.Tensor]] = [[], [], []]
    for state in states:
        for bank in range(3):
            banks[bank].append(zeros if state.memory is None else state.memory[bank])
    packed = tuple(torch.cat(bank, dim=0) for bank in banks)
    read, next_memory = model.memory.step(event_emb, packed)  # type: ignore[arg-type]
    for lane in range(batch):
        states[lane].memory = tuple(value[lane : lane + 1] for value in next_memory)  # type: ignore[assignment]
    return read


def _advance_stream_batch_grad(
    model: CompoundHierarchicalGPT,
    records: torch.Tensor,
    states: list[StreamState],
) -> torch.Tensor:
    """Advance one TBPTT event for all lanes in one batched execution step.

    State transitions mirror ``compound_tbptt._advance_stream_grad``. The only
    intended difference is execution granularity: independent lanes that have
    the same history length share Transformer calls and all recurrent-memory
    lanes share one GRU step.
    """
    if records.ndim != 2 or records.shape[-1] != 12:
        raise ValueError("records must have shape [batch, 12]")
    if len(states) != records.shape[0]:
        raise ValueError("one StreamState is required per batch lane")

    device = next(model.parameters()).device
    raw = records.to(device=device, dtype=torch.long)
    batch = raw.shape[0]

    for lane in range(batch):
        states[lane].local_records.append(raw[lane].reshape(12))
        if len(states[lane].local_records) > model.config.local_window:
            states[lane].local_records.pop(0)

    local_hidden_list = _group_local_hidden(model, states, device=device)
    local_hidden = torch.stack(local_hidden_list, dim=0)

    event_emb = model.embedding(raw[:, None])[:, 0]
    memory_read = _batched_memory_step(model, event_emb, states)

    boundary_lanes: list[int] = []
    for lane, state in enumerate(states):
        state.medium_buffer.append(local_hidden[lane])
        if len(state.medium_buffer) >= model.config.medium_stride:
            summary = torch.stack(state.medium_buffer, dim=0).mean(dim=0)
            state.medium_buffer.clear()
            state.medium_history.append(summary)
            if len(state.medium_history) > model.config.medium_window:
                state.medium_history.pop(0)
            boundary_lanes.append(lane)

    if boundary_lanes:
        boundary_sequences = [states[lane].medium_history for lane in boundary_lanes]
        boundary_out = _group_last_hidden(
            model.medium,
            boundary_sequences,
            device=device,
            width=model.config.d_model,
        )
        for local_index, lane in enumerate(boundary_lanes):
            state = states[lane]
            state.global_buffer.append(boundary_out[local_index])
            if len(state.global_buffer) >= model.config.global_stride:
                global_summary = torch.stack(state.global_buffer, dim=0).mean(dim=0)
                state.global_buffer.clear()
                state.global_history.append(global_summary)
                if len(state.global_history) > model.config.global_window:
                    state.global_history.pop(0)

    medium_context = torch.stack(
        _group_last_hidden(
            model.medium,
            [state.medium_history for state in states],
            device=device,
            width=model.config.d_model,
        ),
        dim=0,
    )
    global_context = torch.stack(
        _group_last_hidden(
            model.global_stack,
            [state.global_history for state in states],
            device=device,
            width=model.config.d_model,
        ),
        dim=0,
    )

    for state in states:
        state.steps += 1

    return model.fusion(
        torch.cat([local_hidden, medium_context, global_context, memory_read], dim=-1)
    )


def encode_tbptt_chunk(
    model: CompoundHierarchicalGPT,
    records: torch.Tensor,
    states: list[StreamState],
    *,
    reset_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[StreamState]]:
    """Lane-batched generation-equivalent state-carry TBPTT encoder."""
    if records.ndim != 3 or records.shape[-1] != 12:
        raise ValueError("records must have shape [batch, time, 12]")
    batch, steps, _ = records.shape
    if len(states) != batch:
        raise ValueError("one StreamState is required per batch lane")
    if reset_mask is None:
        reset_mask = torch.zeros(batch, dtype=torch.bool, device=records.device)
    if reset_mask.numel() != batch:
        raise ValueError("reset_mask must have one entry per batch lane")

    resets = reset_mask.detach().to(device="cpu", dtype=torch.bool).tolist()
    for lane, reset in enumerate(resets):
        if reset:
            states[lane] = model.initial_stream_state()

    contexts = [
        _advance_stream_batch_grad(model, records[:, step], states)
        for step in range(steps)
    ]
    if not contexts:
        return records.new_empty((batch, 0, model.config.d_model), dtype=next(model.parameters()).dtype), states
    return torch.stack(contexts, dim=1), states


def tbptt_loss(
    model: CompoundHierarchicalGPT,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    states: list[StreamState],
    *,
    reset_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float], list[StreamState]]:
    contexts, states = encode_tbptt_chunk(model, inputs, states, reset_mask=reset_mask)
    loss, parts = model.decoder.loss(contexts, targets)
    return loss, parts, states
