from __future__ import annotations

from collections import defaultdict

import torch

from orbitune.compound_base import CompoundHierarchicalGPT, StreamState, TransformerStack, _causal_bias


def _zero_hidden(stack: TransformerStack, width: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(width, device=device, dtype=next(stack.parameters()).dtype)


def _group_hidden_last(
    stack: TransformerStack,
    sequences: list[list[torch.Tensor]],
    *,
    device: torch.device,
    width: int,
) -> list[torch.Tensor]:
    """Evaluate independent hidden-state histories in length-matched batches.

    Every requested history is still evaluated independently.  We only move the
    lane/time dimension into the Transformer batch dimension, so training-time
    dropout keeps one stochastic evaluation per request instead of reusing a
    cached value across events.
    """
    outputs: list[torch.Tensor | None] = [None] * len(sequences)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, sequence in enumerate(sequences):
        if sequence:
            groups[len(sequence)].append(index)

    for length, indices in groups.items():
        batch = torch.stack(
            [torch.stack(sequences[index], dim=0) for index in indices],
            dim=0,
        )
        hidden = stack(batch, _causal_bias(length, device))[:, -1]
        for row, index in enumerate(indices):
            outputs[index] = hidden[row]

    zero = _zero_hidden(stack, width, device)
    return [zero if value is None else value for value in outputs]


def _group_local_last(
    model: CompoundHierarchicalGPT,
    windows: list[list[torch.Tensor]],
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor | None] = [None] * len(windows)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, window in enumerate(windows):
        if not window:
            raise RuntimeError("local window cannot be empty")
        groups[len(window)].append(index)

    for length, indices in groups.items():
        records = torch.stack(
            [torch.stack(windows[index], dim=0) for index in indices],
            dim=0,
        )
        event = model.embedding(records)
        hidden = model.local(
            event,
            _causal_bias(length, device, window=model.config.local_window),
        )[:, -1]
        for row, index in enumerate(indices):
            outputs[index] = hidden[row]

    if any(value is None for value in outputs):
        raise RuntimeError("local time batching produced an empty result")
    return [value for value in outputs if value is not None]


def _local_contexts(
    model: CompoundHierarchicalGPT,
    raw: torch.Tensor,
    states: list[StreamState],
) -> torch.Tensor:
    """Compute all local per-event windows with time folded into batch."""
    batch, steps, _ = raw.shape
    device = raw.device
    windows: list[list[torch.Tensor]] = []
    final_histories: list[list[torch.Tensor]] = []

    for lane in range(batch):
        history = list(states[lane].local_records)
        for step in range(steps):
            history.append(raw[lane, step].reshape(12))
            if len(history) > model.config.local_window:
                history.pop(0)
            windows.append(list(history))
        final_histories.append(history)

    values = _group_local_last(model, windows, device=device)
    for lane, history in enumerate(final_histories):
        states[lane].local_records = history
    return torch.stack(values, dim=0).view(batch, steps, model.config.d_model)


def _pack_memory_state(
    event_emb: torch.Tensor,
    states: list[StreamState],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, width = event_emb.shape
    zeros = event_emb.new_zeros(1, width)
    banks: list[list[torch.Tensor]] = [[], [], []]
    for state in states:
        for bank in range(3):
            banks[bank].append(zeros if state.memory is None else state.memory[bank])
    return tuple(torch.cat(bank, dim=0) for bank in banks)  # type: ignore[return-value]


def _memory_contexts(
    model: CompoundHierarchicalGPT,
    raw: torch.Tensor,
    states: list[StreamState],
) -> torch.Tensor:
    """Keep the true recurrent dependency in time, but batch all lanes."""
    batch, steps, _ = raw.shape
    event = model.embedding(raw)
    packed = _pack_memory_state(event[:, 0], states)
    outputs: list[torch.Tensor] = []
    next_memory = packed
    for step in range(steps):
        read, next_memory = model.memory.step(event[:, step], next_memory)
        outputs.append(read)

    for lane in range(batch):
        states[lane].memory = tuple(
            value[lane : lane + 1] for value in next_memory
        )  # type: ignore[assignment]
    return torch.stack(outputs, dim=1)


def _medium_contexts(
    model: CompoundHierarchicalGPT,
    local_hidden: torch.Tensor,
    states: list[StreamState],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Update medium state, then batch all per-event Transformer evaluations.

    Legacy streaming evaluates ``medium`` twice at a medium boundary: once for
    the value entering the global buffer and once for the current medium
    context.  We deliberately keep those as two separate batched evaluations,
    which preserves the training-time stochastic structure when dropout > 0.
    """
    batch, steps, width = local_hidden.shape
    device = local_hidden.device
    snapshots: list[list[torch.Tensor]] = []
    boundary_snapshots: list[list[torch.Tensor]] = []
    boundary_indices: list[int] = []

    for lane in range(batch):
        state = states[lane]
        for step in range(steps):
            state.medium_buffer.append(local_hidden[lane, step])
            boundary = len(state.medium_buffer) >= model.config.medium_stride
            if boundary:
                summary = torch.stack(state.medium_buffer, dim=0).mean(dim=0)
                state.medium_buffer.clear()
                state.medium_history.append(summary)
                if len(state.medium_history) > model.config.medium_window:
                    state.medium_history.pop(0)
                boundary_snapshots.append(list(state.medium_history))
                boundary_indices.append(lane * steps + step)
            snapshots.append(list(state.medium_history))

    context_values = _group_hidden_last(
        model.medium,
        snapshots,
        device=device,
        width=width,
    )
    boundary_values = _group_hidden_last(
        model.medium,
        boundary_snapshots,
        device=device,
        width=width,
    )
    boundary_map = {
        flat_index: value
        for flat_index, value in zip(boundary_indices, boundary_values)
    }
    return (
        torch.stack(context_values, dim=0).view(batch, steps, width),
        boundary_map,
    )


def _global_contexts(
    model: CompoundHierarchicalGPT,
    *,
    batch: int,
    steps: int,
    boundary_medium: dict[int, torch.Tensor],
    states: list[StreamState],
    device: torch.device,
) -> torch.Tensor:
    snapshots: list[list[torch.Tensor]] = []

    for lane in range(batch):
        state = states[lane]
        for step in range(steps):
            flat_index = lane * steps + step
            medium_value = boundary_medium.get(flat_index)
            if medium_value is not None:
                state.global_buffer.append(medium_value)
                if len(state.global_buffer) >= model.config.global_stride:
                    summary = torch.stack(state.global_buffer, dim=0).mean(dim=0)
                    state.global_buffer.clear()
                    state.global_history.append(summary)
                    if len(state.global_history) > model.config.global_window:
                        state.global_history.pop(0)
            snapshots.append(list(state.global_history))

    values = _group_hidden_last(
        model.global_stack,
        snapshots,
        device=device,
        width=model.config.d_model,
    )
    return torch.stack(values, dim=0).view(batch, steps, model.config.d_model)


def encode_tbptt_chunk(
    model: CompoundHierarchicalGPT,
    records: torch.Tensor,
    states: list[StreamState],
    *,
    reset_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[StreamState]]:
    """Generation-equivalent state carry with Transformer work batched in time.

    The Local/Medium/Global stacks see the same logical histories as the legacy
    streaming path.  Independent (lane, time) evaluations with the same history
    length are executed as one Transformer batch.  Only recurrent memory keeps
    a true event-by-event loop.
    """
    if records.ndim != 3 or records.shape[-1] != 12:
        raise ValueError("records must have shape [batch, time, 12]")
    batch, steps, _ = records.shape
    if steps <= 0:
        raise ValueError("TBPTT chunk must contain at least one event")
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

    device = next(model.parameters()).device
    raw = records.to(device=device, dtype=torch.long)

    local_hidden = _local_contexts(model, raw, states)
    memory_context = _memory_contexts(model, raw, states)
    medium_context, boundary_medium = _medium_contexts(model, local_hidden, states)
    global_context = _global_contexts(
        model,
        batch=batch,
        steps=steps,
        boundary_medium=boundary_medium,
        states=states,
        device=device,
    )

    for state in states:
        state.steps += steps

    fused = model.fusion(
        torch.cat(
            [local_hidden, medium_context, global_context, memory_context],
            dim=-1,
        )
    )
    return fused, states


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
