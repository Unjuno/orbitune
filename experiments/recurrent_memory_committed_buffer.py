from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import real_compound_memory_experiment_matched as memory  # noqa: E402


@dataclass
class CommittedMemoryState:
    bank_state: object | None
    hot_records: torch.Tensor | None
    cold_tokens: torch.Tensor | None
    committed_events: int = 0


def empty_state() -> CommittedMemoryState:
    return CommittedMemoryState(
        bank_state=None,
        hot_records=None,
        cold_tokens=None,
        committed_events=0,
    )


def push_event(
    model: memory.RoutedMultiBank,
    record: torch.Tensor,
    state: CommittedMemoryState,
    *,
    hot_window: int,
) -> CommittedMemoryState:
    """Move one event through exact-hot -> compressed-cold memory.

    ``record`` has shape ``[batch, 1, fields]``. The cold recurrent state only
    sees events that have fallen out of the bounded hot window. This creates a
    strict non-overlapping partition of history: compressed old events plus
    exact recent events.
    """

    if hot_window <= 0:
        raise ValueError("hot_window must be positive")
    if record.ndim != 3 or record.shape[1] != 1:
        raise ValueError("record must have shape [batch, 1, fields]")
    if state.hot_records is None:
        hot = record
    else:
        if state.hot_records.shape[0] != record.shape[0]:
            raise ValueError("batch size changed across committed-memory stream")
        hot = torch.cat([state.hot_records, record], dim=1)

    bank_state = state.bank_state
    cold_tokens = state.cold_tokens
    committed = state.committed_events
    if hot.shape[1] > hot_window:
        evicted = hot[:, :1]
        hot = hot[:, 1:]
        hidden = model.embedding(evicted)
        if bank_state is None:
            bank_state = (None, None, None)
        fast, fast_state = model.fast_memory.forward_chunk(hidden, bank_state[0])
        medium, medium_state = model.medium_memory.forward_chunk(hidden, bank_state[1])
        slow, slow_state = model.slow_memory.forward_chunk(hidden, bank_state[2])
        cold_tokens = torch.stack([fast[:, -1], medium[:, -1], slow[:, -1]], dim=1)
        bank_state = (fast_state, medium_state, slow_state)
        committed += 1

    return CommittedMemoryState(
        bank_state=bank_state,
        hot_records=hot,
        cold_tokens=cold_tokens,
        committed_events=committed,
    )


def push_chunk(
    model: memory.RoutedMultiBank,
    records: torch.Tensor,
    state: CommittedMemoryState,
    *,
    hot_window: int,
) -> CommittedMemoryState:
    if records.ndim != 3:
        raise ValueError("records must have shape [batch, time, fields]")
    next_state = state
    for index in range(records.shape[1]):
        next_state = push_event(
            model,
            records[:, index : index + 1],
            next_state,
            hot_window=hot_window,
        )
    return next_state
