from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import random

import torch

from orbitune.compound_base import CompoundHierarchicalGPT, StreamState, _causal_bias
from orbitune.compound_training import CompoundSong


@dataclass(slots=True)
class ChunkBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    reset_mask: torch.Tensor
    song_indices: tuple[int, ...]
    offsets: tuple[int, ...]


class SequentialSongChunkSampler:
    """Song-local sequential sampler for state-carry TBPTT.

    Each batch lane advances monotonically through one song in fixed-size
    chunks. When fewer than ``seq_len + 1`` records remain, that lane starts a
    newly sampled song at offset zero and marks the lane in ``reset_mask``.
    Windows never cross song boundaries. The lane positions are serializable
    so an exact checkpoint/resume can continue from the same next chunks.
    """

    def __init__(
        self,
        songs: list[CompoundSong],
        *,
        batch_size: int,
        seq_len: int,
        rng: random.Random,
    ) -> None:
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        self.songs = songs
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.rng = rng
        self.eligible = [i for i, song in enumerate(songs) if len(song.records) >= seq_len + 1]
        if not self.eligible:
            raise ValueError("no song is long enough for the requested seq_len")
        self.song_indices = [-1] * self.batch_size
        self.offsets = [0] * self.batch_size

    def _start_lane(self, lane: int) -> None:
        self.song_indices[lane] = self.rng.choice(self.eligible)
        self.offsets[lane] = 0

    def sample(self, device: str | torch.device) -> ChunkBatch:
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        resets: list[bool] = []
        starts: list[int] = []
        songs_used: list[int] = []
        for lane in range(self.batch_size):
            song_index = self.song_indices[lane]
            reset = song_index < 0
            if not reset:
                song = self.songs[song_index]
                if self.offsets[lane] + self.seq_len >= len(song.records):
                    reset = True
            if reset:
                self._start_lane(lane)
                song_index = self.song_indices[lane]
            song = self.songs[song_index]
            start = self.offsets[lane]
            window = song.records[start : start + self.seq_len + 1]
            if len(window) != self.seq_len + 1:
                raise RuntimeError("sequential sampler produced a short chunk")
            tensor = torch.tensor(window, dtype=torch.long)
            xs.append(tensor[:-1])
            ys.append(tensor[1:])
            resets.append(reset)
            starts.append(start)
            songs_used.append(song_index)
            self.offsets[lane] += self.seq_len
        return ChunkBatch(
            inputs=torch.stack(xs).to(device),
            targets=torch.stack(ys).to(device),
            reset_mask=torch.tensor(resets, dtype=torch.bool, device=device),
            song_indices=tuple(songs_used),
            offsets=tuple(starts),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "song_indices": list(self.song_indices),
            "offsets": list(self.offsets),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("batch_size", -1)) != self.batch_size:
            raise ValueError("TBPTT sampler batch_size mismatch")
        if int(state.get("seq_len", -1)) != self.seq_len:
            raise ValueError("TBPTT sampler seq_len mismatch")
        song_indices = [int(v) for v in state.get("song_indices", [])]
        offsets = [int(v) for v in state.get("offsets", [])]
        if len(song_indices) != self.batch_size or len(offsets) != self.batch_size:
            raise ValueError("TBPTT sampler lane-state length mismatch")
        for song_index, offset in zip(song_indices, offsets):
            if song_index != -1 and song_index not in self.eligible:
                raise ValueError(f"invalid TBPTT sampler song index {song_index}")
            if offset < 0:
                raise ValueError("TBPTT sampler offset must be non-negative")
        self.song_indices = song_indices
        self.offsets = offsets


def initial_batch_stream_states(model: CompoundHierarchicalGPT, batch_size: int) -> list[StreamState]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [model.initial_stream_state() for _ in range(batch_size)]


def detach_stream_state(state: StreamState) -> StreamState:
    """Detach every differentiable tensor carried across a TBPTT boundary."""
    return StreamState(
        local_records=[tensor.detach() for tensor in state.local_records],
        medium_buffer=[tensor.detach() for tensor in state.medium_buffer],
        medium_history=[tensor.detach() for tensor in state.medium_history],
        global_buffer=[tensor.detach() for tensor in state.global_buffer],
        global_history=[tensor.detach() for tensor in state.global_history],
        memory=None if state.memory is None else tuple(tensor.detach() for tensor in state.memory),
        steps=int(state.steps),
    )


def detach_batch_stream_states(states: list[StreamState]) -> list[StreamState]:
    return [detach_stream_state(state) for state in states]


def _advance_stream_grad(
    model: CompoundHierarchicalGPT,
    record: torch.Tensor,
    state: StreamState,
) -> torch.Tensor:
    """Differentiable counterpart of ``CompoundHierarchicalGPT.advance_stream``.

    The value semantics intentionally mirror generation: local raw-record
    history, completed medium/global summaries and recurrent fast/medium/slow
    memory are carried continuously. Unlike generation, tensors are *not*
    detached inside the current chunk. The caller detaches the complete state
    once per TBPTT boundary.
    """
    device = next(model.parameters()).device
    raw = record.to(device=device, dtype=torch.long).reshape(12)
    state.local_records.append(raw)
    if len(state.local_records) > model.config.local_window:
        state.local_records.pop(0)
    local_records = torch.stack(state.local_records)[None]
    local_event = model.embedding(local_records)
    local_hidden = model.local(
        local_event,
        _causal_bias(local_event.shape[1], device, window=model.config.local_window),
    )[0, -1]

    event_emb = model.embedding(raw[None, None])[0, 0]
    memory_read, state.memory = model.memory.step(event_emb[None], state.memory)
    memory_read = memory_read[0]

    state.medium_buffer.append(local_hidden)
    if len(state.medium_buffer) >= model.config.medium_stride:
        summary = torch.stack(state.medium_buffer).mean(dim=0)
        state.medium_buffer.clear()
        state.medium_history.append(summary)
        if len(state.medium_history) > model.config.medium_window:
            state.medium_history.pop(0)
        medium_sequence = torch.stack(state.medium_history)[None]
        medium_out = model.medium(
            medium_sequence,
            _causal_bias(medium_sequence.shape[1], device),
        )[0, -1]
        state.global_buffer.append(medium_out)
        if len(state.global_buffer) >= model.config.global_stride:
            global_summary = torch.stack(state.global_buffer).mean(dim=0)
            state.global_buffer.clear()
            state.global_history.append(global_summary)
            if len(state.global_history) > model.config.global_window:
                state.global_history.pop(0)

    medium_context = local_hidden.new_zeros(local_hidden.shape)
    if state.medium_history:
        medium_sequence = torch.stack(state.medium_history)[None]
        medium_context = model.medium(
            medium_sequence,
            _causal_bias(medium_sequence.shape[1], device),
        )[0, -1]

    global_context = local_hidden.new_zeros(local_hidden.shape)
    if state.global_history:
        global_sequence = torch.stack(state.global_history)[None]
        global_context = model.global_stack(
            global_sequence,
            _causal_bias(global_sequence.shape[1], device),
        )[0, -1]

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
    """Encode a song-sequential chunk while carrying generation-equivalent state.

    ``records`` has shape ``[batch, time, 12]``. State is independent per batch
    lane. A true entry in ``reset_mask`` resets only that lane at the song
    boundary. Returned state remains attached to the current chunk graph; call
    :func:`detach_batch_stream_states` after ``backward``/``optimizer.step``.
    """
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
        lane_contexts = [
            _advance_stream_grad(model, records[lane, step], states[lane])
            for lane in range(batch)
        ]
        per_step.append(torch.stack(lane_contexts, dim=0))
    return torch.stack(per_step, dim=1), states


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


def stream_state_to_cpu(state: StreamState) -> dict[str, Any]:
    def cpu_list(values: list[torch.Tensor]) -> list[torch.Tensor]:
        return [value.detach().cpu() for value in values]

    return {
        "local_records": cpu_list(state.local_records),
        "medium_buffer": cpu_list(state.medium_buffer),
        "medium_history": cpu_list(state.medium_history),
        "global_buffer": cpu_list(state.global_buffer),
        "global_history": cpu_list(state.global_history),
        "memory": None if state.memory is None else [value.detach().cpu() for value in state.memory],
        "steps": int(state.steps),
    }


def stream_state_from_cpu(payload: dict[str, Any], device: str | torch.device) -> StreamState:
    device = torch.device(device)

    def move(values: Any) -> list[torch.Tensor]:
        if values is None:
            return []
        return [value.to(device) for value in values]

    memory_payload = payload.get("memory")
    memory = None if memory_payload is None else tuple(value.to(device) for value in memory_payload)
    return StreamState(
        local_records=move(payload.get("local_records")),
        medium_buffer=move(payload.get("medium_buffer")),
        medium_history=move(payload.get("medium_history")),
        global_buffer=move(payload.get("global_buffer")),
        global_history=move(payload.get("global_history")),
        memory=memory,  # type: ignore[arg-type]
        steps=int(payload.get("steps", 0)),
    )


def batch_stream_states_to_cpu(states: list[StreamState]) -> list[dict[str, Any]]:
    return [stream_state_to_cpu(state) for state in states]


def batch_stream_states_from_cpu(
    payload: list[dict[str, Any]],
    device: str | torch.device,
) -> list[StreamState]:
    return [stream_state_from_cpu(state, device) for state in payload]
