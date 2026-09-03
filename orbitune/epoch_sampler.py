"""Epoch-aware, deterministic, no-replacement TBPTT sampler.

The production commercial-base pretrain must (1) train on every song
with at least one next-event target exactly once per epoch with no
replacement, (2) deterministically shuffle the per-epoch song order
from a stable per-epoch RNG state, (3) pad the final partial chunk to
``seq_len + 1`` with ``event_weight = 0`` so padding is lossless,
(4) keep idle lanes at the epoch tail at ``event_weight = 0``, (5) not
prefetch the next epoch, and (6) round-trip a complete ``state_dict``
so an exact checkpoint/resume can continue from the same next chunks.
The sampler is the source of truth for ``epoch_events_seen`` and
``epoch_events_total``; both count real next-event pairs, independent
of padding, batch size, sequence length, or loss weighting.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from orbitune.compound_indexed import IndexedCompoundSong
from orbitune.compound_tbptt import ChunkBatch


SAMPLER_SCHEMA = "orbitune-epoch-sampler-v2"


def _corpus_identity(songs: list[IndexedCompoundSong]) -> str:
    """Stable identity for sampler-visible corpus order, content and weights.

    Mirrors the production-mode identity used by :mod:`orbitune.indexed_sampling`
    so a checkpoint cannot be silently resumed against a different corpus
    (fail-closed).
    """
    digest = hashlib.sha256()
    for index, song in enumerate(songs):
        values = (
            index,
            song.sha256,
            song.composition_fingerprint,
            len(song.records),
            repr(float(song.quality_weight)),
            repr(float(song.sampling_weight)),
            int(song.tracks),
            song.source_id,
            song.license,
        )
        for value in values:
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


@dataclass(slots=True)
class EpochSample:
    """One optimizer-step's data and per-event loss weights.

    ``event_weight`` has shape ``(batch_size, seq_len)``. Real
    next-event pairs receive either ``1.0`` (unweighted mode) or the
    source song's manifest ``sampling_weight`` (weighted mode).
    Padding and idle lanes receive ``0.0``. ``events_counted`` counts
    real next-event pairs, not the sum of loss weights.
    """

    batch: ChunkBatch
    event_weight: torch.Tensor
    events_counted: int


class EpochAwareNoReplacementSampler:
    """No-replacement epoch-aware TBPTT sampler.

    Each epoch visits every song with at least two Compound records
    exactly once. A song may be shorter than ``seq_len + 1``; in that
    case its only chunk is a padded partial chunk and every real
    ``len(song) - 1`` target pair still participates exactly once.

    Per-epoch song order is a deterministic shuffle driven by the
    sampler RNG. Within a song, offsets advance monotonically in
    ``seq_len`` chunks. Idle lanes at the epoch tail receive a fully
    padded chunk with zero loss weight, and the next epoch is never
    prefetched into the current epoch.

    The ``state_dict`` is the source of truth for exact resume:
    ``corpus_identity / epoch_index / epoch_seed / shuffled_song_order /
    order_cursor / lane song_indices / lane offsets / lane tail state /
    epoch_events_seen / epoch_events_total / batch_size / seq_len /
    weighting mode``. Loading is fail-closed on incompatible state.
    """

    def __init__(
        self,
        songs: list[IndexedCompoundSong],
        *,
        batch_size: int,
        seq_len: int,
        epoch_seed: int,
        weighted: bool = False,
        rng: random.Random | None = None,
    ) -> None:
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        self.songs = songs
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.epoch_seed = int(epoch_seed)
        self.weighted = bool(weighted)
        self.rng = rng if rng is not None else random.Random(epoch_seed)
        self.corpus_identity = _corpus_identity(songs)

        # Any song with at least one next-event pair is trainable. Short songs
        # are padded instead of being silently dropped based on seq_len.
        self.eligible = [i for i, song in enumerate(songs) if len(song.records) >= 2]
        if not self.eligible:
            raise ValueError("no song contains a next-event training pair")
        self._eligible_position = {corpus_index: local for local, corpus_index in enumerate(self.eligible)}

        if self.weighted:
            for corpus_index in self.eligible:
                value = float(self.songs[corpus_index].sampling_weight)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"invalid sampling_weight for corpus song {corpus_index}: {value!r}"
                    )

        # Number of complete seq_len chunks and real-event tail size. No
        # max(1, ...) here: a song shorter than seq_len has zero complete
        # chunks and one padded tail.
        self._chunks_per_song: list[int] = [
            (len(songs[i].records) - 1) // self.seq_len for i in self.eligible
        ]
        self._tail_per_song: list[int] = [
            (len(songs[i].records) - 1) % self.seq_len for i in self.eligible
        ]

        self.epoch_index: int = 0
        self.shuffled_song_order: list[int] = []
        self.order_cursor: int = 0
        self.lane_song_indices: list[int] = [-1] * self.batch_size
        self.lane_offsets: list[int] = [0] * self.batch_size
        self.lane_tail_left: list[int] = [0] * self.batch_size
        self.epoch_events_seen: int = 0
        self.epoch_events_total: int = 0
        self._begin_epoch()

    # --- epoch management ------------------------------------------------

    def _begin_epoch(self) -> None:
        local_order = list(range(len(self.eligible)))
        self.rng.shuffle(local_order)
        self.shuffled_song_order = [self.eligible[local] for local in local_order]
        self.order_cursor = 0
        self.lane_song_indices = [-1] * self.batch_size
        self.lane_offsets = [0] * self.batch_size
        self.lane_tail_left = [0] * self.batch_size
        self.epoch_events_seen = 0

        # This is exactly sum(len(song)-1) across trainable songs and therefore
        # must be invariant to seq_len, batch size, shuffle order, and weights.
        self.epoch_events_total = sum(
            self._chunks_per_song[local] * self.seq_len + self._tail_per_song[local]
            for local in local_order
        )

    def _start_lane(self, lane: int) -> None:
        if self.order_cursor >= len(self.shuffled_song_order):
            self.lane_song_indices[lane] = -1
            self.lane_offsets[lane] = 0
            self.lane_tail_left[lane] = 0
            return
        corpus_index = self.shuffled_song_order[self.order_cursor]
        self.order_cursor += 1
        self.lane_song_indices[lane] = corpus_index
        self.lane_offsets[lane] = 0
        local_pos = self._eligible_position[corpus_index]
        self.lane_tail_left[lane] = self._tail_per_song[local_pos]

    def _lane_complete_chunks_left(self, lane: int) -> int:
        corpus_index = self.lane_song_indices[lane]
        if corpus_index < 0:
            return 0
        local_pos = self._eligible_position[corpus_index]
        chunks_total = self._chunks_per_song[local_pos]
        consumed = self.lane_offsets[lane] // self.seq_len
        return max(0, chunks_total - consumed)

    def _song_loss_weight(self, corpus_index: int) -> float:
        if not self.weighted:
            return 1.0
        return float(self.songs[corpus_index].sampling_weight)

    def _idle_window(self) -> np.ndarray:
        # ``eligible`` is non-empty and every eligible song has >=2 records.
        song = self.songs[self.eligible[0]]
        last_record_row = np.asarray(song.records[-1]).reshape(1, -1)
        return np.tile(last_record_row, (self.seq_len + 1, 1)).astype(np.int64)

    # --- public API ------------------------------------------------------

    @property
    def is_epoch_complete(self) -> bool:
        return self.epoch_events_seen >= self.epoch_events_total

    def sample(self, device: str | torch.device) -> EpochSample:
        """Emit one optimizer step's chunk + per-event weight tensor."""
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        resets: list[bool] = []
        starts: list[int] = []
        songs_used: list[int] = []
        event_weights: list[list[float]] = []
        active_count = 0

        for lane in range(self.batch_size):
            corpus_index = self.lane_song_indices[lane]
            reset = corpus_index < 0
            if (
                not reset
                and self._lane_complete_chunks_left(lane) <= 0
                and self.lane_tail_left[lane] <= 0
            ):
                reset = True
            if reset:
                self._start_lane(lane)
                corpus_index = self.lane_song_indices[lane]
                reset = True

            if corpus_index < 0:
                pad_window = self._idle_window()
                xs.append(torch.from_numpy(pad_window[:-1]))
                ys.append(torch.from_numpy(pad_window[1:]))
                resets.append(True)
                starts.append(0)
                songs_used.append(-1)
                event_weights.append([0.0] * self.seq_len)
                continue

            song = self.songs[corpus_index]
            start = self.lane_offsets[lane]
            chunks_left = self._lane_complete_chunks_left(lane)
            tail_left = self.lane_tail_left[lane]
            loss_weight = self._song_loss_weight(corpus_index)

            if chunks_left > 0:
                window = np.asarray(
                    song.records[start : start + self.seq_len + 1], dtype=np.int64
                )
                if window.shape[0] != self.seq_len + 1:
                    raise RuntimeError("epoch sampler produced a short chunk")
                weight = [loss_weight] * self.seq_len
                real_events = self.seq_len
                self.lane_offsets[lane] += self.seq_len
                resets.append(reset)
            elif tail_left > 0:
                full_end = start + self.seq_len + 1
                records = np.asarray(song.records[start : min(full_end, len(song.records))], dtype=np.int64)
                if records.shape[0] < self.seq_len + 1:
                    pad_needed = self.seq_len + 1 - records.shape[0]
                    last_row = np.asarray(song.records[-1]).reshape(1, -1)
                    pad = np.tile(last_row, (pad_needed, 1))
                    window = np.concatenate([records, pad]).astype(np.int64)
                else:
                    window = records
                if window.shape[0] != self.seq_len + 1:
                    raise RuntimeError("epoch sampler produced a short tail chunk")
                weight = [loss_weight] * tail_left + [0.0] * (self.seq_len - tail_left)
                real_events = tail_left
                self.lane_offsets[lane] += self.seq_len
                self.lane_tail_left[lane] = 0
                # A song shorter than seq_len starts directly with this tail;
                # preserve the song-boundary reset in that case.
                resets.append(reset)
            else:
                # Defensive fallback. Normal control flow resets exhausted lanes
                # at the top of the next call.
                self._start_lane(lane)
                corpus_index = self.lane_song_indices[lane]
                if corpus_index < 0:
                    pad_window = self._idle_window()
                    xs.append(torch.from_numpy(pad_window[:-1]))
                    ys.append(torch.from_numpy(pad_window[1:]))
                    resets.append(True)
                    starts.append(0)
                    songs_used.append(-1)
                    event_weights.append([0.0] * self.seq_len)
                    continue
                song = self.songs[corpus_index]
                start = 0
                local_pos = self._eligible_position[corpus_index]
                chunks_left = self._chunks_per_song[local_pos]
                tail_left = self._tail_per_song[local_pos]
                loss_weight = self._song_loss_weight(corpus_index)
                if chunks_left > 0:
                    window = np.asarray(song.records[: self.seq_len + 1], dtype=np.int64)
                    if window.shape[0] != self.seq_len + 1:
                        raise RuntimeError("epoch sampler produced a short chunk")
                    weight = [loss_weight] * self.seq_len
                    real_events = self.seq_len
                    self.lane_offsets[lane] = self.seq_len
                elif tail_left > 0:
                    records = np.asarray(song.records[:], dtype=np.int64)
                    pad_needed = self.seq_len + 1 - records.shape[0]
                    last_row = np.asarray(song.records[-1]).reshape(1, -1)
                    pad = np.tile(last_row, (pad_needed, 1))
                    window = np.concatenate([records, pad]).astype(np.int64)
                    weight = [loss_weight] * tail_left + [0.0] * (self.seq_len - tail_left)
                    real_events = tail_left
                    self.lane_offsets[lane] = self.seq_len
                    self.lane_tail_left[lane] = 0
                else:
                    raise RuntimeError("eligible song has no next-event pairs")
                resets.append(True)

            starts.append(start)
            songs_used.append(corpus_index)
            xs.append(torch.from_numpy(window[:-1]))
            ys.append(torch.from_numpy(window[1:]))
            event_weights.append(weight)
            active_count += real_events

        if self.epoch_events_seen + active_count > self.epoch_events_total:
            raise RuntimeError("epoch sampler emitted more real events than epoch_events_total")
        self.epoch_events_seen += active_count

        ew_tensor = torch.tensor(event_weights, dtype=torch.float32, device=device)
        return EpochSample(
            batch=ChunkBatch(
                inputs=torch.stack(xs).to(device),
                targets=torch.stack(ys).to(device),
                reset_mask=torch.tensor(resets, dtype=torch.bool, device=device),
                song_indices=tuple(songs_used),
                offsets=tuple(starts),
            ),
            event_weight=ew_tensor,
            events_counted=active_count,
        )

    def advance_epoch(self) -> None:
        """Move to the next epoch with a fresh deterministic shuffle."""
        self.epoch_index += 1
        self._begin_epoch()

    # --- state round-trip ------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": SAMPLER_SCHEMA,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "weighted": self.weighted,
            "corpus_identity": self.corpus_identity,
            "epoch_index": self.epoch_index,
            "epoch_seed": self.epoch_seed,
            "shuffled_song_order": list(self.shuffled_song_order),
            "order_cursor": self.order_cursor,
            "lane_song_indices": list(self.lane_song_indices),
            "lane_offsets": list(self.lane_offsets),
            "lane_tail_left": list(self.lane_tail_left),
            "epoch_events_seen": self.epoch_events_seen,
            "epoch_events_total": self.epoch_events_total,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if str(state.get("schema", "")) != SAMPLER_SCHEMA:
            raise ValueError(f"unknown epoch-sampler schema: {state.get('schema')!r}")
        if int(state.get("batch_size", -1)) != self.batch_size:
            raise ValueError("epoch sampler batch_size mismatch")
        if int(state.get("seq_len", -1)) != self.seq_len:
            raise ValueError("epoch sampler seq_len mismatch")
        if bool(state.get("weighted", False)) != self.weighted:
            raise ValueError("epoch sampler weighted-mode mismatch")
        if str(state.get("corpus_identity", "")) != self.corpus_identity:
            raise ValueError("epoch sampler corpus identity mismatch")
        order = [int(v) for v in state.get("shuffled_song_order", [])]
        if set(order) != set(self.eligible) or len(order) != len(self.eligible):
            raise ValueError("epoch sampler shuffled order does not match eligible set")
        self.epoch_index = int(state.get("epoch_index", 0))
        self.epoch_seed = int(state.get("epoch_seed", self.epoch_seed))
        self.shuffled_song_order = order
        self.order_cursor = int(state.get("order_cursor", 0))
        self.lane_song_indices = [int(v) for v in state.get("lane_song_indices", [])]
        self.lane_offsets = [int(v) for v in state.get("lane_offsets", [])]
        self.lane_tail_left = [int(v) for v in state.get("lane_tail_left", [])]
        if (
            len(self.lane_song_indices) != self.batch_size
            or len(self.lane_offsets) != self.batch_size
            or len(self.lane_tail_left) != self.batch_size
        ):
            raise ValueError("epoch sampler lane-state length mismatch")
        if not 0 <= self.order_cursor <= len(self.shuffled_song_order):
            raise ValueError("epoch sampler order_cursor out of range")
        for corpus_index, offset, tail_left in zip(
            self.lane_song_indices, self.lane_offsets, self.lane_tail_left
        ):
            if corpus_index != -1 and corpus_index not in self._eligible_position:
                raise ValueError(f"invalid epoch sampler song index {corpus_index}")
            if offset < 0 or tail_left < 0:
                raise ValueError("epoch sampler lane offset/tail must be non-negative")
        self.epoch_events_seen = int(state.get("epoch_events_seen", 0))
        self.epoch_events_total = int(state.get("epoch_events_total", 0))
        expected_total = sum(len(self.songs[i].records) - 1 for i in self.eligible)
        if self.epoch_events_total != expected_total:
            raise ValueError(
                "epoch sampler event total mismatch: "
                f"checkpoint={self.epoch_events_total} live={expected_total}"
            )
        if not 0 <= self.epoch_events_seen <= self.epoch_events_total:
            raise ValueError("epoch sampler epoch_events_seen out of range")
