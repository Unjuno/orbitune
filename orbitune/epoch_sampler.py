"""Epoch-aware, deterministic, no-replacement TBPTT sampler.

The production commercial-base pretrain must (1) train on every song
exactly once per epoch with no replacement, (2) deterministically
shuffle the per-epoch song order from a stable per-epoch seed, (3)
pad the final partial chunk to ``seq_len + 1`` with
``event_weight = 0`` so the loss/gradient are unaffected, (4) keep
idle lanes at the epoch tail at ``event_weight = 0``, (5) not
prefetch the next epoch, and (6) round-trip a complete
``state_dict`` so an exact checkpoint/resume can continue from the
same next chunks. The sampler is the source of truth for the
``epoch_events_seen`` and ``epoch_events_total`` counters; idle and
padding positions are excluded from both.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from orbitune.compound_indexed import IndexedCompoundSong
from orbitune.compound_tbptt import ChunkBatch


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
    """One optimizer-step's data: the TBPTT chunk and a per-event weight tensor.

    ``event_weight`` has shape ``(batch_size, seq_len)`` with values in
    ``{0, 1}``: ``1`` for active (loss-bearing) positions and ``0`` for
    padding inside a song or for an idle lane at the epoch tail. The
    ``events_counted`` field is the number of ``1`` entries and is
    what the production trainer adds to ``epoch_events_seen``.
    """

    batch: ChunkBatch
    event_weight: torch.Tensor
    events_counted: int


class EpochAwareNoReplacementSampler:
    """No-replacement epoch-aware TBPTT sampler.

    Each epoch visits every eligible song exactly once. Per-epoch the
    song order is a deterministic ``random.Random(epoch_seed).shuffle``
    of the song index list. Within a song the sampler advances in
    monotonic ``seq_len`` chunks. The final partial chunk of a song
    is padded with the song's own last record (a no-op target pair)
    and the per-position ``event_weight`` is set to ``0`` for the
    padded positions so they contribute neither loss nor gradient.

    Idle lanes at the epoch tail (``batch_size > remaining_songs``)
    receive a fully padded chunk with ``event_weight = 0`` and a
    sentinel ``song_index = -1``; the next epoch starts a fresh
    shuffle and never pre-fetches into the new shuffle.

    The ``state_dict`` is the source of truth for exact resume:
    ``corpus_identity / epoch_index / epoch_seed / shuffled_song_order /
    order_cursor / lane song_indices / lane offsets /
    epoch_events_seen / epoch_events_total / batch_size / seq_len /
    weighting mode``. ``load_state_dict`` is fail-closed: it raises if
    the saved corpus identity, batch size, seq length, or weighting
    mode does not match the live sampler.
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
        self.eligible = [i for i, song in enumerate(songs) if len(song.records) >= seq_len + 1]
        if not self.eligible:
            raise ValueError("no song is long enough for the requested seq_len")
        # Per-song eligible completeness counts: number of full chunks per song
        # and the size of the (possibly partial) final tail. These are stable
        # per (corpus_identity, seq_len) so the epoch total event count is
        # deterministic.
        self._chunks_per_song: list[int] = [max(1, (len(songs[i].records) - 1) // self.seq_len) for i in self.eligible]
        self._tail_per_song: list[int] = [
            (len(songs[i].records) - 1) % self.seq_len for i in self.eligible
        ]
        # State
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
        # Shuffle local eligible positions (0..len(eligible)-1) so the
        # per-epoch song order is independent of the corpus song index.
        local_order = list(range(len(self.eligible)))
        self.rng.shuffle(local_order)
        # Map each local position back to the corpus song index for sampler
        # bookkeeping. ``shuffled_song_order`` is the corpus-indexed shuffle;
        # it round-trips through state_dict verbatim.
        self.shuffled_song_order = [self.eligible[local] for local in local_order]
        self.order_cursor = 0
        self.lane_song_indices = [-1] * self.batch_size
        self.lane_offsets = [0] * self.batch_size
        self.lane_tail_left = [0] * self.batch_size
        self.epoch_events_seen = 0
        # Compute deterministic total over the *current* shuffled order.
        # ``_chunks_per_song`` and ``_tail_per_song`` are indexed by the
        # *local* eligible position.
        total = 0
        for local in local_order:
            chunks = self._chunks_per_song[local]
            tail = self._tail_per_song[local]
            # Full chunks contribute seq_len active events each
            total += chunks * self.seq_len
            # Partial tail contributes tail active events (and seq_len - tail padding)
            total += tail
        self.epoch_events_total = total

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
        # Map back to local position for tail lookup
        self._lane_local_index: dict[int, int] = getattr(self, "_lane_local_index", {})
        # Find the local position for the lane's corpus index
        # (linear search; only run on song start so O(N) is fine)
        for local_pos, elig in enumerate(self.eligible):
            if elig == corpus_index:
                self.lane_tail_left[lane] = self._tail_per_song[local_pos]
                break
        else:
            self.lane_tail_left[lane] = 0

    def _lane_complete_chunks_left(self, lane: int) -> int:
        if self.lane_song_indices[lane] < 0:
            return 0
        corpus_index = self.lane_song_indices[lane]
        # Find local position
        for local_pos, elig in enumerate(self.eligible):
            if elig == corpus_index:
                chunks_total = self._chunks_per_song[local_pos]
                consumed = self.lane_offsets[lane] // self.seq_len
                return max(0, chunks_total - consumed)
        return 0

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
        event_weights: list[list[int]] = []
        active_count = 0

        for lane in range(self.batch_size):
            eligible_index = self.lane_song_indices[lane]
            reset = eligible_index < 0
            if not reset and self._lane_complete_chunks_left(lane) <= 0 and self.lane_tail_left[lane] <= 0:
                reset = True
            if reset:
                self._start_lane(lane)
                eligible_index = self.lane_song_indices[lane]
                reset = True
            if eligible_index < 0:
                # Epoch tail: idle lane, fully padded
                song_index_in_corpus = -1
                song = self.songs[0]
                last_record_row = np.asarray(song.records[-1]).reshape(1, -1)
                pad_window = np.tile(last_record_row, (self.seq_len + 1, 1)).astype(np.int64)
                xs.append(torch.from_numpy(pad_window[:-1]))
                ys.append(torch.from_numpy(pad_window[1:]))
                resets.append(True)
                starts.append(0)
                songs_used.append(song_index_in_corpus)
                event_weights.append([0] * self.seq_len)
                continue

            song = self.songs[eligible_index]
            start = self.lane_offsets[lane]
            chunks_left = self._lane_complete_chunks_left(lane)
            tail_left = self.lane_tail_left[lane]
            if chunks_left > 0:
                # Full chunk: every position is active
                window = np.asarray(song.records[start : start + self.seq_len + 1], dtype=np.int64)
                if window.shape[0] != self.seq_len + 1:
                    raise RuntimeError("epoch sampler produced a short chunk")
                weight = [1] * self.seq_len
                self.lane_offsets[lane] += self.seq_len
                resets.append(reset)
            elif tail_left > 0:
                # Partial tail chunk: full window from records, but event_weight
                # is 0 for the padded positions (events beyond song length).
                full_end = start + self.seq_len + 1
                if full_end <= len(song.records):
                    window = np.asarray(song.records[start:full_end], dtype=np.int64)
                else:
                    records = np.asarray(song.records[start:len(song.records)], dtype=np.int64)
                    pad_needed = full_end - len(song.records)
                    last_row = np.asarray(song.records[-1]).reshape(1, -1)
                    pad = np.tile(last_row, (pad_needed, 1))
                    window = np.concatenate([records, pad]).astype(np.int64)
                if window.shape[0] != self.seq_len + 1:
                    raise RuntimeError("epoch sampler produced a short tail chunk")
                weight = [1] * tail_left + [0] * (self.seq_len - tail_left)
                self.lane_offsets[lane] += self.seq_len
                self.lane_tail_left[lane] = 0
                resets.append(False)
            else:
                # No chunks left (already exhausted): start a new song
                self._start_lane(lane)
                eligible_index = self.lane_song_indices[lane]
                if eligible_index < 0:
                    last_record_row = np.asarray(song.records[-1]).reshape(1, -1)
                    pad_window = np.tile(last_record_row, (self.seq_len + 1, 1)).astype(np.int64)
                    xs.append(torch.from_numpy(pad_window[:-1]))
                    ys.append(torch.from_numpy(pad_window[1:]))
                    resets.append(True)
                    starts.append(0)
                    songs_used.append(-1)
                    event_weights.append([0] * self.seq_len)
                    continue
                song = self.songs[eligible_index]
                start = 0
                window = np.asarray(song.records[start : start + self.seq_len + 1], dtype=np.int64)
                if window.shape[0] != self.seq_len + 1:
                    raise RuntimeError("epoch sampler produced a short chunk")
                weight = [1] * self.seq_len
                self.lane_offsets[lane] = self.seq_len
                resets.append(True)
            starts.append(start)
            songs_used.append(eligible_index)
            xs.append(torch.from_numpy(window[:-1]))
            ys.append(torch.from_numpy(window[1:]))
            event_weights.append(weight)
            active_count += sum(weight)

        # Cap epoch_events_seen at the deterministic total. This guards against
        # rounding/edge effects in partial-tail bookkeeping.
        if self.epoch_events_seen + active_count > self.epoch_events_total:
            active_count = self.epoch_events_total - self.epoch_events_seen
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
            "schema": "orbitune-epoch-sampler-v1",
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
        if str(state.get("schema", "")) != "orbitune-epoch-sampler-v1":
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
        if set(order) != set(self.eligible):
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
        self.epoch_events_seen = int(state.get("epoch_events_seen", 0))
        self.epoch_events_total = int(state.get("epoch_events_total", 0))
