from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np
import torch

from orbitune.compound_indexed import IndexedCompoundSong
from orbitune.compound_tbptt import ChunkBatch


def _corpus_identity(songs: list[IndexedCompoundSong]) -> str:
    """Stable identity for sampler-visible corpus order, content and weights."""
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


class IndexedTensorSampler:
    """Fixed-window sampler that copies only sampled windows from a memmap."""

    def __init__(self, songs: list[IndexedCompoundSong], *, weighted: bool = False) -> None:
        self.songs = songs
        self.weighted = bool(weighted)
        self.corpus_identity = _corpus_identity(songs)
        self._eligible_cache: dict[int, list[int]] = {}

    def _eligible(self, seq: int) -> list[int]:
        eligible = self._eligible_cache.get(seq)
        if eligible is None:
            eligible = [index for index, song in enumerate(self.songs) if len(song.records) >= seq + 1]
            if not eligible:
                raise ValueError("no indexed song is long enough for requested seq_len")
            self._eligible_cache[seq] = eligible
        return eligible

    def _choose(self, eligible: list[int], rng: random.Random) -> int:
        if not self.weighted:
            return rng.choice(eligible)
        weights = [max(0.0, float(self.songs[index].sampling_weight)) for index in eligible]
        if not any(weights):
            raise ValueError("all indexed corpus sampling weights are zero")
        return rng.choices(eligible, weights=weights, k=1)[0]

    def sample(self, batch: int, seq: int, rng: random.Random, device: torch.device):
        eligible = self._eligible(seq)
        windows: list[torch.Tensor] = []
        for _ in range(batch):
            song = self.songs[self._choose(eligible, rng)]
            start = rng.randrange(0, len(song.records) - seq)
            window = np.asarray(song.records[start : start + seq + 1], dtype=np.int64)
            windows.append(torch.from_numpy(window))
        joined = torch.stack(windows).to(device)
        return joined[:, :-1], joined[:, 1:]


class IndexedSequentialSongChunkSampler:
    """Song-sequential TBPTT sampler over memory-mapped indexed songs."""

    def __init__(
        self,
        songs: list[IndexedCompoundSong],
        *,
        batch_size: int,
        seq_len: int,
        rng: random.Random,
        weighted: bool = False,
    ) -> None:
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        self.songs = songs
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.rng = rng
        self.weighted = bool(weighted)
        self.corpus_identity = _corpus_identity(songs)
        self.eligible = [i for i, song in enumerate(songs) if len(song.records) >= seq_len + 1]
        if not self.eligible:
            raise ValueError("no indexed song is long enough for the requested seq_len")
        self.song_indices = [-1] * self.batch_size
        self.offsets = [0] * self.batch_size

    def _start_lane(self, lane: int) -> None:
        if self.weighted:
            weights = [max(0.0, float(self.songs[index].sampling_weight)) for index in self.eligible]
            if not any(weights):
                raise ValueError("all indexed corpus sampling weights are zero")
            song_index = self.rng.choices(self.eligible, weights=weights, k=1)[0]
        else:
            song_index = self.rng.choice(self.eligible)
        self.song_indices[lane] = song_index
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
            window = np.asarray(song.records[start : start + self.seq_len + 1], dtype=np.int64)
            if window.shape[0] != self.seq_len + 1:
                raise RuntimeError("indexed sequential sampler produced a short chunk")
            tensor = torch.from_numpy(window)
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
            "weighted": self.weighted,
            "corpus_identity": self.corpus_identity,
            "song_indices": list(self.song_indices),
            "offsets": list(self.offsets),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("batch_size", -1)) != self.batch_size:
            raise ValueError("TBPTT sampler batch_size mismatch")
        if int(state.get("seq_len", -1)) != self.seq_len:
            raise ValueError("TBPTT sampler seq_len mismatch")
        if bool(state.get("weighted", False)) != self.weighted:
            raise ValueError("TBPTT sampler weighted-mode mismatch")
        if str(state.get("corpus_identity", "")) != self.corpus_identity:
            raise ValueError("TBPTT sampler corpus identity mismatch")
        song_indices = [int(value) for value in state.get("song_indices", [])]
        offsets = [int(value) for value in state.get("offsets", [])]
        if len(song_indices) != self.batch_size or len(offsets) != self.batch_size:
            raise ValueError("TBPTT sampler lane-state length mismatch")
        for song_index, offset in zip(song_indices, offsets):
            if song_index != -1 and song_index not in self.eligible:
                raise ValueError(f"invalid TBPTT sampler song index {song_index}")
            if offset < 0:
                raise ValueError("TBPTT sampler offset must be non-negative")
        self.song_indices = song_indices
        self.offsets = offsets
