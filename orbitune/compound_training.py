from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


COMPOUND_RECORD_WIDTH = 12


@dataclass(frozen=True, slots=True)
class CompoundSong:
    path: str
    sha256: str
    records: tuple[tuple[int, ...], ...]


def load_compound_jsonl(paths: str | Path | Iterable[str | Path]) -> list[CompoundSong]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    songs: list[CompoundSong] = []
    for source in paths:
        with Path(source).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                raw_records = payload.get("records")
                if not isinstance(raw_records, list) or not raw_records:
                    raise ValueError(f"{source}:{line_number}: missing compound records")
                records: list[tuple[int, ...]] = []
                for record in raw_records:
                    if not isinstance(record, list) or len(record) != COMPOUND_RECORD_WIDTH:
                        raise ValueError(
                            f"{source}:{line_number}: each compound record must have width {COMPOUND_RECORD_WIDTH}"
                        )
                    records.append(tuple(int(value) for value in record))
                songs.append(
                    CompoundSong(
                        path=str(payload.get("path", "")),
                        sha256=str(payload.get("sha256", "")),
                        records=tuple(records),
                    )
                )
    if not songs:
        raise ValueError("no compound songs loaded")
    return songs


def sample_compound_batch(
    songs: list[CompoundSong],
    *,
    batch_size: int,
    seq_len: int,
    rng: random.Random | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample song-local next-event windows without crossing composition boundaries."""

    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    eligible = [song for song in songs if len(song.records) >= seq_len + 1]
    if not eligible:
        raise ValueError("no song is long enough for the requested seq_len")
    rng = rng or random.Random()
    inputs: list[list[tuple[int, ...]]] = []
    targets: list[list[tuple[int, ...]]] = []
    for _ in range(batch_size):
        song = rng.choice(eligible)
        start = rng.randrange(0, len(song.records) - seq_len)
        window = song.records[start : start + seq_len + 1]
        inputs.append(list(window[:-1]))
        targets.append(list(window[1:]))
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )
