from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from orbitune.compound import COMPOUND_TOKENIZER_ABI


COMPOUND_RECORD_WIDTH = 12


@dataclass(frozen=True, slots=True)
class CompoundSong:
    path: str
    sha256: str
    tokenizer_abi: str
    records: tuple[tuple[int, ...], ...]


def _validate_record(record: tuple[int, ...], *, source: str | Path, line_number: int) -> None:
    if len(record) != COMPOUND_RECORD_WIDTH:
        raise ValueError(
            f"{source}:{line_number}: each compound record must have width {COMPOUND_RECORD_WIDTH}"
        )
    if any(value < 0 for value in record):
        raise ValueError(f"{source}:{line_number}: compound record values must be non-negative")
    event_type, channel, delta_coarse, delta_residual = record[:4]
    duration_coarse, duration_residual, continuous_coarse, continuous_residual = record[8:12]
    if not 0 <= event_type <= 9:
        raise ValueError(f"{source}:{line_number}: invalid compound event type {event_type}")
    if not 0 <= channel <= 15:
        raise ValueError(f"{source}:{line_number}: invalid MIDI channel {channel}")
    if not 0 <= delta_coarse < 7 or not 0 <= duration_coarse < 7:
        raise ValueError(f"{source}:{line_number}: invalid time coarse index")
    if not 0 <= delta_residual < 16 or not 0 <= duration_residual < 16:
        raise ValueError(f"{source}:{line_number}: invalid time residual index")
    if not 0 <= continuous_coarse < 8 or not 0 <= continuous_residual < 8:
        raise ValueError(f"{source}:{line_number}: invalid continuous factor index")


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
                tokenizer_abi = str(payload.get("tokenizer_abi", ""))
                if tokenizer_abi != COMPOUND_TOKENIZER_ABI:
                    raise ValueError(
                        f"{source}:{line_number}: tokenizer ABI {tokenizer_abi!r} does not match "
                        f"{COMPOUND_TOKENIZER_ABI!r}"
                    )
                if int(payload.get("record_width", -1)) != COMPOUND_RECORD_WIDTH:
                    raise ValueError(
                        f"{source}:{line_number}: record_width does not match {COMPOUND_RECORD_WIDTH}"
                    )
                raw_records = payload.get("records")
                if not isinstance(raw_records, list) or not raw_records:
                    raise ValueError(f"{source}:{line_number}: missing compound records")
                records: list[tuple[int, ...]] = []
                for raw_record in raw_records:
                    if not isinstance(raw_record, list):
                        raise ValueError(f"{source}:{line_number}: compound record must be a list")
                    record = tuple(int(value) for value in raw_record)
                    _validate_record(record, source=source, line_number=line_number)
                    records.append(record)
                songs.append(
                    CompoundSong(
                        path=str(payload.get("path", "")),
                        sha256=str(payload.get("sha256", "")),
                        tokenizer_abi=tokenizer_abi,
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
