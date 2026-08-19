from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MidiMetadata:
    midi_format: int
    track_count: int
    ticks_per_beat: int
    time_signatures: tuple[tuple[int, int], ...]


def _read_vlq(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        if offset >= len(data):
            raise ValueError("truncated MIDI VLQ")
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, offset


def _track_time_signatures(track: bytes) -> list[tuple[int, int]]:
    signatures: list[tuple[int, int]] = []
    running_status: int | None = None
    i = 0
    while i < len(track):
        _, i = _read_vlq(track, i)
        if i >= len(track):
            raise ValueError("truncated MIDI event")
        raw_status = track[i]
        if raw_status < 0x80:
            if running_status is None:
                raise ValueError("running status without prior channel status")
            status = running_status
        else:
            status = raw_status
            i += 1
            if status < 0xF0:
                running_status = status

        if status == 0xFF:
            if i >= len(track):
                raise ValueError("truncated MIDI meta event")
            meta_type = track[i]
            i += 1
            length, i = _read_vlq(track, i)
            end = i + length
            if end > len(track):
                raise ValueError("truncated MIDI meta payload")
            if meta_type == 0x58 and length >= 2:
                numerator = track[i]
                denominator = 1 << track[i + 1]
                signatures.append((numerator, denominator))
            if meta_type == 0x2F:
                break
            i = end
            continue

        if status in (0xF0, 0xF7):
            length, i = _read_vlq(track, i)
            i += length
            if i > len(track):
                raise ValueError("truncated MIDI SysEx payload")
            continue

        command = status & 0xF0
        if command in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
            i += 2
        elif command in (0xC0, 0xD0):
            i += 1
        else:
            raise ValueError(f"unsupported MIDI status: 0x{status:02x}")
        if i > len(track):
            raise ValueError("truncated MIDI channel event")
    return signatures


def inspect_midi_metadata(path: str | Path) -> MidiMetadata:
    data = Path(path).read_bytes()
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError("not a MIDI file")
    header_len = int.from_bytes(data[4:8], "big")
    if header_len < 6 or len(data) < 8 + header_len:
        raise ValueError("invalid MIDI header")
    midi_format = int.from_bytes(data[8:10], "big")
    track_count = int.from_bytes(data[10:12], "big")
    division = int.from_bytes(data[12:14], "big")
    if division & 0x8000:
        raise ValueError("SMPTE time division is not supported")

    offset = 8 + header_len
    signatures: list[tuple[int, int]] = []
    for _ in range(track_count):
        if offset + 8 > len(data) or data[offset : offset + 4] != b"MTrk":
            raise ValueError("missing or truncated MIDI track chunk")
        length = int.from_bytes(data[offset + 4 : offset + 8], "big")
        start = offset + 8
        end = start + length
        if end > len(data):
            raise ValueError("truncated MIDI track")
        signatures.extend(_track_time_signatures(data[start:end]))
        offset = end

    unique = tuple(dict.fromkeys(signatures))
    return MidiMetadata(
        midi_format=midi_format,
        track_count=track_count,
        ticks_per_beat=division,
        time_signatures=unique,
    )


def is_4_4_compatible(metadata: MidiMetadata) -> bool:
    """v0 assumes 4/4. Files without an explicit time signature are accepted as the v0 default."""
    return not metadata.time_signatures or metadata.time_signatures == ((4, 4),)
