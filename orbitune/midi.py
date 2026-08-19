from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from orbitune.events import NoteEvent


def _u16(value: int) -> bytes:
    return int(value).to_bytes(2, "big")


def _u32(value: int) -> bytes:
    return int(value).to_bytes(4, "big")


def _vlq(value: int) -> bytes:
    if value < 0:
        raise ValueError("VLQ cannot encode negative values")
    buffer = value & 0x7F
    value >>= 7
    out = [buffer]
    while value:
        buffer = (value & 0x7F) | 0x80
        out.insert(0, buffer)
        value >>= 7
    return bytes(out)


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


def write_midi(
    events: list[NoteEvent],
    path: str | Path,
    *,
    bpm: int = 84,
    ticks_per_beat: int = 480,
    positions_per_bar: int = 16,
    beats_per_bar: int = 4,
) -> None:
    """Write a deterministic type-0 MIDI file for generated Orbitune output."""

    ticks_per_position = ticks_per_beat * beats_per_bar // positions_per_bar
    scheduled: list[tuple[int, bytes]] = []
    tempo_us_per_quarter = int(60_000_000 / bpm)
    scheduled.append((0, b"\xff\x51\x03" + tempo_us_per_quarter.to_bytes(3, "big")))
    for event in events:
        event.validate(positions_per_bar=positions_per_bar)
        start = (event.bar * positions_per_bar + event.position) * ticks_per_position
        end = start + max(1, event.duration) * ticks_per_position
        pitch = max(0, min(127, event.pitch))
        velocity = max(1, min(127, event.velocity))
        scheduled.append((start, bytes([0x90, pitch, velocity])))
        scheduled.append((end, bytes([0x80, pitch, 0])))
    scheduled.sort(key=lambda item: (item[0], item[1][0] == 0x80))

    track = bytearray()
    last_time = 0
    for time, message in scheduled:
        track.extend(_vlq(time - last_time))
        track.extend(message)
        last_time = time
    track.extend(_vlq(0))
    track.extend(b"\xff\x2f\x00")

    header = b"MThd" + _u32(6) + _u16(0) + _u16(1) + _u16(ticks_per_beat)
    chunk = b"MTrk" + _u32(len(track)) + bytes(track)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(header + chunk)


def _parse_track(track: bytes, *, ticks_per_position: int, positions_per_bar: int) -> list[NoteEvent]:
    active: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    events: list[NoteEvent] = []
    running_status: int | None = None
    time = 0
    i = 0
    while i < len(track):
        delta, i = _read_vlq(track, i)
        time += delta
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
            if i + length > len(track):
                raise ValueError("truncated MIDI meta payload")
            if meta_type == 0x2F:
                break
            i += length
            continue

        if status in (0xF0, 0xF7):
            length, i = _read_vlq(track, i)
            if i + length > len(track):
                raise ValueError("truncated MIDI SysEx payload")
            i += length
            continue

        command = status & 0xF0
        channel = status & 0x0F
        if command in (0x80, 0x90):
            if i + 2 > len(track):
                raise ValueError("truncated MIDI note event")
            pitch = track[i]
            velocity = track[i + 1]
            i += 2
            key = (channel, pitch)
            if command == 0x90 and velocity > 0:
                active[key].append((time, velocity))
            elif active[key]:
                start, start_velocity = active[key].pop(0)
                start_unit = round(start / ticks_per_position)
                end_unit = max(start_unit + 1, round(time / ticks_per_position))
                events.append(
                    NoteEvent(
                        bar=start_unit // positions_per_bar,
                        position=start_unit % positions_per_bar,
                        pitch=pitch,
                        duration=end_unit - start_unit,
                        velocity=start_velocity,
                    )
                )
            continue

        if command in (0xA0, 0xB0, 0xE0):
            i += 2
        elif command in (0xC0, 0xD0):
            i += 1
        else:
            raise ValueError(f"unsupported MIDI status: 0x{status:02x}")
        if i > len(track):
            raise ValueError("truncated MIDI channel event")
    return events


def read_midi(
    path: str | Path,
    *,
    positions_per_bar: int = 16,
    beats_per_bar: int = 4,
) -> list[NoteEvent]:
    """Read note events from Standard MIDI File type 0 or type 1.

    All note tracks are merged into one symbolic event stream. Type 2 files are
    rejected because their tracks represent independent sequences rather than a
    single synchronized song.
    """

    data = Path(path).read_bytes()
    if len(data) < 14 or data[:4] != b"MThd":
        raise ValueError("not a MIDI file")
    header_len = int.from_bytes(data[4:8], "big")
    if header_len < 6 or len(data) < 8 + header_len:
        raise ValueError("invalid MIDI header")
    midi_format = int.from_bytes(data[8:10], "big")
    track_count = int.from_bytes(data[10:12], "big")
    division = int.from_bytes(data[12:14], "big")
    if midi_format not in (0, 1):
        raise ValueError(f"unsupported MIDI format {midi_format}; Orbitune v0 supports type 0 and type 1")
    if division & 0x8000:
        raise ValueError("SMPTE time division is not supported in Orbitune v0")
    if division <= 0:
        raise ValueError("invalid ticks-per-beat division")

    ticks_per_position = division * beats_per_bar // positions_per_bar
    if ticks_per_position <= 0:
        raise ValueError("MIDI division is too small for the configured quantization")

    offset = 8 + header_len
    events: list[NoteEvent] = []
    tracks_read = 0
    while tracks_read < track_count:
        if offset + 8 > len(data) or data[offset : offset + 4] != b"MTrk":
            raise ValueError("missing or truncated MIDI track chunk")
        track_len = int.from_bytes(data[offset + 4 : offset + 8], "big")
        start = offset + 8
        end = start + track_len
        if end > len(data):
            raise ValueError("truncated MIDI track")
        events.extend(
            _parse_track(
                data[start:end],
                ticks_per_position=ticks_per_position,
                positions_per_bar=positions_per_bar,
            )
        )
        tracks_read += 1
        offset = end

    return sorted(events, key=lambda e: (e.bar, e.position, e.pitch, e.duration, e.velocity))
