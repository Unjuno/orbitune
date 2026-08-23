from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from orbitune.compound import CompoundEvent, CompoundEventType, TEMPORAL_RESOLUTION, canonicalize_events
from orbitune.midi import _read_vlq


def _step(ticks: int, division: int) -> int:
    return round(ticks * TEMPORAL_RESOLUTION / division)


def _parse_compound_track(track: bytes, *, division: int) -> list[CompoundEvent]:
    active: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    events: list[CompoundEvent] = []
    bank_msb = [0] * 16
    bank_lsb = [0] * 16
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
            payload = track[i : i + length]
            i += length
            if meta_type == 0x2F:
                break
            if meta_type == 0x51 and length == 3:
                microseconds_per_quarter = int.from_bytes(payload, "big")
                if microseconds_per_quarter > 0:
                    bpm = max(1, min(999, round(60_000_000 / microseconds_per_quarter)))
                    events.append(CompoundEvent(CompoundEventType.TEMPO, _step(time, division), 0, bpm))
            elif meta_type == 0x58 and length >= 2:
                events.append(
                    CompoundEvent(
                        CompoundEventType.TIME_SIGNATURE,
                        _step(time, division),
                        0,
                        payload[0],
                        1 << payload[1],
                    )
                )
            continue

        if status in (0xF0, 0xF7):
            length, i = _read_vlq(track, i)
            if i + length > len(track):
                raise ValueError("truncated MIDI SysEx payload")
            i += length
            continue

        command = status & 0xF0
        channel = status & 0x0F
        step = _step(time, division)

        if command in (0x80, 0x90):
            if i + 2 > len(track):
                raise ValueError("truncated MIDI note event")
            pitch, velocity = track[i], track[i + 1]
            i += 2
            key = (channel, pitch)
            if command == 0x90 and velocity > 0:
                active[key].append((time, velocity))
            elif active[key]:
                start, start_velocity = active[key].pop(0)
                start_step = _step(start, division)
                end_step = max(start_step + 1, step)
                events.append(
                    CompoundEvent(
                        CompoundEventType.NOTE,
                        start_step,
                        channel,
                        pitch,
                        end_step - start_step,
                        start_velocity,
                    )
                )
            continue

        if command == 0xA0:
            if i + 2 > len(track):
                raise ValueError("truncated poly pressure event")
            pitch, value = track[i], track[i + 1]
            i += 2
            events.append(CompoundEvent(CompoundEventType.POLY_PRESSURE, step, channel, pitch, value))
            continue

        if command == 0xB0:
            if i + 2 > len(track):
                raise ValueError("truncated control-change event")
            controller, value = track[i], track[i + 1]
            i += 2
            if controller == 0:
                bank_msb[channel] = value
                events.append(CompoundEvent(CompoundEventType.BANK, step, channel, bank_msb[channel], bank_lsb[channel]))
            elif controller == 32:
                bank_lsb[channel] = value
                events.append(CompoundEvent(CompoundEventType.BANK, step, channel, bank_msb[channel], bank_lsb[channel]))
            elif controller == 64:
                events.append(CompoundEvent(CompoundEventType.PEDAL, step, channel, int(value >= 64)))
            else:
                events.append(CompoundEvent(CompoundEventType.CC, step, channel, controller, value))
            continue

        if command == 0xC0:
            if i >= len(track):
                raise ValueError("truncated program-change event")
            events.append(CompoundEvent(CompoundEventType.PROGRAM, step, channel, track[i]))
            i += 1
            continue

        if command == 0xD0:
            if i >= len(track):
                raise ValueError("truncated channel-pressure event")
            events.append(CompoundEvent(CompoundEventType.CHANNEL_PRESSURE, step, channel, track[i]))
            i += 1
            continue

        if command == 0xE0:
            if i + 2 > len(track):
                raise ValueError("truncated pitch-bend event")
            lsb, msb = track[i], track[i + 1]
            i += 2
            events.append(CompoundEvent(CompoundEventType.PITCH_BEND, step, channel, (msb << 7) | lsb))
            continue

        raise ValueError(f"unsupported MIDI status: 0x{status:02x}")

    return events


def read_compound_midi(path: str | Path) -> list[CompoundEvent]:
    """Read a Standard MIDI type-0/1 file into canonical Compound Events.

    This parser is intentionally separate from :func:`orbitune.midi.read_midi`
    so the legacy `theory-remi-v0` pipeline remains behaviorally stable while
    the experimental Compound ABI is validated.
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
        raise ValueError(f"unsupported MIDI format {midi_format}; Compound parser supports type 0 and type 1")
    if division & 0x8000:
        raise ValueError("SMPTE time division is not supported")
    if division <= 0:
        raise ValueError("invalid ticks-per-beat division")

    offset = 8 + header_len
    events: list[CompoundEvent] = []
    for _ in range(track_count):
        if offset + 8 > len(data) or data[offset : offset + 4] != b"MTrk":
            raise ValueError("missing or truncated MIDI track chunk")
        track_len = int.from_bytes(data[offset + 4 : offset + 8], "big")
        start = offset + 8
        end = start + track_len
        if end > len(data):
            raise ValueError("truncated MIDI track")
        events.extend(_parse_compound_track(data[start:end], division=division))
        offset = end

    return canonicalize_events(events)
