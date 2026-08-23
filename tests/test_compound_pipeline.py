from pathlib import Path

from orbitune.compound import CompoundEvent, CompoundEventType
from orbitune.compound_midi import read_compound_midi
from orbitune.midi import _u16, _u32, _vlq
from orbitune.quantization import dequantize_unsigned, quantize_unsigned
from orbitune.tokenizer import CompoundEventTokenizer


def _midi(messages: list[tuple[int, bytes]], *, division: int = 480) -> bytes:
    track = bytearray()
    for delta, message in messages:
        track.extend(_vlq(delta))
        track.extend(message)
    track.extend(_vlq(0))
    track.extend(b"\xff\x2f\x00")
    return (
        b"MThd"
        + _u32(6)
        + _u16(0)
        + _u16(1)
        + _u16(division)
        + b"MTrk"
        + _u32(len(track))
        + bytes(track)
    )


def test_compound_midi_preserves_supported_event_types(tmp_path: Path) -> None:
    payload = _midi(
        [
            (0, b"\xff\x51\x03\x07\xa1\x20"),  # 120 BPM
            (0, b"\xff\x58\x04\x04\x02\x18\x08"),
            (0, bytes([0xC0, 5])),
            (0, bytes([0xB0, 64, 127])),
            (0, bytes([0xE0, 0, 64])),
            (0, bytes([0xD0, 70])),
            (0, bytes([0xA0, 60, 80])),
            (0, bytes([0x90, 60, 90])),
            (480, bytes([0x80, 60, 0])),
        ]
    )
    path = tmp_path / "compound.mid"
    path.write_bytes(payload)
    events = read_compound_midi(path)
    types = {event.type for event in events}
    assert CompoundEventType.TEMPO in types
    assert CompoundEventType.TIME_SIGNATURE in types
    assert CompoundEventType.PROGRAM in types
    assert CompoundEventType.PEDAL in types
    assert CompoundEventType.PITCH_BEND in types
    assert CompoundEventType.CHANNEL_PRESSURE in types
    assert CompoundEventType.POLY_PRESSURE in types
    assert CompoundEventType.NOTE in types
    tempo = next(event for event in events if event.type is CompoundEventType.TEMPO)
    assert tempo.a1 == 120
    note = next(event for event in events if event.type is CompoundEventType.NOTE)
    assert note.step == 0
    assert note.a2 == 96


def test_compound_tokenizer_is_one_record_per_event() -> None:
    events = [
        CompoundEvent(CompoundEventType.PROGRAM, 0, 0, 5),
        CompoundEvent(CompoundEventType.NOTE, 24, 0, 60, 48, 90),
        CompoundEvent(CompoundEventType.CC, 48, 0, 1, 100),
        CompoundEvent(CompoundEventType.PITCH_BEND, 72, 0, 10000),
    ]
    tokenizer = CompoundEventTokenizer()
    records = tokenizer.encode_events(events)
    assert len(records) == len(events)
    assert all(len(record.as_tuple()) == 12 for record in records)
    decoded = tokenizer.decode_records(records)
    assert [event.type for event in decoded] == [event.type for event in events]


def test_continuous_factorization_has_bounded_error() -> None:
    for maximum, tolerance in [(127, 3), (16383, 300)]:
        for raw in [0, maximum // 8, maximum // 2, 7 * maximum // 8, maximum]:
            quantized = quantize_unsigned(raw, maximum=maximum)
            restored = dequantize_unsigned(quantized, maximum=maximum)
            assert abs(restored - raw) <= tolerance
