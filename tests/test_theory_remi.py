from orbitune.events import NoteEvent
from orbitune.tokenizer import TheoryRemiTokenizer


def test_theory_remi_roundtrip_events():
    tokenizer = TheoryRemiTokenizer()
    events = [
        NoteEvent(bar=0, position=0, pitch=60, duration=4, velocity=64),
        NoteEvent(bar=0, position=8, pitch=67, duration=4, velocity=48),
        NoteEvent(bar=1, position=0, pitch=64, duration=8, velocity=40),
    ]
    tokens = tokenizer.encode_events(events)
    decoded = tokenizer.decode_events(tokens)
    assert [(e.bar, e.position, e.pitch, e.duration) for e in decoded] == [
        (0, 0, 60, 4),
        (0, 8, 67, 4),
        (1, 0, 64, 8),
    ]
    assert all(1 <= event.velocity <= 127 for event in decoded)
