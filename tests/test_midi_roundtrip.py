from orbitune.demo import make_demo_events
from orbitune.midi import read_midi, write_midi
from orbitune.tokenizer import TheoryRemiTokenizer


def test_demo_midi_roundtrip(tmp_path):
    midi_path = tmp_path / "demo.mid"
    write_midi(make_demo_events(bars=2), midi_path, bpm=84)
    events = read_midi(midi_path)
    assert events
    tokens = TheoryRemiTokenizer().encode_events(events)
    assert "BAR" in tokens
    assert any(token.startswith("NOTE_PITCH_") for token in tokens)
