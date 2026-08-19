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


def test_type1_multitrack_midi_reads_note_track(tmp_path):
    type0_path = tmp_path / "type0.mid"
    write_midi(make_demo_events(bars=1), type0_path, bpm=84)
    raw = type0_path.read_bytes()
    division = raw[12:14]
    note_track_chunk = raw[14:]

    meta_body = b"\x00\xff\x51\x03\x07\xa1\x20\x00\xff\x2f\x00"
    meta_track_chunk = b"MTrk" + len(meta_body).to_bytes(4, "big") + meta_body
    header = b"MThd" + (6).to_bytes(4, "big") + (1).to_bytes(2, "big") + (2).to_bytes(2, "big") + division
    type1_path = tmp_path / "type1.mid"
    type1_path.write_bytes(header + meta_track_chunk + note_track_chunk)

    events = read_midi(type1_path)
    assert events
    assert any(event.pitch >= 21 for event in events)
