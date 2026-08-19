from orbitune.dataset import prepare_corpus
from orbitune.demo import make_demo_events
from orbitune.midi import write_midi
from orbitune.midi_metadata import inspect_midi_metadata, is_4_4_compatible


def _with_time_signature(raw: bytes, numerator: int, denominator_power: int) -> bytes:
    header = raw[:14]
    track_length = int.from_bytes(raw[18:22], "big")
    body = raw[22 : 22 + track_length]
    event = bytes([0x00, 0xFF, 0x58, 0x04, numerator, denominator_power, 24, 8])
    new_body = event + body
    return header + b"MTrk" + len(new_body).to_bytes(4, "big") + new_body


def test_metadata_detects_three_four(tmp_path):
    original = tmp_path / "original.mid"
    three_four = tmp_path / "three-four.mid"
    write_midi(make_demo_events(bars=1), original)
    three_four.write_bytes(_with_time_signature(original.read_bytes(), 3, 2))

    metadata = inspect_midi_metadata(three_four)
    assert metadata.time_signatures == ((3, 4),)
    assert not is_4_4_compatible(metadata)


def test_prepare_corpus_rejects_non_four_four(tmp_path):
    source = tmp_path / "midi"
    source.mkdir()
    four_four = source / "four-four.mid"
    three_four = source / "three-four.mid"
    write_midi(make_demo_events(bars=1), four_four)
    three_four.write_bytes(_with_time_signature(four_four.read_bytes(), 3, 2))

    report = prepare_corpus(source, tmp_path / "corpus.tokens", tmp_path / "report.json")
    assert report["files_seen"] == 2
    assert report["files_accepted"] == 1
    assert report["files_rejected"] == 1
    assert "unsupported_time_signature" in report["rejected"][0]["reason"]
