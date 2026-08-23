import json
import random

from orbitune.compound_training import COMPOUND_RECORD_WIDTH, load_compound_jsonl, sample_compound_batch


def test_compound_loader_and_song_local_windows(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    records = [[0, 0, 0, 0, i % 128, 0, 90, 0, 0, 0, 0, 0] for i in range(12)]
    payloads = [
        {"path": "a.mid", "sha256": "a", "records": records},
        {"path": "b.mid", "sha256": "b", "records": records[::-1]},
    ]
    path.write_text("\n".join(json.dumps(item) for item in payloads) + "\n", encoding="utf-8")
    songs = load_compound_jsonl(path)
    assert len(songs) == 2
    inputs, targets = sample_compound_batch(
        songs,
        batch_size=4,
        seq_len=5,
        rng=random.Random(7),
    )
    assert inputs.shape == (4, 5, COMPOUND_RECORD_WIDTH)
    assert targets.shape == inputs.shape
    assert (inputs[:, 1:] == targets[:, :-1]).all()


def test_compound_loader_rejects_wrong_record_width(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"path": "bad.mid", "sha256": "x", "records": [[1, 2, 3]]}) + "\n")
    try:
        load_compound_jsonl(path)
    except ValueError as exc:
        assert "record must have width" in str(exc)
    else:
        raise AssertionError("invalid record width must fail")
