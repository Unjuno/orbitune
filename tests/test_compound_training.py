import json
import random

from orbitune.compound import COMPOUND_TOKENIZER_ABI
from orbitune.compound_training import COMPOUND_RECORD_WIDTH, load_compound_jsonl, sample_compound_batch


def _payload(path: str, sha256: str, records: list[list[int]]) -> dict[str, object]:
    return {
        "tokenizer_abi": COMPOUND_TOKENIZER_ABI,
        "record_width": COMPOUND_RECORD_WIDTH,
        "path": path,
        "sha256": sha256,
        "records": records,
    }


def test_compound_loader_and_song_local_windows(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    records = [[0, 0, 0, 0, i % 128, 0, 90, 0, 0, 0, 0, 0] for i in range(12)]
    payloads = [
        _payload("a.mid", "a", records),
        _payload("b.mid", "b", records[::-1]),
    ]
    path.write_text("\n".join(json.dumps(item) for item in payloads) + "\n", encoding="utf-8")
    songs = load_compound_jsonl(path)
    assert len(songs) == 2
    assert all(song.tokenizer_abi == COMPOUND_TOKENIZER_ABI for song in songs)
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
    payload = _payload("bad.mid", "x", [[1, 2, 3]])
    path.write_text(json.dumps(payload) + "\n")
    try:
        load_compound_jsonl(path)
    except ValueError as exc:
        assert "record must have width" in str(exc)
    else:
        raise AssertionError("invalid record width must fail")


def test_compound_loader_rejects_missing_or_wrong_abi(tmp_path) -> None:
    path = tmp_path / "wrong-abi.jsonl"
    record = [0, 0, 0, 0, 60, 0, 90, 0, 0, 0, 0, 0]
    payload = _payload("a.mid", "a", [record])
    payload["tokenizer_abi"] = "future-incompatible-abi"
    path.write_text(json.dumps(payload) + "\n")
    try:
        load_compound_jsonl(path)
    except ValueError as exc:
        assert "tokenizer ABI" in str(exc)
    else:
        raise AssertionError("mismatched tokenizer ABI must fail")
