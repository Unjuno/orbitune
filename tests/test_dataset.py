from __future__ import annotations

from orbitune.dataset import prepare_corpus
from orbitune.demo import make_demo_events
from orbitune.midi import write_midi


def test_prepare_corpus(tmp_path):
    source = tmp_path / "midi"
    source.mkdir()
    write_midi(make_demo_events(bars=2, bpm=84), source / "a.mid", bpm=84)
    write_midi(make_demo_events(bars=1, bpm=72), source / "b.mid", bpm=72)

    tokens = tmp_path / "corpus.tokens"
    report_path = tmp_path / "report.json"
    report = prepare_corpus(source, tokens, report_path)

    assert report["files_seen"] == 2
    assert report["files_accepted"] == 2
    assert report["files_rejected"] == 0
    assert report["total_tokens"] > 0
    assert tokens.exists()
    assert report_path.exists()
    assert "BAR" in tokens.read_text(encoding="utf-8")
