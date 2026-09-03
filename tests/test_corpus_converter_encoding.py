from __future__ import annotations

from types import SimpleNamespace

from scripts import build_pretrain_corpus as build


def _fake_run_factory(asserted: list[dict[str, object]]):
    def fake_run(command, **kwargs):
        asserted.append(kwargs)
        output_index = command.index("-o") + 1
        output = build.Path(command[output_index])
        if command[0] == "musescore-test":
            output.write_bytes(b"MThd")
        else:
            output.with_suffix(".midi").write_bytes(b"MThd")
        return SimpleNamespace(returncode=0, stdout="diagnostic: café ✓")

    return fake_run


def test_musescore_converter_uses_utf8_replacement_decoding(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "scores"
    source_root.mkdir()
    score = source_root / "piece.mscz"
    score.write_bytes(b"score")
    converted_root = tmp_path / "converted"

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(build.subprocess, "run", _fake_run_factory(calls))

    report = build.convert_scores_to_midi(
        source_root,
        converted_root,
        {"score_globs": ["*.mscz"]},
        musescore_bin="musescore-test",
    )

    assert report["converted"] == 1
    assert calls == [
        {
            "stdout": build.subprocess.PIPE,
            "stderr": build.subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "check": False,
        }
    ]


def test_lilypond_converter_uses_utf8_replacement_decoding(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "mutopia"
    source = source_root / "ftp" / "Composer" / "piece.ly"
    source.parent.mkdir(parents=True)
    source.write_text('mutopiatitle = "Piece"\nlicense = "Public Domain"\n\\midi { }\n', encoding="utf-8")
    converted_root = tmp_path / "converted"

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(build.subprocess, "run", _fake_run_factory(calls))

    report = build.convert_mutopia_to_midi(
        source_root,
        converted_root,
        lilypond_bin="lilypond-test",
    )

    assert report["converted"] == 1
    assert calls == [
        {
            "cwd": source.parent,
            "stdout": build.subprocess.PIPE,
            "stderr": build.subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "check": False,
        }
    ]
