from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from orbitune.pretrain_corpus import commercial_safe_sources, load_registry
from scripts import build_pretrain_corpus as build


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "configs" / "pretrain_corpus_commercial_v2.json"
V3 = ROOT / "configs" / "pretrain_corpus_commercial_v3.json"


def test_commercial_v3_extends_v2_with_only_pinned_nifc_sources() -> None:
    v2 = load_registry(V2)
    v3 = load_registry(V3)
    v2_ids = [source.id for source in commercial_safe_sources(v2)]
    v3_ids = [source.id for source in commercial_safe_sources(v3)]

    assert v3["name"] == "orbitune-commercial-safe-v3"
    assert v3_ids[:-2] == v2_ids
    assert v3_ids[-2:] == ["nifc_polish_scores", "nifc_chopin_first_editions"]
    assert v3["split"]["seed"] == v2["split"]["seed"]


def test_nifc_sources_are_immutable_ccby4_humdrum_inputs() -> None:
    registry = load_registry(V3)
    by_id = {item["id"]: item for item in registry["sources"]}

    polish = by_id["nifc_polish_scores"]
    assert polish["commercial_safe"] is True
    assert polish["license"] == "cc-by-4.0"
    assert polish["git_url"] == "https://github.com/pl-wnifc/humdrum-polish-scores.git"
    assert polish["ref"] == "13ac964e0dd8bcd5fffd837169cbf653242c12e8"
    assert len(polish["ref"]) == 40
    assert polish["converter"] == "hum2mid"
    assert polish["humdrum_globs"] == ["**/kern/*.krn"]
    assert polish["attribution_required"] is True

    chopin = by_id["nifc_chopin_first_editions"]
    assert chopin["commercial_safe"] is True
    assert chopin["license"] == "cc-by-4.0"
    assert chopin["git_url"] == "https://github.com/pl-wnifc/humdrum-chopin-first-editions.git"
    assert chopin["ref"] == "95dfb105c1669c72d10b04088566154f12d3dc1c"
    assert len(chopin["ref"]) == 40
    assert chopin["converter"] == "hum2mid"
    assert chopin["humdrum_globs"] == ["kern/*.krn"]
    assert chopin["attribution_required"] is True

    deny = set(registry["license_policy"]["deny_markers"])
    assert {"-nc", "noncommercial", "sharealike", "cc-by-sa"} <= deny


def test_humdrum_converter_uses_official_hum2mid_contract_and_utf8(tmp_path, monkeypatch) -> None:
    source_root = tmp_path / "scores"
    source = source_root / "archive" / "kern" / "piece.krn"
    source.parent.mkdir(parents=True)
    source.write_text("**kern\n4c\n*-\n", encoding="utf-8")
    converted_root = tmp_path / "converted"

    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((list(command), kwargs))
        output = Path(command[command.index("-o") + 1])
        output.write_bytes(b"MThd")
        return SimpleNamespace(returncode=0, stdout="diagnostic: café ✓")

    monkeypatch.setattr(build.subprocess, "run", fake_run)

    report = build.convert_humdrum_to_midi(
        source_root,
        converted_root,
        {"humdrum_globs": ["**/kern/*.krn"]},
        hum2mid_bin="hum2mid-test",
    )

    expected_output = converted_root / "archive" / "kern" / "piece.mid"
    assert report == {"score_candidates": 1, "converted": 1, "cached": 0, "failed": []}
    assert calls == [
        (
            ["hum2mid-test", str(source), "-o", str(expected_output)],
            {
                "stdout": build.subprocess.PIPE,
                "stderr": build.subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "check": False,
            },
        )
    ]


def test_humdrum_converter_fails_closed_when_cli_is_missing(tmp_path) -> None:
    source_root = tmp_path / "scores"
    source = source_root / "kern" / "piece.krn"
    source.parent.mkdir(parents=True)
    source.write_text("**kern\n4c\n*-\n", encoding="utf-8")

    report = build.convert_humdrum_to_midi(
        source_root,
        tmp_path / "converted",
        {"humdrum_globs": ["kern/*.krn"]},
        hum2mid_bin=None,
    )

    assert report["score_candidates"] == 1
    assert report["converted"] == 0
    assert "blocked" in report
    assert "--hum2mid-bin" in str(report["blocked"])
