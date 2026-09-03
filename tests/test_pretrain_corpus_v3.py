from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from orbitune.pretrain_corpus import commercial_safe_sources, load_registry
from scripts import build_pretrain_corpus as build
from scripts import install_pretrain_corpora as install


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "configs" / "pretrain_corpus_commercial_v2.json"
V3 = ROOT / "configs" / "pretrain_corpus_commercial_v3.json"


def test_commercial_v3_extends_v2_with_muse_omr_and_pinned_nifc_sources() -> None:
    v2 = load_registry(V2)
    v3 = load_registry(V3)
    v2_ids = [source.id for source in commercial_safe_sources(v2)]
    v3_ids = [source.id for source in commercial_safe_sources(v3)]

    assert v3["name"] == "orbitune-commercial-safe-v3"
    assert v3_ids[:-3] == v2_ids
    assert v3_ids[-3:] == ["muse_omr_benchmark", "nifc_polish_scores", "nifc_chopin_first_editions"]
    assert v3["split"]["seed"] == v2["split"]["seed"]


def test_muse_omr_is_cc0_score_only_and_resolved_to_exact_install_lock() -> None:
    registry = load_registry(V3)
    source = next(item for item in registry["sources"] if item["id"] == "muse_omr_benchmark")

    assert source["commercial_safe"] is True
    assert source["license"] == "cc0-1.0"
    assert source["kind"] == "huggingface_score_snapshot"
    assert source["repo_id"] == "musegroup/omr_benchmark"
    assert source["revision_policy"] == "resolve-exact-at-install"
    assert source["score_globs"] == ["*.mscz", "**/*.mscz"]
    assert all(".pdf" not in pattern.lower() for pattern in source["allow_patterns"])
    assert "benchmark_dataset.json" in source["allow_patterns"]


def test_hf_score_snapshot_resolves_once_then_reuses_exact_locked_sha(tmp_path, monkeypatch) -> None:
    target = tmp_path / "muse_omr"
    resolved_sha = "0123456789abcdef0123456789abcdef01234567"
    api_calls: list[str] = []
    download_calls: list[dict[str, object]] = []

    fake_hf = ModuleType("huggingface_hub")

    class FakeApi:
        def dataset_info(self, repo_id: str):
            api_calls.append(repo_id)
            return SimpleNamespace(sha=resolved_sha)

    def fake_snapshot_download(**kwargs):
        download_calls.append(dict(kwargs))
        root = Path(str(kwargs["local_dir"]))
        score = root / "data" / "score_0001.mscz"
        score.parent.mkdir(parents=True, exist_ok=True)
        score.write_bytes(b"fake-score")
        (root / "benchmark_dataset.json").write_text("{}\n", encoding="utf-8")
        return str(root)

    fake_hf.HfApi = FakeApi
    fake_hf.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    source = {
        "repo_id": "musegroup/omr_benchmark",
        "revision_policy": "resolve-exact-at-install",
        "allow_patterns": ["*.mscz", "**/*.mscz", "benchmark_dataset.json"],
        "score_globs": ["*.mscz", "**/*.mscz"],
    }

    first = install.install_hf_score_snapshot(source, target)
    assert api_calls == ["musegroup/omr_benchmark"]
    assert first["revision"] == resolved_sha
    assert first["score_files"] == 1
    assert download_calls[-1]["revision"] == resolved_sha
    assert download_calls[-1]["repo_type"] == "dataset"
    assert all(".pdf" not in pattern.lower() for pattern in download_calls[-1]["allow_patterns"])

    lock = json.loads((target / ".orbitune_source_lock.json").read_text(encoding="utf-8"))
    assert lock["revision"] == resolved_sha
    assert lock["repo_id"] == "musegroup/omr_benchmark"

    class MustNotResolveAgain:
        def dataset_info(self, repo_id: str):
            raise AssertionError("locked snapshot must not resolve moving Hub state again")

    fake_hf.HfApi = MustNotResolveAgain
    second = install.install_hf_score_snapshot(source, target)
    assert second["revision"] == resolved_sha
    assert download_calls[-1]["revision"] == resolved_sha


def test_hf_score_snapshot_rejects_unlocked_nonempty_target(tmp_path, monkeypatch) -> None:
    fake_hf = ModuleType("huggingface_hub")
    fake_hf.HfApi = object
    fake_hf.snapshot_download = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    target = tmp_path / "muse_omr"
    target.mkdir()
    (target / "unknown.mscz").write_bytes(b"unprovenanced")

    with pytest.raises(RuntimeError, match="non-empty target has no"):
        install.install_hf_score_snapshot(
            {
                "repo_id": "musegroup/omr_benchmark",
                "revision_policy": "resolve-exact-at-install",
                "allow_patterns": ["**/*.mscz"],
                "score_globs": ["**/*.mscz"],
            },
            target,
        )


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
