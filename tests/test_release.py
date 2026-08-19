import json
from pathlib import Path

import pytest

from orbitune.model import OrbituneConfig, OrbituneGPT
from orbitune.release import BASE_PARAMETER_COUNT, package_base_release, sha256_file, validate_base_checkpoint
from orbitune.tokenizer.vocab import TheoryRemiVocab


def _fixed_checkpoint(path: Path) -> Path:
    model = OrbituneGPT(OrbituneConfig(vocab_size=len(TheoryRemiVocab())))
    assert model.parameter_count() == BASE_PARAMETER_COUNT
    model.save_checkpoint(path)
    return path


def test_package_base_release_writes_hashes_and_runtime_config(tmp_path: Path):
    base = _fixed_checkpoint(tmp_path / "trained.pt")
    web_onnx = tmp_path / "model.onnx"
    web_onnx.write_bytes(b"orbitune-onnx-test-payload")
    out = tmp_path / "release"

    manifest = package_base_release(
        base,
        web_onnx,
        out,
        repository="Unjuno/orbitune",
        release_tag="orbitune-tiny-v0-test",
    )

    checkpoint = out / "orbitune-tiny-v0.pt"
    onnx = out / "orbitune-tiny-v0-web.onnx"
    manifest_path = out / "orbitune-tiny-v0-manifest.json"
    runtime_path = out / "runtime-config.json"
    assert checkpoint.exists() and onnx.exists() and manifest_path.exists() and runtime_path.exists()
    assert manifest["parameters"] == BASE_PARAMETER_COUNT
    assert manifest["artifacts"]["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert manifest["artifacts"]["web_onnx"]["sha256"] == sha256_file(onnx)
    assert manifest["artifacts"]["web_onnx"]["url"].endswith(
        "/releases/download/orbitune-tiny-v0-test/orbitune-tiny-v0-web.onnx"
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert runtime["model_url"] == manifest["artifacts"]["web_onnx"]["url"]
    assert runtime["model_sha256"] == manifest["artifacts"]["web_onnx"]["sha256"]
    assert runtime["execution_providers"] == ["wasm"]


def test_validate_base_checkpoint_rejects_non_v0_shape(tmp_path: Path):
    path = tmp_path / "tiny.pt"
    OrbituneGPT(
        OrbituneConfig(vocab_size=32, max_seq_len=16, n_layer=1, n_embd=32, n_head=4)
    ).save_checkpoint(path)
    with pytest.raises(ValueError, match="not the fixed Orbitune v0 base"):
        validate_base_checkpoint(path)


def test_package_base_release_rejects_empty_onnx(tmp_path: Path):
    base = _fixed_checkpoint(tmp_path / "trained.pt")
    empty = tmp_path / "empty.onnx"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="must not be empty"):
        package_base_release(base, empty, tmp_path / "release")
