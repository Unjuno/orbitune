import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orbitune.base_registry import _reference_web_shape, build_base_registry, validate_base_manifest
from orbitune.compat import (
    ARCHITECTURE_ABI,
    REFERENCE_MAX_SEQ_LEN,
    REFERENCE_N_EMBD,
    REFERENCE_N_HEAD,
    REFERENCE_N_LAYER,
    REFERENCE_PARAMETER_COUNT,
    TOKENIZER_ABI,
    sha256_file,
)
from orbitune.model import OrbituneConfig, OrbituneGPT


def _manifest() -> dict[str, object]:
    return {
        "artifact_type": "orbitune_base",
        "id": "test-base",
        "display_name": "Test Base",
        "architecture": ARCHITECTURE_ABI,
        "tokenizer": TOKENIZER_ABI,
        "parameter_count": 123,
        "checkpoint": {"filename": "model.pt", "sha256": "0" * 64, "bytes": 1},
        "web_onnx": {"filename": "web.onnx", "sha256": "1" * 64, "bytes": 1},
        "license": "Apache-2.0",
        "training_data": {
            "source_type": "test",
            "license": "CC0-1.0",
            "rights_confirmed": True,
        },
        "tags": [],
    }


def test_base_manifest_rejects_artifact_path_traversal() -> None:
    manifest = _manifest()
    manifest["checkpoint"] = {"filename": "../model.pt", "sha256": "0" * 64, "bytes": 1}
    errors = validate_base_manifest(manifest)
    assert any("simple file name" in error for error in errors)


def test_base_manifest_requires_training_provenance_fields_and_unique_tags() -> None:
    manifest = _manifest()
    manifest["training_data"] = {"rights_confirmed": True}
    manifest["tags"] = ["x", "x"]
    errors = validate_base_manifest(manifest)
    assert any("source_type" in error for error in errors)
    assert any("license" in error for error in errors)
    assert any("duplicates" in error for error in errors)


def _write_tiny_current_abi_base(root: Path) -> tuple[Path, dict[str, object]]:
    base = root / "test-base"
    base.mkdir(parents=True)
    checkpoint = base / "model.pt"
    onnx = base / "web.onnx"
    model = OrbituneGPT(
        OrbituneConfig(vocab_size=32, max_seq_len=16, n_layer=1, n_embd=32, n_head=4, dropout=0.0)
    )
    model.save_checkpoint(checkpoint)
    onnx.write_bytes(b"not-a-reference-web-graph")
    manifest = _manifest()
    manifest["parameter_count"] = model.parameter_count()
    manifest["checkpoint"] = {
        "filename": "model.pt",
        "sha256": sha256_file(checkpoint),
        "bytes": checkpoint.stat().st_size,
    }
    manifest["web_onnx"] = {
        "filename": "web.onnx",
        "sha256": sha256_file(onnx),
        "bytes": onnx.stat().st_size,
    }
    (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (base / "README.md").write_text("test", encoding="utf-8")
    return base, manifest


def test_registry_rejects_manifest_parameter_count_that_disagrees_with_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "bases"
    base, manifest = _write_tiny_current_abi_base(root)
    manifest["parameter_count"] = int(manifest["parameter_count"]) + 1
    (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="parameter_count"):
        build_base_registry(root)


def test_registry_does_not_mark_nonreference_shape_as_web_compatible(tmp_path: Path) -> None:
    root = tmp_path / "bases"
    _write_tiny_current_abi_base(root)
    registry = build_base_registry(root)
    assert registry["bases"][0]["web_runtime_compatible"] is False


def test_reference_web_shape_requires_exact_reference_contract() -> None:
    reference = SimpleNamespace(
        config=SimpleNamespace(
            max_seq_len=REFERENCE_MAX_SEQ_LEN,
            n_layer=REFERENCE_N_LAYER,
            n_embd=REFERENCE_N_EMBD,
            n_head=REFERENCE_N_HEAD,
        )
    )
    manifest = _manifest()
    manifest["parameter_count"] = REFERENCE_PARAMETER_COUNT
    assert _reference_web_shape(reference, manifest) is True
    reference.config.n_embd = 32
    assert _reference_web_shape(reference, manifest) is False
