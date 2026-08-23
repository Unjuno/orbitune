import json
from pathlib import Path

from orbitune.base_registry import build_base_registry, validate_base_manifest
from orbitune.compat import ARCHITECTURE_ABI, TOKENIZER_ABI, sha256_file


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


def test_registry_marks_only_current_theory_remi_web_abi_compatible(tmp_path: Path) -> None:
    root = tmp_path / "bases"
    base = root / "test-base"
    base.mkdir(parents=True)
    checkpoint = base / "model.pt"
    onnx = base / "web.onnx"
    checkpoint.write_bytes(b"x")
    onnx.write_bytes(b"y")
    manifest = _manifest()
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
    registry = build_base_registry(root)
    assert registry["bases"][0]["web_runtime_compatible"] is True

    manifest["tokenizer"] = "future-compound-abi"
    (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    registry = build_base_registry(root)
    assert registry["bases"][0]["web_runtime_compatible"] is False
