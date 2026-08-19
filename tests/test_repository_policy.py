from __future__ import annotations

import json
from pathlib import Path


def test_generated_registries_are_multi_base_shaped() -> None:
    adapters = json.loads(Path("registry/adapters.json").read_text(encoding="utf-8"))
    bases = json.loads(Path("registry/bases.json").read_text(encoding="utf-8"))
    assert isinstance(adapters["adapters"], list)
    assert isinstance(bases["bases"], list)
    assert "base_model" not in adapters


def test_legacy_models_directory_contains_no_weights() -> None:
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin"}
    files = [p for p in Path("models").rglob("*") if p.is_file()]
    assert not [p for p in files if p.suffix in forbidden_suffixes]


def test_base_artifacts_are_managed_only_under_bases() -> None:
    for path in Path("bases").rglob("*"):
        if path.is_file() and path.suffix in {".pt", ".onnx"}:
            assert path.parent.parent == Path("bases")
