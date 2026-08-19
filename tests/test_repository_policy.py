from __future__ import annotations

import json
from pathlib import Path


def test_registry_targets_v0_base() -> None:
    registry = json.loads(Path("registry/adapters.json").read_text(encoding="utf-8"))
    assert registry["base_model"] == "orbitune-tiny-v0"
    assert registry["tokenizer"] == "theory-remi-v0"
    assert isinstance(registry["adapters"], list)


def test_models_directory_contains_no_weights() -> None:
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin"}
    files = [p for p in Path("models").rglob("*") if p.is_file()]
    assert not [p for p in files if p.suffix in forbidden_suffixes]
