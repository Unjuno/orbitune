from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.build_compound_web_runtime_config import build_runtime_config


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _release(stream: bytes, decoder: bytes) -> dict:
    return {
        "schema_version": 1,
        "status": "approved_for_publication",
        "id": "test-base-web",
        "display_name": "Test Base",
        "base": {
            "model_id": "test-base", "checkpoint_sha256": "a" * 64,
            "architecture": "orbitune-compound-hierarchical-gpt-v1",
            "tokenizer": "orbitune-compound-v0-experimental",
        },
        "runtime": {"abi": "native-stream-state+decoder-prefix-v2", "execution_providers": ["wasm"]},
        "graphs": {
            "stream": {"filename": "stream.onnx", "url": "./models/test/stream.onnx", "sha256": _sha(stream), "bytes": len(stream)},
            "decoder": {"filename": "decoder_prefix.onnx", "url": "./models/test/decoder_prefix.onnx", "sha256": _sha(decoder), "bytes": len(decoder)},
        },
        "rights": {"distribution_scope": "research-noncommercial", "commercial_eligible": False, "redistribution_review": "completed"},
        "publication": {"runtime_publication_status": "runtime_model_published"},
    }


def test_release_config_is_built_only_from_exact_reviewed_graph_bytes(tmp_path: Path) -> None:
    stream = b"stream-bytes"; decoder = b"decoder-bytes"
    (tmp_path / "stream.onnx").write_bytes(stream)
    (tmp_path / "decoder_prefix.onnx").write_bytes(decoder)
    config = build_runtime_config(_release(stream, decoder), tmp_path)
    assert config["publication_status"] == "runtime_model_published"
    assert config["redistribution_review"] == "completed"
    assert config["commercial_eligible"] is False
    assert config["variants"][0]["kind"] == "base"
    assert config["variants"][0]["stream"]["sha256"] == _sha(stream)


def test_release_config_rejects_changed_graph_bytes(tmp_path: Path) -> None:
    stream = b"stream-bytes"; decoder = b"decoder-bytes"
    release = _release(stream, decoder)
    (tmp_path / "stream.onnx").write_bytes(stream + b"changed")
    (tmp_path / "decoder_prefix.onnx").write_bytes(decoder)
    with pytest.raises(ValueError, match="byte-size mismatch|SHA-256 mismatch"):
        build_runtime_config(release, tmp_path)


def test_release_config_rejects_unreviewed_or_commercialized_release(tmp_path: Path) -> None:
    stream = b"stream"; decoder = b"decoder"
    (tmp_path / "stream.onnx").write_bytes(stream)
    (tmp_path / "decoder_prefix.onnx").write_bytes(decoder)
    release = _release(stream, decoder)
    release["rights"]["redistribution_review"] = "pending"
    with pytest.raises(ValueError, match="not completed"):
        build_runtime_config(release, tmp_path)
    release = _release(stream, decoder)
    release["rights"]["commercial_eligible"] = True
    with pytest.raises(ValueError, match="noncommercial"):
        build_runtime_config(release, tmp_path)
