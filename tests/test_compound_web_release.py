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


def _write_graphs(path: Path, stream: bytes, decoder: bytes) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "stream.onnx").write_bytes(stream)
    (path / "decoder_prefix.onnx").write_bytes(decoder)


def test_release_config_is_built_only_from_exact_reviewed_graph_bytes(tmp_path: Path) -> None:
    stream = b"stream-bytes"; decoder = b"decoder-bytes"
    _write_graphs(tmp_path, stream, decoder)
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
    _write_graphs(tmp_path, stream, decoder)
    release = _release(stream, decoder)
    release["rights"]["redistribution_review"] = "pending"
    with pytest.raises(ValueError, match="not completed"):
        build_runtime_config(release, tmp_path)
    release = _release(stream, decoder)
    release["rights"]["commercial_eligible"] = True
    with pytest.raises(ValueError, match="noncommercial"):
        build_runtime_config(release, tmp_path)


def test_reviewed_premerged_lora_variant_is_appended_with_exact_base_binding(tmp_path: Path) -> None:
    base_stream = b"base-stream"; base_decoder = b"base-decoder"
    lora_stream = b"lora-stream"; lora_decoder = b"lora-decoder"
    base_dir = tmp_path / "base"; lora_dir = tmp_path / "lora"
    _write_graphs(base_dir, base_stream, base_decoder)
    _write_graphs(lora_dir, lora_stream, lora_decoder)
    base = _release(base_stream, base_decoder)
    lora = _release(lora_stream, lora_decoder)
    lora.update({
        "id": "test-style-web-v1",
        "display_name": "Test Style",
        "kind": "lora-premerged",
        "adapter": {"id": "test-style-adapter-v1"},
    })
    lora["graphs"]["stream"]["url"] = "./models/test-style/stream.onnx"
    lora["graphs"]["decoder"]["url"] = "./models/test-style/decoder_prefix.onnx"
    config = build_runtime_config(base, base_dir, additional_releases=[(lora, lora_dir)])
    assert [variant["kind"] for variant in config["variants"]] == ["base", "lora-premerged"]
    assert config["variants"][1]["adapter_id"] == "test-style-adapter-v1"
    assert config["variants"][1]["base_checkpoint_sha256"] == "a" * 64
    assert config["variants"][1]["stream"]["sha256"] == _sha(lora_stream)


def test_premerged_lora_release_cannot_rebind_base_or_skip_review(tmp_path: Path) -> None:
    stream = b"stream"; decoder = b"decoder"
    base_dir = tmp_path / "base"; lora_dir = tmp_path / "lora"
    _write_graphs(base_dir, stream, decoder); _write_graphs(lora_dir, stream, decoder)
    base = _release(stream, decoder)
    lora = _release(stream, decoder)
    lora.update({"id": "lora", "kind": "lora-premerged", "adapter": {"id": "adapter-v1"}})
    lora["base"]["checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="canonical published Base"):
        build_runtime_config(base, base_dir, additional_releases=[(lora, lora_dir)])

    lora = _release(stream, decoder)
    lora.update({"id": "lora", "kind": "lora-premerged", "adapter": {"id": "adapter-v1"}})
    lora["rights"]["redistribution_review"] = "pending"
    with pytest.raises(ValueError, match="not completed"):
        build_runtime_config(base, base_dir, additional_releases=[(lora, lora_dir)])


def test_premerged_lora_release_requires_adapter_identity(tmp_path: Path) -> None:
    stream = b"stream"; decoder = b"decoder"
    base_dir = tmp_path / "base"; lora_dir = tmp_path / "lora"
    _write_graphs(base_dir, stream, decoder); _write_graphs(lora_dir, stream, decoder)
    base = _release(stream, decoder)
    lora = _release(stream, decoder)
    lora.update({"id": "lora", "kind": "lora-premerged"})
    with pytest.raises(ValueError, match="adapter.id"):
        build_runtime_config(base, base_dir, additional_releases=[(lora, lora_dir)])
