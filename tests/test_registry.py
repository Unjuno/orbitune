import json
from pathlib import Path

from orbitune.demo import make_demo_events
from orbitune.midi import write_midi
from orbitune.registry import build_registry, write_registry

BASE_SHA = "a" * 64


def _write_test_safetensors(path: Path, *, base_sha: str = BASE_SHA) -> None:
    metadata = {
        "format": "orbitune-lora-v0",
        "base_sha256": base_sha,
        "rank": "4",
        "alpha": "8.0",
        "dropout": "0.0",
        "target_modules": json.dumps(["q_proj", "v_proj"]),
    }
    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    for layer in range(4):
        for target in ("q_proj", "v_proj"):
            for suffix, shape in (("lora_a", [4, 240]), ("lora_b", [240, 4])):
                size = 4 * 240 * 4
                header[f"blocks.{layer}.attn.{target}.{suffix}"] = {
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, offset + size],
                }
                offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(offset))


def _write_adapter(root: Path, name: str, display_name: str, source: str, *, base_sha: str = BASE_SHA) -> None:
    directory = root / source / name
    directory.mkdir(parents=True)
    manifest = {
        "artifact_type": "orbitune_adapter",
        "name": name,
        "version": "0.1.0",
        "display_name": display_name,
        "description": "test adapter",
        "adapter_family": "style",
        "base_model": "orbitune-base",
        "base_sha256": base_sha,
        "architecture": "orbitune-midi-gpt-v0",
        "parameter_scale": "3m",
        "tokenizer": "theory-remi-v0",
        "adapter_type": "lora",
        "rank": 4,
        "target_modules": ["q_proj", "v_proj"],
        "generation_defaults": {"bpm": 84, "bars": 8, "temperature": 0.85},
        "license": "CC0-1.0",
        "training_data": {"source_type": "original", "license": "CC0-1.0", "rights_confirmed": True},
        "tags": ["test"],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_test_safetensors(directory / "adapter.safetensors", base_sha=base_sha)
    write_midi(make_demo_events(bars=1), directory / "demo.mid", bpm=84)
    (directory / "README.md").write_text("# test\n", encoding="utf-8")


def test_registry_is_built_from_bundled_adapter_directories(tmp_path: Path):
    root = tmp_path / "adapters"
    _write_adapter(root, "community-test-v0", "Community Test", "community")
    _write_adapter(root, "official-test-v0", "Official Test", "official")
    registry = build_registry(root)
    assert [item["id"] for item in registry["adapters"]] == ["official-test-v0", "community-test-v0"]
    assert registry["adapters"][0]["adapter_url"].endswith("official/official-test-v0/adapter.safetensors")
    assert registry["base_model"] == "orbitune-base"
    assert registry["base_sha256"] == BASE_SHA

    out = tmp_path / "registry.json"
    write_registry(out, root)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["base_model"] == "orbitune-base"
    assert written["base_sha256"] == BASE_SHA


def test_registry_rejects_adapters_for_different_base_checkpoints(tmp_path: Path):
    root = tmp_path / "adapters"
    _write_adapter(root, "first-v0", "First", "community", base_sha="a" * 64)
    _write_adapter(root, "second-v0", "Second", "community", base_sha="b" * 64)
    try:
        build_registry(root)
    except ValueError as exc:
        assert "multiple Base checkpoints" in str(exc)
    else:
        raise AssertionError("registry must reject mixed Base hashes")
