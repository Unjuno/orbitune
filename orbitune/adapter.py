from __future__ import annotations

import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

from orbitune.midi import read_midi

REQUIRED_MANIFEST_FIELDS = {
    "artifact_type",
    "name",
    "version",
    "display_name",
    "base_model",
    "architecture",
    "parameter_scale",
    "tokenizer",
    "adapter_type",
    "rank",
    "target_modules",
    "license",
    "training_data",
    "generation_defaults",
    "tags",
}
OPTIONAL_MANIFEST_FIELDS = {"description", "adapter_family"}
ALLOWED_MANIFEST_FIELDS = REQUIRED_MANIFEST_FIELDS | OPTIONAL_MANIFEST_FIELDS
V0_LORA_RANK = 4
V0_HIDDEN_SIZE = 240
V0_LAYERS = 4
V0_TARGET_MODULES = ["q_proj", "v_proj"]
V0_ADAPTER_FAMILIES = {"style", "genre", "texture-like", "control", "instrument", "experimental"}
ADAPTER_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*-v[0-9]+$")


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("adapter manifest must be a JSON object")
    return payload


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_MANIFEST_FIELDS - manifest.keys())
    unknown = sorted(manifest.keys() - ALLOWED_MANIFEST_FIELDS)
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if unknown:
        errors.append(f"unknown manifest fields: {', '.join(unknown)}")

    expected = {
        "artifact_type": "orbitune_adapter",
        "base_model": "orbitune-tiny-v0",
        "architecture": "orbitune-midi-gpt-v0",
        "parameter_scale": "3m",
        "tokenizer": "theory-remi-v0",
        "adapter_type": "lora",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            errors.append(f"{key} must be {value}")

    name = manifest.get("name")
    if not isinstance(name, str) or not ADAPTER_NAME_RE.fullmatch(name):
        errors.append("name must match ^[a-z0-9][a-z0-9-]*-v[0-9]+$")
    if not isinstance(manifest.get("version"), str) or not manifest.get("version"):
        errors.append("version must be a non-empty string")
    if not isinstance(manifest.get("display_name"), str) or not manifest.get("display_name"):
        errors.append("display_name must be a non-empty string")
    description = manifest.get("description")
    if description is not None and not isinstance(description, str):
        errors.append("description must be a string")
    family = manifest.get("adapter_family")
    if family is not None and family not in V0_ADAPTER_FAMILIES:
        errors.append(f"adapter_family must be one of {sorted(V0_ADAPTER_FAMILIES)}")

    if manifest.get("rank") != V0_LORA_RANK:
        errors.append(f"rank must be {V0_LORA_RANK} for Orbitune v0 browser compatibility")
    if manifest.get("target_modules") != V0_TARGET_MODULES:
        errors.append(f"target_modules must be {V0_TARGET_MODULES}")

    defaults = manifest.get("generation_defaults")
    if isinstance(defaults, dict):
        allowed_default_fields = {"bpm", "bars", "temperature", "top_p"}
        extra_defaults = sorted(defaults.keys() - allowed_default_fields)
        if extra_defaults:
            errors.append(f"unknown generation_defaults fields: {', '.join(extra_defaults)}")
        try:
            temperature = float(defaults.get("temperature"))
            if not 0.6 <= temperature <= 1.2:
                errors.append("generation_defaults.temperature must be between 0.6 and 1.2")
        except (TypeError, ValueError):
            errors.append("generation_defaults.temperature must be a number between 0.6 and 1.2")
        try:
            bars = int(defaults.get("bars"))
            if bars not in {4, 8, 16}:
                errors.append("generation_defaults.bars must be one of 4, 8, 16")
        except (TypeError, ValueError):
            errors.append("generation_defaults.bars must be one of 4, 8, 16")
        try:
            bpm = int(defaults.get("bpm"))
            if not 40 <= bpm <= 220:
                errors.append("generation_defaults.bpm must be between 40 and 220")
        except (TypeError, ValueError):
            errors.append("generation_defaults.bpm must be between 40 and 220")
        if "top_p" in defaults:
            try:
                top_p = float(defaults["top_p"])
                if not 0.5 <= top_p <= 1.0:
                    errors.append("generation_defaults.top_p must be between 0.5 and 1.0")
            except (TypeError, ValueError):
                errors.append("generation_defaults.top_p must be between 0.5 and 1.0")
    else:
        errors.append("generation_defaults must be an object")

    training = manifest.get("training_data")
    if isinstance(training, dict):
        allowed_training_fields = {"source_type", "license", "num_files", "num_tokens", "rights_confirmed", "notes"}
        extra_training = sorted(training.keys() - allowed_training_fields)
        if extra_training:
            errors.append(f"unknown training_data fields: {', '.join(extra_training)}")
        if not isinstance(training.get("source_type"), str) or not training.get("source_type"):
            errors.append("training_data.source_type is required")
        if not isinstance(training.get("license"), str) or not training.get("license"):
            errors.append("training_data.license is required")
        if training.get("rights_confirmed") is not True:
            errors.append("training_data.rights_confirmed must be true")
        for key in ("num_files", "num_tokens"):
            if key in training and (not isinstance(training[key], int) or training[key] < 0):
                errors.append(f"training_data.{key} must be a non-negative integer")
        if "notes" in training and not isinstance(training["notes"], str):
            errors.append("training_data.notes must be a string")
    else:
        errors.append("training_data must be an object")

    if not isinstance(manifest.get("license"), str) or not manifest.get("license"):
        errors.append("license is required")
    tags = manifest.get("tags")
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        errors.append("tags must be an array of strings")
    elif len(tags) != len(set(tags)):
        errors.append("tags must not contain duplicates")
    return errors


def validate_manifest_file(path: str | Path) -> None:
    manifest = load_manifest(path)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid adapter manifest: " + "; ".join(errors))


def _read_safetensors_header(path: str | Path) -> tuple[dict[str, Any], int, int]:
    path = Path(path)
    size = path.stat().st_size
    if size < 8:
        raise ValueError("adapter.safetensors is too small")
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        if header_length <= 0 or 8 + header_length > size:
            raise ValueError("invalid Safetensors header length")
        header_bytes = handle.read(header_length)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("invalid Safetensors header object")
    return header, 8 + header_length, size


def validate_adapter_weights(path: str | Path) -> dict[str, str]:
    header, data_start, file_size = _read_safetensors_header(path)
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise ValueError("adapter Safetensors metadata is missing")
    if metadata.get("format") != "orbitune-lora-v0":
        raise ValueError("adapter Safetensors format must be orbitune-lora-v0")
    try:
        rank = int(metadata.get("rank", ""))
        alpha = float(metadata.get("alpha", ""))
        dropout = float(metadata.get("dropout", "0"))
        targets = json.loads(metadata.get("target_modules", "[]"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid adapter Safetensors metadata") from exc
    if rank != V0_LORA_RANK:
        raise ValueError(f"adapter Safetensors rank must be {V0_LORA_RANK}")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("adapter Safetensors alpha must be positive and finite")
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("adapter Safetensors dropout must be in [0, 1)")
    if targets != V0_TARGET_MODULES:
        raise ValueError(f"adapter Safetensors targets must be {V0_TARGET_MODULES}")

    tensor_specs = {key: value for key, value in header.items() if key != "__metadata__"}
    expected: dict[str, tuple[list[int], int]] = {}
    for layer in range(V0_LAYERS):
        for target in V0_TARGET_MODULES:
            prefix = f"blocks.{layer}.attn.{target}"
            expected[f"{prefix}.lora_a"] = ([V0_LORA_RANK, V0_HIDDEN_SIZE], V0_LORA_RANK * V0_HIDDEN_SIZE * 4)
            expected[f"{prefix}.lora_b"] = ([V0_HIDDEN_SIZE, V0_LORA_RANK], V0_HIDDEN_SIZE * V0_LORA_RANK * 4)
    if set(tensor_specs) != set(expected):
        missing = sorted(set(expected) - set(tensor_specs))
        extra = sorted(set(tensor_specs) - set(expected))
        raise ValueError(f"adapter tensor set mismatch: missing={missing}, extra={extra}")

    ranges: list[tuple[int, int, str]] = []
    data_size = file_size - data_start
    for name, (shape, expected_bytes) in expected.items():
        spec = tensor_specs[name]
        if not isinstance(spec, dict) or spec.get("dtype") != "F32" or spec.get("shape") != shape:
            raise ValueError(f"invalid tensor spec for {name}; expected F32 {shape}")
        offsets = spec.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(not isinstance(value, int) for value in offsets):
            raise ValueError(f"invalid data_offsets for {name}")
        start, end = offsets
        if start < 0 or end < start or end > data_size or end - start != expected_bytes:
            raise ValueError(f"invalid data range for {name}")
        ranges.append((start, end, name))
    ranges.sort()
    previous_end = 0
    for start, end, name in ranges:
        if start != previous_end:
            raise ValueError(f"Safetensors data must be contiguous; gap/overlap before {name}")
        previous_end = end
    if previous_end != data_size:
        raise ValueError("Safetensors contains trailing or unreferenced tensor data")
    return {str(key): str(value) for key, value in metadata.items()}


def validate_adapter_directory(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    weights_path = directory / "adapter.safetensors"
    demo_path = directory / "demo.mid"
    readme_path = directory / "README.md"
    for path in (manifest_path, weights_path, demo_path, readme_path):
        if not path.is_file():
            raise ValueError(f"missing required adapter file: {path.name}")
    validate_manifest_file(manifest_path)
    manifest = load_manifest(manifest_path)
    metadata = validate_adapter_weights(weights_path)
    if int(metadata["rank"]) != int(manifest["rank"]):
        raise ValueError("manifest rank does not match adapter Safetensors metadata")
    if json.loads(metadata["target_modules"]) != manifest["target_modules"]:
        raise ValueError("manifest target_modules do not match adapter Safetensors metadata")
    events = read_midi(demo_path)
    if not events:
        raise ValueError("demo.mid must contain at least one note event")
    if not readme_path.read_text(encoding="utf-8").strip():
        raise ValueError("README.md must not be empty")
    return manifest


def create_adapter_scaffold(
    directory: str | Path,
    *,
    name: str,
    display_name: str,
    adapter_family: str = "style",
    rank: int = V0_LORA_RANK,
    bpm: int = 84,
    bars: int = 8,
    temperature: float = 0.85,
) -> Path:
    if rank != V0_LORA_RANK:
        raise ValueError(f"Orbitune v0 adapters must use rank {V0_LORA_RANK}")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "artifact_type": "orbitune_adapter",
        "name": name,
        "version": "0.1.0",
        "display_name": display_name,
        "description": "TODO: describe the generation tendency of this adapter.",
        "adapter_family": adapter_family,
        "base_model": "orbitune-tiny-v0",
        "architecture": "orbitune-midi-gpt-v0",
        "parameter_scale": "3m",
        "tokenizer": "theory-remi-v0",
        "adapter_type": "lora",
        "rank": V0_LORA_RANK,
        "target_modules": V0_TARGET_MODULES,
        "generation_defaults": {"bpm": bpm, "bars": bars, "temperature": temperature},
        "license": "TODO",
        "training_data": {
            "source_type": "user_provided_midi",
            "license": "TODO",
            "num_files": 0,
            "num_tokens": 0,
            "rights_confirmed": False,
            "notes": "Set rights_confirmed=true only after checking the training-data rights.",
        },
        "tags": [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "README.md").write_text(
        f"# {display_name}\n\nOrbitune community adapter scaffold.\n\n"
        "## Before publishing\n\n"
        "- Train and place `adapter.safetensors` in this directory.\n"
        "- Add a generated `demo.mid`.\n"
        "- Complete `manifest.json`, including license and training-data rights.\n"
        "- Run `orbitune validate-adapter manifest.json` and CI validation.\n",
        encoding="utf-8",
    )
    return root


def package_adapter(adapter_dir: str | Path, manifest_path: str | Path, out_path: str | Path) -> None:
    adapter_dir = Path(adapter_dir)
    manifest_path = Path(manifest_path)
    out_path = Path(out_path)
    if manifest_path.resolve() != (adapter_dir / "manifest.json").resolve():
        raise ValueError("manifest path must be adapter_dir/manifest.json")
    validate_adapter_directory(adapter_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for child in sorted(adapter_dir.rglob("*")):
            if child.is_file():
                zf.write(child, str(child.relative_to(adapter_dir)))
