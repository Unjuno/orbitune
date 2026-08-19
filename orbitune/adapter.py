from __future__ import annotations

import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

from orbitune.compat import ADAPTER_FORMAT_ABI, ARCHITECTURE_ABI, BASE_MODEL_ID, TOKENIZER_ABI, validate_sha256
from orbitune.midi import read_midi

REQUIRED_MANIFEST_FIELDS = {
    "artifact_type", "name", "version", "display_name", "base_model", "base_sha256",
    "architecture", "parameter_scale", "tokenizer", "adapter_type", "rank",
    "target_modules", "license", "training_data", "generation_defaults", "tags",
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
        "base_model": BASE_MODEL_ID,
        "architecture": ARCHITECTURE_ABI,
        "parameter_scale": "3m",
        "tokenizer": TOKENIZER_ABI,
        "adapter_type": "lora",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            errors.append(f"{key} must be {value}")

    base_sha = manifest.get("base_sha256")
    if not isinstance(base_sha, str) or not validate_sha256(base_sha):
        errors.append("base_sha256 must be the exact 64-character SHA-256 of the immutable Orbitune Base checkpoint")

    name = manifest.get("name")
    if not isinstance(name, str) or not ADAPTER_NAME_RE.fullmatch(name):
        errors.append("name must match ^[a-z0-9][a-z0-9-]*-v[0-9]+$")
    if not isinstance(manifest.get("version"), str) or not manifest.get("version"):
        errors.append("version must be a non-empty string")
    if not isinstance(manifest.get("display_name"), str) or not manifest.get("display_name"):
        errors.append("display_name must be a non-empty string")
    if manifest.get("rank") != V0_LORA_RANK:
        errors.append(f"rank must be {V0_LORA_RANK} for the current adapter ABI")
    if manifest.get("target_modules") != V0_TARGET_MODULES:
        errors.append(f"target_modules must be {V0_TARGET_MODULES}")
    family = manifest.get("adapter_family")
    if family is not None and family not in V0_ADAPTER_FAMILIES:
        errors.append(f"adapter_family must be one of {sorted(V0_ADAPTER_FAMILIES)}")

    defaults = manifest.get("generation_defaults")
    if not isinstance(defaults, dict):
        errors.append("generation_defaults must be an object")
    else:
        extra = sorted(defaults.keys() - {"bpm", "bars", "temperature", "top_p"})
        if extra:
            errors.append(f"unknown generation_defaults fields: {', '.join(extra)}")
        try:
            if not 0.6 <= float(defaults.get("temperature")) <= 1.2:
                errors.append("generation_defaults.temperature must be between 0.6 and 1.2")
        except (TypeError, ValueError):
            errors.append("generation_defaults.temperature must be a number between 0.6 and 1.2")
        if defaults.get("bars") not in {4, 8, 16}:
            errors.append("generation_defaults.bars must be one of 4, 8, 16")
        bpm = defaults.get("bpm")
        if not isinstance(bpm, int) or not 40 <= bpm <= 220:
            errors.append("generation_defaults.bpm must be between 40 and 220")
        if "top_p" in defaults:
            try:
                if not 0.5 <= float(defaults["top_p"]) <= 1.0:
                    errors.append("generation_defaults.top_p must be between 0.5 and 1.0")
            except (TypeError, ValueError):
                errors.append("generation_defaults.top_p must be between 0.5 and 1.0")

    training = manifest.get("training_data")
    if not isinstance(training, dict):
        errors.append("training_data must be an object")
    else:
        extra = sorted(training.keys() - {"source_type", "license", "num_files", "num_tokens", "rights_confirmed", "notes"})
        if extra:
            errors.append(f"unknown training_data fields: {', '.join(extra)}")
        if not isinstance(training.get("source_type"), str) or not training.get("source_type"):
            errors.append("training_data.source_type is required")
        if not isinstance(training.get("license"), str) or not training.get("license"):
            errors.append("training_data.license is required")
        if training.get("rights_confirmed") is not True:
            errors.append("training_data.rights_confirmed must be true")
        for key in ("num_files", "num_tokens"):
            if key in training and (not isinstance(training[key], int) or training[key] < 0):
                errors.append(f"training_data.{key} must be a non-negative integer")

    if not isinstance(manifest.get("license"), str) or not manifest.get("license"):
        errors.append("license is required")
    tags = manifest.get("tags")
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        errors.append("tags must be an array of strings")
    elif len(tags) != len(set(tags)):
        errors.append("tags must not contain duplicates")
    return errors


def validate_manifest_file(path: str | Path) -> None:
    errors = validate_manifest(load_manifest(path))
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
        header = json.loads(handle.read(header_length).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("invalid Safetensors header object")
    return header, 8 + header_length, size


def validate_adapter_weights(path: str | Path) -> dict[str, str]:
    header, data_start, file_size = _read_safetensors_header(path)
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        raise ValueError("adapter Safetensors metadata is missing")
    if metadata.get("format") != ADAPTER_FORMAT_ABI:
        raise ValueError(f"adapter Safetensors format must be {ADAPTER_FORMAT_ABI}")
    base_sha = str(metadata.get("base_sha256", "")).lower()
    if not validate_sha256(base_sha):
        raise ValueError("adapter Safetensors metadata must include a valid base_sha256")
    try:
        rank = int(metadata.get("rank", ""))
        alpha = float(metadata.get("alpha", ""))
        dropout = float(metadata.get("dropout", "0"))
        targets = json.loads(metadata.get("target_modules", "[]"))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid adapter Safetensors metadata") from exc
    if rank != V0_LORA_RANK or targets != V0_TARGET_MODULES:
        raise ValueError("adapter Safetensors rank/targets do not match the current adapter ABI")
    if not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("invalid adapter alpha/dropout metadata")

    specs = {key: value for key, value in header.items() if key != "__metadata__"}
    expected: dict[str, tuple[list[int], int]] = {}
    for layer in range(V0_LAYERS):
        for target in V0_TARGET_MODULES:
            prefix = f"blocks.{layer}.attn.{target}"
            expected[f"{prefix}.lora_a"] = ([V0_LORA_RANK, V0_HIDDEN_SIZE], V0_LORA_RANK * V0_HIDDEN_SIZE * 4)
            expected[f"{prefix}.lora_b"] = ([V0_HIDDEN_SIZE, V0_LORA_RANK], V0_HIDDEN_SIZE * V0_LORA_RANK * 4)
    if set(specs) != set(expected):
        raise ValueError("adapter tensor set mismatch")
    data_size = file_size - data_start
    ranges: list[tuple[int, int]] = []
    for name, (shape, expected_bytes) in expected.items():
        spec = specs[name]
        if not isinstance(spec, dict) or spec.get("dtype") != "F32" or spec.get("shape") != shape:
            raise ValueError(f"invalid tensor spec for {name}")
        offsets = spec.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"invalid data_offsets for {name}")
        start, end = offsets
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end - start != expected_bytes or end > data_size:
            raise ValueError(f"invalid data range for {name}")
        ranges.append((start, end))
    ranges.sort()
    previous = 0
    for start, end in ranges:
        if start != previous:
            raise ValueError("Safetensors tensor data must be contiguous")
        previous = end
    if previous != data_size:
        raise ValueError("Safetensors contains trailing data")
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
    if metadata["base_sha256"].lower() != manifest["base_sha256"].lower():
        raise ValueError("manifest base_sha256 does not match adapter Safetensors metadata")
    if int(metadata["rank"]) != int(manifest["rank"]):
        raise ValueError("manifest rank does not match adapter Safetensors metadata")
    if json.loads(metadata["target_modules"]) != manifest["target_modules"]:
        raise ValueError("manifest target_modules do not match adapter Safetensors metadata")
    if not read_midi(demo_path):
        raise ValueError("demo.mid must contain at least one note event")
    if not readme_path.read_text(encoding="utf-8").strip():
        raise ValueError("README.md must not be empty")
    return manifest


def create_adapter_scaffold(directory: str | Path, *, name: str, display_name: str, adapter_family: str = "style", rank: int = V0_LORA_RANK, bpm: int = 84, bars: int = 8, temperature: float = 0.85) -> Path:
    if rank != V0_LORA_RANK:
        raise ValueError(f"current adapter ABI requires rank {V0_LORA_RANK}")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "artifact_type": "orbitune_adapter",
        "name": name,
        "version": "0.1.0",
        "display_name": display_name,
        "description": "TODO: describe the generation tendency of this adapter.",
        "adapter_family": adapter_family,
        "base_model": BASE_MODEL_ID,
        "base_sha256": "TODO: copy the exact Base checkpoint SHA-256 from training output",
        "architecture": ARCHITECTURE_ABI,
        "parameter_scale": "3m",
        "tokenizer": TOKENIZER_ABI,
        "adapter_type": "lora",
        "rank": V0_LORA_RANK,
        "target_modules": V0_TARGET_MODULES,
        "generation_defaults": {"bpm": bpm, "bars": bars, "temperature": temperature},
        "license": "TODO",
        "training_data": {"source_type": "user_provided_midi", "license": "TODO", "num_files": 0, "num_tokens": 0, "rights_confirmed": False, "notes": "Confirm rights before publishing."},
        "tags": [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "README.md").write_text(
        f"# {display_name}\n\nOrbitune community adapter scaffold.\n\n"
        "Before publishing, train adapter.safetensors, add demo.mid, copy the exact Base SHA-256 into manifest.json, complete rights/license fields, and run validation.\n",
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
