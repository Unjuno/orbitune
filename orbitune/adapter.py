from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

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

V0_LORA_RANK = 4
V0_TARGET_MODULES = ["q_proj", "v_proj"]


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_MANIFEST_FIELDS - manifest.keys())
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
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

    if manifest.get("rank") != V0_LORA_RANK:
        errors.append(f"rank must be {V0_LORA_RANK} for Orbitune v0 browser compatibility")
    if manifest.get("target_modules") != V0_TARGET_MODULES:
        errors.append(f"target_modules must be {V0_TARGET_MODULES}")

    defaults = manifest.get("generation_defaults", {})
    if isinstance(defaults, dict):
        temperature = defaults.get("temperature")
        if temperature is None or not (0.6 <= float(temperature) <= 1.2):
            errors.append("generation_defaults.temperature must be between 0.6 and 1.2")
        bars = defaults.get("bars")
        if bars is None or int(bars) not in {4, 8, 16}:
            errors.append("generation_defaults.bars must be one of 4, 8, 16")
        bpm = defaults.get("bpm")
        if bpm is None or not 40 <= int(bpm) <= 220:
            errors.append("generation_defaults.bpm must be between 40 and 220")
    else:
        errors.append("generation_defaults must be an object")

    training = manifest.get("training_data")
    if not isinstance(training, dict):
        errors.append("training_data must be an object")
    else:
        if not training.get("source_type"):
            errors.append("training_data.source_type is required")
        if not training.get("license"):
            errors.append("training_data.license is required")
        if training.get("rights_confirmed") is not True:
            errors.append("training_data.rights_confirmed must be true")

    if not manifest.get("license"):
        errors.append("license is required")
    tags = manifest.get("tags")
    if not isinstance(tags, list):
        errors.append("tags must be an array")
    return errors


def validate_manifest_file(path: str | Path) -> None:
    manifest = load_manifest(path)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid adapter manifest: " + "; ".join(errors))


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
        "generation_defaults": {
            "bpm": bpm,
            "bars": bars,
            "temperature": temperature,
        },
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
        f"# {display_name}\n\n"
        "Orbitune community adapter scaffold.\n\n"
        "## Before publishing\n\n"
        "- Train and place `adapter.safetensors` in this directory.\n"
        "- Add at least one generated demo MIDI, normally `demo.mid`.\n"
        "- Complete `manifest.json`, including license and training-data rights.\n"
        "- Run `orbitune validate-adapter manifest.json`.\n",
        encoding="utf-8",
    )
    return root


def package_adapter(adapter_dir: str | Path, manifest_path: str | Path, out_path: str | Path) -> None:
    adapter_dir = Path(adapter_dir)
    manifest_path = Path(manifest_path)
    out_path = Path(out_path)
    validate_manifest_file(manifest_path)
    weights = adapter_dir / "adapter.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"missing adapter weights: {weights}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, "manifest.json")
        for child in sorted(adapter_dir.rglob("*")):
            if child.is_file() and child.resolve() != manifest_path.resolve():
                zf.write(child, str(child.relative_to(adapter_dir)))
