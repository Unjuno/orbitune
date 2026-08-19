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

    rank = manifest.get("rank")
    if not isinstance(rank, int) or not 1 <= rank <= 32:
        errors.append("rank must be an integer between 1 and 32")
    targets = manifest.get("target_modules")
    if not isinstance(targets, list) or not targets or any(not isinstance(item, str) for item in targets):
        errors.append("target_modules must be a non-empty string array")
    elif any(item not in {"q_proj", "v_proj"} for item in targets):
        errors.append("v0 target_modules may only contain q_proj and v_proj")

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
