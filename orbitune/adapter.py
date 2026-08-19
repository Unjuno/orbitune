from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

REQUIRED_MANIFEST_FIELDS = {
    "artifact_type",
    "name",
    "version",
    "base_model",
    "architecture",
    "tokenizer",
    "adapter_type",
    "rank",
    "target_modules",
    "license",
    "training_data",
    "generation_defaults",
}


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_MANIFEST_FIELDS - manifest.keys())
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if manifest.get("artifact_type") != "orbitune_adapter":
        errors.append("artifact_type must be orbitune_adapter")
    if manifest.get("base_model") != "orbitune-tiny-v0":
        errors.append("base_model must be orbitune-tiny-v0")
    if manifest.get("tokenizer") != "theory-remi-v0":
        errors.append("tokenizer must be theory-remi-v0")
    if manifest.get("adapter_type") != "lora":
        errors.append("adapter_type must be lora")
    rank = manifest.get("rank")
    if not isinstance(rank, int) or rank <= 0:
        errors.append("rank must be a positive integer")
    defaults = manifest.get("generation_defaults", {})
    if isinstance(defaults, dict):
        temperature = defaults.get("temperature")
        if temperature is not None and not (0.6 <= float(temperature) <= 1.2):
            errors.append("generation_defaults.temperature must be between 0.6 and 1.2")
        bars = defaults.get("bars")
        if bars is not None and int(bars) not in {4, 8, 16}:
            errors.append("generation_defaults.bars must be one of 4, 8, 16")
    else:
        errors.append("generation_defaults must be an object")
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, "manifest.json")
        for child in sorted(adapter_dir.rglob("*")):
            if child.is_file():
                zf.write(child, str(child.relative_to(adapter_dir)))
