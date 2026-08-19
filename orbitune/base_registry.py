from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from orbitune.compat import validate_sha256, sha256_file

BASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
REQUIRED_FIELDS = {
    "artifact_type", "id", "display_name", "architecture", "tokenizer",
    "parameter_count", "checkpoint", "web_onnx", "license", "training_data", "tags",
}
OPTIONAL_FIELDS = {"description", "author"}
ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS
MAX_BASE_FILE_BYTES = 95 * 1024 * 1024


def load_base_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Base manifest must be a JSON object")
    return payload


def validate_base_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - manifest.keys())
    unknown = sorted(manifest.keys() - ALLOWED_FIELDS)
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if unknown:
        errors.append(f"unknown Base manifest fields: {', '.join(unknown)}")
    if manifest.get("artifact_type") != "orbitune_base":
        errors.append("artifact_type must be orbitune_base")
    base_id = manifest.get("id")
    if not isinstance(base_id, str) or not BASE_ID_RE.fullmatch(base_id):
        errors.append("id must match ^[a-z0-9][a-z0-9-]*$")
    if not isinstance(manifest.get("display_name"), str) or not manifest.get("display_name"):
        errors.append("display_name is required")
    for field in ("architecture", "tokenizer", "license"):
        if not isinstance(manifest.get(field), str) or not manifest.get(field):
            errors.append(f"{field} is required")
    count = manifest.get("parameter_count")
    if not isinstance(count, int) or count <= 0 or count > 100_000_000:
        errors.append("parameter_count must be an integer in 1..100000000")
    for field in ("checkpoint", "web_onnx"):
        spec = manifest.get(field)
        if not isinstance(spec, dict):
            errors.append(f"{field} must be an object")
            continue
        if set(spec) != {"filename", "sha256", "bytes"}:
            errors.append(f"{field} must contain exactly filename, sha256, bytes")
            continue
        if not isinstance(spec.get("filename"), str) or not spec["filename"]:
            errors.append(f"{field}.filename is required")
        if not isinstance(spec.get("sha256"), str) or not validate_sha256(spec["sha256"]):
            errors.append(f"{field}.sha256 must be a 64-character SHA-256")
        if not isinstance(spec.get("bytes"), int) or not 0 < spec["bytes"] <= MAX_BASE_FILE_BYTES:
            errors.append(f"{field}.bytes must be in 1..{MAX_BASE_FILE_BYTES}")
    training = manifest.get("training_data")
    if not isinstance(training, dict) or training.get("rights_confirmed") is not True:
        errors.append("training_data.rights_confirmed must be true")
    tags = manifest.get("tags")
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        errors.append("tags must be an array of strings")
    return errors


def validate_base_directory(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    readme_path = directory / "README.md"
    if not manifest_path.is_file() or not readme_path.is_file():
        raise ValueError("Base directory requires manifest.json and README.md")
    manifest = load_base_manifest(manifest_path)
    errors = validate_base_manifest(manifest)
    if errors:
        raise ValueError("invalid Base manifest: " + "; ".join(errors))
    if directory.name != manifest["id"]:
        raise ValueError("Base directory name must equal manifest id")
    for field in ("checkpoint", "web_onnx"):
        spec = manifest[field]
        path = directory / spec["filename"]
        if not path.is_file():
            raise ValueError(f"missing Base artifact: {spec['filename']}")
        if path.stat().st_size != spec["bytes"]:
            raise ValueError(f"{field} byte size mismatch")
        if sha256_file(path) != spec["sha256"].lower():
            raise ValueError(f"{field} SHA-256 mismatch")
    if not readme_path.read_text(encoding="utf-8").strip():
        raise ValueError("Base README.md must not be empty")
    return manifest


def discover_base_directories(root: str | Path = "bases") -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(path.parent for path in root.glob("*/manifest.json"))


def build_base_registry(root: str | Path = "bases") -> dict[str, Any]:
    root = Path(root)
    bases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for directory in discover_base_directories(root):
        manifest = validate_base_directory(directory)
        base_id = manifest["id"]
        if base_id in seen:
            raise ValueError(f"duplicate Base id: {base_id}")
        seen.add(base_id)
        bases.append({
            "id": base_id,
            "display_name": manifest["display_name"],
            "description": manifest.get("description", ""),
            "architecture": manifest["architecture"],
            "tokenizer": manifest["tokenizer"],
            "parameter_count": manifest["parameter_count"],
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "checkpoint_url": f"./bases/{base_id}/{manifest['checkpoint']['filename']}",
            "web_onnx_sha256": manifest["web_onnx"]["sha256"],
            "web_onnx_url": f"./bases/{base_id}/{manifest['web_onnx']['filename']}",
            "license": manifest["license"],
            "tags": manifest.get("tags", []),
        })
    bases.sort(key=lambda item: (item["display_name"].lower(), item["id"]))
    return {"schema_version": "0.1.0", "bases": bases}
