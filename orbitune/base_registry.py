from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from orbitune.compat import (
    ARCHITECTURE_ABI,
    REFERENCE_MAX_SEQ_LEN,
    REFERENCE_N_EMBD,
    REFERENCE_N_HEAD,
    REFERENCE_N_LAYER,
    REFERENCE_PARAMETER_COUNT,
    TOKENIZER_ABI,
    sha256_file,
    validate_sha256,
)

BASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
REQUIRED_FIELDS = {
    "artifact_type", "id", "display_name", "architecture", "tokenizer",
    "parameter_count", "checkpoint", "web_onnx", "license", "training_data", "tags",
}
OPTIONAL_FIELDS = {"description", "author"}
ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS
MAX_BASE_FILE_BYTES = 95 * 1024 * 1024
TRAINING_DATA_FIELDS = {"source_type", "license", "rights_confirmed", "notes"}


def load_base_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Base manifest must be a JSON object")
    return payload


def _safe_artifact_filename(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return path.name == value and value not in {".", ".."}


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
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or count > 100_000_000:
        errors.append("parameter_count must be an integer in 1..100000000")
    for field in ("checkpoint", "web_onnx"):
        spec = manifest.get(field)
        if not isinstance(spec, dict):
            errors.append(f"{field} must be an object")
            continue
        if set(spec) != {"filename", "sha256", "bytes"}:
            errors.append(f"{field} must contain exactly filename, sha256, bytes")
            continue
        if not _safe_artifact_filename(spec.get("filename")):
            errors.append(f"{field}.filename must be a simple file name without path separators")
        if not isinstance(spec.get("sha256"), str) or not validate_sha256(spec["sha256"]):
            errors.append(f"{field}.sha256 must be a 64-character SHA-256")
        size = spec.get("bytes")
        if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_BASE_FILE_BYTES:
            errors.append(f"{field}.bytes must be in 1..{MAX_BASE_FILE_BYTES}")
    training = manifest.get("training_data")
    if not isinstance(training, dict):
        errors.append("training_data must be an object")
    else:
        unknown_training = sorted(training.keys() - TRAINING_DATA_FIELDS)
        missing_training = sorted({"source_type", "license", "rights_confirmed"} - training.keys())
        if missing_training:
            errors.append(f"training_data missing required fields: {', '.join(missing_training)}")
        if unknown_training:
            errors.append(f"unknown training_data fields: {', '.join(unknown_training)}")
        if not isinstance(training.get("source_type"), str) or not training.get("source_type"):
            errors.append("training_data.source_type is required")
        if not isinstance(training.get("license"), str) or not training.get("license"):
            errors.append("training_data.license is required")
        if training.get("rights_confirmed") is not True:
            errors.append("training_data.rights_confirmed must be true")
        if "notes" in training and not isinstance(training["notes"], str):
            errors.append("training_data.notes must be a string")
    tags = manifest.get("tags")
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        errors.append("tags must be an array of strings")
    elif len(tags) != len(set(tags)):
        errors.append("tags must not contain duplicates")
    return errors


def _validate_current_checkpoint(path: Path, manifest: dict[str, Any]):
    """Validate checkpoint contents for the currently implemented Python ABI.

    Unknown/future ABIs remain registry-extensible, but a Base claiming the
    current OrbituneGPT/Theory-REMI ABI must actually deserialize as that model
    and agree with its manifest parameter count.
    """
    if manifest["architecture"] != ARCHITECTURE_ABI or manifest["tokenizer"] != TOKENIZER_ABI:
        return None
    from orbitune.model import OrbituneGPT

    try:
        model = OrbituneGPT.load_checkpoint(path, map_location="cpu").eval()
    except Exception as exc:
        raise ValueError("Base claims the current Orbitune ABI but checkpoint is incompatible") from exc
    actual_count = model.parameter_count()
    if actual_count != manifest["parameter_count"]:
        raise ValueError(
            f"checkpoint parameter_count {actual_count} does not match manifest {manifest['parameter_count']}"
        )
    return model


def _reference_web_shape(model: object | None, manifest: dict[str, Any]) -> bool:
    if model is None or manifest["parameter_count"] != REFERENCE_PARAMETER_COUNT:
        return False
    cfg = model.config
    return (
        cfg.max_seq_len == REFERENCE_MAX_SEQ_LEN
        and cfg.n_layer == REFERENCE_N_LAYER
        and cfg.n_embd == REFERENCE_N_EMBD
        and cfg.n_head == REFERENCE_N_HEAD
    )


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
    _validate_current_checkpoint(directory / manifest["checkpoint"]["filename"], manifest)
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
        model = _validate_current_checkpoint(directory / manifest["checkpoint"]["filename"], manifest)
        web_runtime_compatible = _reference_web_shape(model, manifest)
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
            "web_runtime_compatible": web_runtime_compatible,
            "license": manifest["license"],
            "tags": manifest.get("tags", []),
        })
    bases.sort(key=lambda item: (item["display_name"].lower(), item["id"]))
    return {"schema_version": "0.2.0", "bases": bases}
