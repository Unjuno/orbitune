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
    "parameter_count", "checkpoint", "web_onnx", "license", "training_data", "lineage", "tags",
}
OPTIONAL_FIELDS = {"description", "author"}
ALLOWED_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS
MAX_BASE_FILE_BYTES = 95 * 1024 * 1024
TRAINING_DATA_FIELDS = {"source_type", "license", "rights_confirmed", "notes"}
LINEAGE_FIELDS = {
    "parent_checkpoint",
    "commercial_eligible",
    "distribution_scope",
    "license_policy",
    "corpus_registry",
    "corpus_manifest_sha256",
    "restricted_source_ids",
    "rights_summary",
}
LICENSE_POLICIES = {"prod-only", "research-nc", "restricted"}
DISTRIBUTION_SCOPES = {"commercial", "noncommercial", "internal-only"}
_POLICY_SEVERITY = {"prod-only": 0, "research-nc": 1, "restricted": 2}

# These common licenses permit commercial use. A Base that is declared
# noncommercial/internal-only cannot simultaneously publish its checkpoint
# under one of them. Custom/source-specific terms remain reviewable rather than
# being guessed here.
COMMERCIAL_USE_LICENSE_IDS = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "mpl-2.0",
    "gpl-2.0",
    "gpl-2.0-only",
    "gpl-3.0",
    "gpl-3.0-only",
    "agpl-3.0",
    "agpl-3.0-only",
    "lgpl-2.1",
    "lgpl-2.1-only",
    "lgpl-3.0",
    "lgpl-3.0-only",
    "cc0-1.0",
    "cc-by-3.0",
    "cc-by-4.0",
    "cc-by-sa-3.0",
    "cc-by-sa-4.0",
}


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


def _validate_parent_checkpoint(value: object, errors: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        errors.append("lineage.parent_checkpoint must be null or an object")
        return
    if set(value) != {"id", "sha256"}:
        errors.append("lineage.parent_checkpoint must contain exactly id and sha256")
        return
    parent_id = value.get("id")
    parent_sha = value.get("sha256")
    if not isinstance(parent_id, str) or not BASE_ID_RE.fullmatch(parent_id):
        errors.append("lineage.parent_checkpoint.id must match ^[a-z0-9][a-z0-9-]*$")
    if not isinstance(parent_sha, str) or not validate_sha256(parent_sha):
        errors.append("lineage.parent_checkpoint.sha256 must be a 64-character SHA-256")


def _validate_lineage(lineage: object, errors: list[str]) -> None:
    if not isinstance(lineage, dict):
        errors.append("lineage must be an object")
        return
    missing = sorted(LINEAGE_FIELDS - lineage.keys())
    unknown = sorted(lineage.keys() - LINEAGE_FIELDS)
    if missing:
        errors.append(f"lineage missing required fields: {', '.join(missing)}")
    if unknown:
        errors.append(f"unknown lineage fields: {', '.join(unknown)}")

    _validate_parent_checkpoint(lineage.get("parent_checkpoint"), errors)

    commercial_eligible = lineage.get("commercial_eligible")
    if not isinstance(commercial_eligible, bool):
        errors.append("lineage.commercial_eligible must be a boolean")

    distribution_scope = lineage.get("distribution_scope")
    if distribution_scope not in DISTRIBUTION_SCOPES:
        errors.append(
            "lineage.distribution_scope must be one of commercial, noncommercial, internal-only"
        )

    license_policy = lineage.get("license_policy")
    if license_policy not in LICENSE_POLICIES:
        errors.append("lineage.license_policy must be one of prod-only, research-nc, restricted")

    corpus_registry = lineage.get("corpus_registry")
    if not isinstance(corpus_registry, str) or not corpus_registry.strip():
        errors.append("lineage.corpus_registry is required")

    corpus_sha = lineage.get("corpus_manifest_sha256")
    if not isinstance(corpus_sha, str) or not validate_sha256(corpus_sha):
        errors.append("lineage.corpus_manifest_sha256 must be a 64-character SHA-256")

    restricted_source_ids = lineage.get("restricted_source_ids")
    if (
        not isinstance(restricted_source_ids, list)
        or any(not isinstance(source_id, str) or not source_id for source_id in restricted_source_ids)
    ):
        errors.append("lineage.restricted_source_ids must be an array of non-empty strings")
    elif len(restricted_source_ids) != len(set(restricted_source_ids)):
        errors.append("lineage.restricted_source_ids must not contain duplicates")

    rights_summary = lineage.get("rights_summary")
    if not isinstance(rights_summary, str) or not rights_summary.strip():
        errors.append("lineage.rights_summary is required")

    if commercial_eligible is True:
        if license_policy != "prod-only":
            errors.append("commercial-eligible Base must use lineage.license_policy=prod-only")
        if distribution_scope != "commercial":
            errors.append("commercial-eligible Base must use lineage.distribution_scope=commercial")
        if isinstance(restricted_source_ids, list) and restricted_source_ids:
            errors.append("commercial-eligible Base must not list restricted_source_ids")
    elif commercial_eligible is False:
        if license_policy == "prod-only":
            errors.append("non-commercial Base must not use lineage.license_policy=prod-only")
        if distribution_scope == "commercial":
            errors.append("non-commercial Base must not use lineage.distribution_scope=commercial")

    if license_policy == "restricted" and distribution_scope != "internal-only":
        errors.append("restricted Base must use lineage.distribution_scope=internal-only")


def _validate_checkpoint_license_scope(manifest: dict[str, Any], errors: list[str]) -> None:
    lineage = manifest.get("lineage")
    checkpoint_license = manifest.get("license")
    if not isinstance(lineage, dict) or not isinstance(checkpoint_license, str):
        return
    scope = lineage.get("distribution_scope")
    normalized = checkpoint_license.strip().lower()
    if scope in {"noncommercial", "internal-only"} and normalized in COMMERCIAL_USE_LICENSE_IDS:
        errors.append(
            "noncommercial/internal-only Base must not use a standard checkpoint license that permits commercial use"
        )
    if scope == "commercial" and (
        "-nc" in normalized or "noncommercial" in normalized or "non-commercial" in normalized
    ):
        errors.append("commercial Base checkpoint license must not contain a noncommercial restriction")


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
    _validate_lineage(manifest.get("lineage"), errors)
    _validate_checkpoint_license_scope(manifest, errors)
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


def _validate_public_registry_policy(manifest: dict[str, Any]) -> None:
    lineage = manifest["lineage"]
    if lineage["distribution_scope"] == "internal-only" or lineage["license_policy"] == "restricted":
        raise ValueError(
            f"Base {manifest['id']} is restricted/internal-only and may not be published through the public Base registry"
        )


def _validate_registry_lineage(manifests: list[dict[str, Any]]) -> None:
    by_id = {manifest["id"]: manifest for manifest in manifests}

    for manifest in manifests:
        lineage = manifest["lineage"]
        parent_spec = lineage["parent_checkpoint"]
        if parent_spec is None:
            continue
        parent = by_id.get(parent_spec["id"])
        if parent is None:
            raise ValueError(
                f"Base {manifest['id']} references unknown parent Base {parent_spec['id']}"
            )
        if parent["checkpoint"]["sha256"].lower() != parent_spec["sha256"].lower():
            raise ValueError(
                f"Base {manifest['id']} parent checkpoint SHA does not match Base {parent_spec['id']}"
            )
        parent_lineage = parent["lineage"]
        child_severity = _POLICY_SEVERITY[lineage["license_policy"]]
        parent_severity = _POLICY_SEVERITY[parent_lineage["license_policy"]]
        if child_severity < parent_severity:
            raise ValueError(
                f"Base {manifest['id']} may not relax parent license policy "
                f"{parent_lineage['license_policy']} to {lineage['license_policy']}"
            )
        if lineage["commercial_eligible"] and not parent_lineage["commercial_eligible"]:
            raise ValueError(
                f"commercial-eligible Base {manifest['id']} may not descend from non-commercial Base {parent_spec['id']}"
            )

    for base_id in by_id:
        path: set[str] = set()
        current_id = base_id
        while True:
            if current_id in path:
                raise ValueError(f"Base lineage cycle detected at {current_id}")
            path.add(current_id)
            parent_spec = by_id[current_id]["lineage"]["parent_checkpoint"]
            if parent_spec is None:
                break
            current_id = parent_spec["id"]
            if current_id not in by_id:
                break


def build_base_registry(root: str | Path = "bases") -> dict[str, Any]:
    root = Path(root)
    directories = discover_base_directories(root)
    validated: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    for directory in directories:
        manifest = validate_base_directory(directory)
        base_id = manifest["id"]
        if base_id in seen:
            raise ValueError(f"duplicate Base id: {base_id}")
        seen.add(base_id)
        _validate_public_registry_policy(manifest)
        validated.append((directory, manifest))

    _validate_registry_lineage([manifest for _, manifest in validated])

    bases: list[dict[str, Any]] = []
    for directory, manifest in validated:
        base_id = manifest["id"]
        model = _validate_current_checkpoint(directory / manifest["checkpoint"]["filename"], manifest)
        web_runtime_compatible = _reference_web_shape(model, manifest)
        lineage = manifest["lineage"]
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
            "commercial_eligible": lineage["commercial_eligible"],
            "distribution_scope": lineage["distribution_scope"],
            "license_policy": lineage["license_policy"],
            "parent_checkpoint": lineage["parent_checkpoint"],
            "corpus_registry": lineage["corpus_registry"],
            "corpus_manifest_sha256": lineage["corpus_manifest_sha256"],
            "restricted_source_ids": lineage["restricted_source_ids"],
            "rights_summary": lineage["rights_summary"],
            "tags": manifest.get("tags", []),
        })
    bases.sort(key=lambda item: (item["display_name"].lower(), item["id"]))
    return {"schema_version": "0.3.0", "bases": bases}
