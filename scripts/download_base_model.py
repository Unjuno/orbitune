#!/usr/bin/env python3
"""Download and verify immutable Orbitune Base release assets."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from orbitune.compat import ARCHITECTURE_ABI, BASE_MODEL_ID, BASE_PARAMETER_COUNT, TOKENIZER_ABI, sha256_file, validate_sha256

DEFAULT_MANIFEST = (
    "https://github.com/Unjuno/orbitune/releases/latest/download/"
    "orbitune-base-manifest.json"
)


def _read_json(location: str) -> dict[str, Any]:
    if location.startswith(("https://", "http://")):
        if not location.startswith("https://"):
            raise ValueError("remote manifests must use HTTPS")
        with urllib.request.urlopen(location, timeout=30) as response:
            return json.load(response)
    return json.loads(Path(location).read_text(encoding="utf-8"))


def _validate_artifact(manifest: dict[str, Any], artifact: str) -> dict[str, Any]:
    if manifest.get("model_id") != BASE_MODEL_ID:
        raise ValueError(f"manifest model_id must be {BASE_MODEL_ID}")
    if manifest.get("architecture") != ARCHITECTURE_ABI:
        raise ValueError("manifest architecture mismatch")
    if manifest.get("tokenizer") != TOKENIZER_ABI:
        raise ValueError("manifest tokenizer mismatch")
    if manifest.get("parameters") != BASE_PARAMETER_COUNT:
        raise ValueError("manifest parameter count mismatch")
    base_sha = str(manifest.get("base_sha256", "")).lower()
    if not validate_sha256(base_sha):
        raise ValueError("manifest base_sha256 is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or artifact not in artifacts:
        raise ValueError(f"manifest does not contain artifact {artifact!r}")
    spec = artifacts[artifact]
    if not isinstance(spec, dict):
        raise ValueError("invalid artifact record")
    required = {"filename", "bytes", "sha256", "url"}
    if not required.issubset(spec):
        raise ValueError(f"artifact record is missing: {sorted(required - spec.keys())}")
    if not str(spec["url"]).startswith("https://"):
        raise ValueError("artifact URL must use HTTPS")
    expected_hash = str(spec["sha256"]).lower()
    if not validate_sha256(expected_hash):
        raise ValueError("artifact sha256 is invalid")
    if artifact == "checkpoint" and expected_hash != base_sha:
        raise ValueError("checkpoint artifact hash must equal manifest base_sha256")
    if int(spec["bytes"]) <= 0:
        raise ValueError("artifact byte size must be positive")
    return spec


def _download_verified(spec: dict[str, Any], target: Path) -> None:
    expected_hash = str(spec["sha256"]).lower()
    expected_bytes = int(spec["bytes"])
    if target.exists() and target.stat().st_size == expected_bytes and sha256_file(target) == expected_hash:
        print(f"already verified: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=target.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with urllib.request.urlopen(str(spec["url"]), timeout=60) as response, temp.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        actual_bytes = temp.stat().st_size
        actual_hash = sha256_file(temp)
        if actual_bytes != expected_bytes:
            raise ValueError(f"downloaded size {actual_bytes} != expected {expected_bytes}")
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch: {actual_hash} != {expected_hash}")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a verified immutable Orbitune Base release asset.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="release manifest URL or local JSON file")
    parser.add_argument("--artifact", choices=["checkpoint", "web_onnx"], default="checkpoint")
    parser.add_argument("--out", default="models", help="output directory")
    args = parser.parse_args()
    manifest = _read_json(args.manifest)
    spec = _validate_artifact(manifest, args.artifact)
    target = Path(args.out) / str(spec["filename"])
    _download_verified(spec, target)
    print(f"verified {args.artifact}: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
