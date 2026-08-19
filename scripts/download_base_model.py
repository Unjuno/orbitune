#!/usr/bin/env python3
"""Download and verify Orbitune base-model release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

MODEL_ID = "orbitune-tiny-v0"
DEFAULT_MANIFEST = (
    "https://github.com/Unjuno/orbitune/releases/latest/download/"
    "orbitune-tiny-v0-manifest.json"
)


def _read_json(location: str) -> dict[str, Any]:
    if location.startswith(("https://", "http://")):
        if not location.startswith("https://"):
            raise ValueError("remote manifests must use HTTPS")
        with urllib.request.urlopen(location, timeout=30) as response:
            return json.load(response)
    return json.loads(Path(location).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifact(manifest: dict[str, Any], artifact: str) -> dict[str, Any]:
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError(f"manifest model_id must be {MODEL_ID}")
    if manifest.get("architecture") != "orbitune-midi-gpt-v0":
        raise ValueError("manifest architecture mismatch")
    if manifest.get("tokenizer") != "theory-remi-v0":
        raise ValueError("manifest tokenizer mismatch")
    if manifest.get("parameters") != 2_945_760:
        raise ValueError("manifest parameter count mismatch")
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
    if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
        raise ValueError("artifact sha256 is invalid")
    if int(spec["bytes"]) <= 0:
        raise ValueError("artifact byte size must be positive")
    return spec


def _download_verified(spec: dict[str, Any], target: Path) -> None:
    expected_hash = str(spec["sha256"]).lower()
    expected_bytes = int(spec["bytes"])
    if target.exists() and target.stat().st_size == expected_bytes and _sha256(target) == expected_hash:
        print(f"already verified: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=target.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with urllib.request.urlopen(str(spec["url"]), timeout=60) as response, temp.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        actual_bytes = temp.stat().st_size
        actual_hash = _sha256(temp)
        if actual_bytes != expected_bytes:
            raise ValueError(f"downloaded size {actual_bytes} != expected {expected_bytes}")
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch: {actual_hash} != {expected_hash}")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a verified Orbitune base-model release asset.")
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
