#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from orbitune.compat import validate_sha256


def validate_runtime_config(config: dict[str, Any], *, allow_unpublished: bool = False) -> None:
    allowed = {"model_url", "model_sha256", "base_sha256", "execution_providers"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"unknown runtime-config fields: {sorted(unknown)}")
    model_url = str(config.get("model_url", ""))
    model_sha256 = str(config.get("model_sha256", "")).lower()
    base_sha256 = str(config.get("base_sha256", "")).lower()
    providers = config.get("execution_providers")
    if providers != ["wasm"]:
        raise ValueError("Orbitune browser runtime execution_providers must be ['wasm']")
    if not model_url:
        if allow_unpublished and not model_sha256 and not base_sha256:
            return
        raise ValueError("model_url is required for a published runtime config")
    if not model_url.startswith("https://"):
        raise ValueError("model_url must use HTTPS")
    if not validate_sha256(model_sha256):
        raise ValueError("model_sha256 must be a 64-character hexadecimal SHA-256")
    if not validate_sha256(base_sha256):
        raise ValueError("base_sha256 must be the immutable Base checkpoint SHA-256")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and install an Orbitune browser runtime config.")
    parser.add_argument("runtime_config", help="runtime-config.json to validate")
    parser.add_argument("--out", help="optional destination; omit to validate only")
    parser.add_argument("--allow-unpublished", action="store_true", help="allow the pre-release state with empty model/base hashes")
    args = parser.parse_args()

    source = Path(args.runtime_config)
    config = json.loads(source.read_text(encoding="utf-8"))
    validate_runtime_config(config, allow_unpublished=args.allow_unpublished)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"installed verified browser runtime config: {out}")
    else:
        print(f"valid browser runtime config: {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
