#!/usr/bin/env python3
# Temporary CI trigger branch change; do not merge.
from __future__ import annotations

import argparse
import json

from orbitune.release import BASE_MODEL_ID, package_base_release


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package a verified Orbitune v0 checkpoint and browser ONNX file for a GitHub Release."
    )
    parser.add_argument("--base", required=True, help="trained orbitune-tiny-v0 PyTorch checkpoint")
    parser.add_argument("--web-onnx", required=True, help="exported external-LoRA browser ONNX graph")
    parser.add_argument("--out-dir", required=True, help="release staging directory")
    parser.add_argument("--repository", default="Unjuno/orbitune")
    parser.add_argument("--release-tag", default=BASE_MODEL_ID)
    args = parser.parse_args()

    manifest = package_base_release(
        args.base,
        args.web_onnx,
        args.out_dir,
        repository=args.repository,
        release_tag=args.release_tag,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
