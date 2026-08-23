#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from orbitune.base_registry import BASE_ID_RE, MAX_BASE_FILE_BYTES
from orbitune.compat import sha256_file
from orbitune.model import OrbituneGPT


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage a contributed Orbitune Base under bases/<id>/")
    parser.add_argument("--id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--web-onnx", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--training-license", required=True)
    parser.add_argument("--rights-confirmed", action="store_true", help="confirm you have the rights required to contribute the training data/model")
    parser.add_argument("--source-type", default="midi_corpus")
    parser.add_argument("--description", default="")
    parser.add_argument("--author", default="")
    parser.add_argument("--out-root", default="bases")
    args = parser.parse_args()

    if not BASE_ID_RE.fullmatch(args.id):
        raise SystemExit("id must match ^[a-z0-9][a-z0-9-]*$")
    if not args.display_name.strip() or not args.license.strip() or not args.training_license.strip() or not args.source_type.strip():
        raise SystemExit("display-name, license, training-license, and source-type must be non-empty")
    if not args.rights_confirmed:
        raise SystemExit("refusing to stage Base without explicit --rights-confirmed")

    checkpoint = Path(args.checkpoint)
    onnx = Path(args.web_onnx)
    if not checkpoint.is_file() or not onnx.is_file():
        raise SystemExit("checkpoint and web ONNX must exist")
    for path in (checkpoint, onnx):
        if path.stat().st_size <= 0:
            raise SystemExit(f"{path} must not be empty")
        if path.stat().st_size > MAX_BASE_FILE_BYTES:
            raise SystemExit(f"{path} exceeds the 95 MiB repository policy")

    # This staging tool intentionally uses the currently operational
    # OrbituneGPT class. Architecture and tokenizer are derived from the loaded
    # checkpoint/model class rather than duplicated as manifest literals.
    model = OrbituneGPT.load_checkpoint(checkpoint, map_location="cpu").eval()
    target = Path(args.out_root) / args.id
    if target.exists():
        raise SystemExit(f"target already exists: {target}; Base ids are immutable")
    target.mkdir(parents=True)
    checkpoint_out = target / "model.pt"
    onnx_out = target / "web.onnx"
    shutil.copy2(checkpoint, checkpoint_out)
    shutil.copy2(onnx, onnx_out)

    manifest = {
        "artifact_type": "orbitune_base",
        "id": args.id,
        "display_name": args.display_name.strip(),
        "description": args.description,
        "author": args.author,
        "architecture": model.architecture,
        "tokenizer": model.tokenizer,
        "parameter_count": model.parameter_count(),
        "checkpoint": {"filename": checkpoint_out.name, "sha256": sha256_file(checkpoint_out), "bytes": checkpoint_out.stat().st_size},
        "web_onnx": {"filename": onnx_out.name, "sha256": sha256_file(onnx_out), "bytes": onnx_out.stat().st_size},
        "license": args.license.strip(),
        "training_data": {
            "source_type": args.source_type.strip(),
            "license": args.training_license.strip(),
            "rights_confirmed": True,
        },
        "tags": [],
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (target / "README.md").write_text(
        f"# {manifest['display_name']}\n\nBase id: `{args.id}`\n\nCheckpoint SHA-256: `{manifest['checkpoint']['sha256']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
