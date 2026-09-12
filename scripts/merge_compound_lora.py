from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from orbitune.compound_base import CompoundHierarchicalGPT
from orbitune.compound_lora import ADAPTER_MANIFEST_FILE, load_adapter
from orbitune.compound_lora_merge import (
    adapter_artifact_digests,
    merge_lora_inplace,
    sha256_file,
    verify_lora_merge_parity,
)


PREMERGED_SCHEMA = "orbitune-compound-lora-premerged-experimental-v0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge an experimental Compound LoRA Adapter into an immutable Base checkpoint"
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--output-manifest", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_path = Path(args.base_checkpoint)
    adapter_dir = Path(args.adapter_dir)
    output_checkpoint = Path(args.output_checkpoint)
    output_manifest = Path(args.output_manifest)

    base_sha = sha256_file(base_path)
    expected_base_sha = args.base_sha256.strip().lower()
    if base_sha != expected_base_sha:
        raise SystemExit(f"Base checkpoint SHA-256 mismatch: {base_sha} != {expected_base_sha}")

    raw_adapter_manifest = json.loads((adapter_dir / ADAPTER_MANIFEST_FILE).read_text(encoding="utf-8"))
    if raw_adapter_manifest.get("base_model_id") != args.base_model_id:
        raise SystemExit(
            f"Adapter Base id mismatch: {raw_adapter_manifest.get('base_model_id')!r} != {args.base_model_id!r}"
        )

    model, base_payload = CompoundHierarchicalGPT.load_checkpoint(base_path, map_location="cpu")
    adapter_manifest = load_adapter(model, adapter_dir, base_sha256=base_sha, strict_base_binding=True)
    model.eval()
    merge_parity = verify_lora_merge_parity(model)
    merged_modules = merge_lora_inplace(model)

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    derived_payload = dict(base_payload)
    derived_payload["model_state_dict"] = model.state_dict()
    derived_payload["optimizer_state_dict"] = None
    derived_payload["derived_artifact"] = {
        "kind": "lora-premerged",
        "schema": PREMERGED_SCHEMA,
        "base_model_id": args.base_model_id,
        "base_checkpoint_sha256": base_sha,
        "adapter_id": args.adapter_id,
        "adapter_schema": adapter_manifest.get("schema"),
        "target_modules": merged_modules,
        "rank": int(adapter_manifest["rank"]),
        "alpha": float(adapter_manifest["alpha"]),
        "scaling": float(adapter_manifest["scaling"]),
        "merge_parity": merge_parity,
    }
    torch.save(derived_payload, output_checkpoint)

    digests = adapter_artifact_digests(adapter_dir)
    merged_sha = sha256_file(output_checkpoint)
    manifest = {
        "schema": PREMERGED_SCHEMA,
        "status": "experimental_premerged_candidate",
        "public_adapter_abi": False,
        "publication_eligible": False,
        "base": {
            "model_id": args.base_model_id,
            "checkpoint_sha256": base_sha,
            "architecture": model.architecture,
            "tokenizer": model.tokenizer,
        },
        "adapter": {
            "id": args.adapter_id,
            "schema": adapter_manifest.get("schema"),
            **digests,
            "target_modules": merged_modules,
            "rank": int(adapter_manifest["rank"]),
            "alpha": float(adapter_manifest["alpha"]),
            "scaling": float(adapter_manifest["scaling"]),
            "source_commit": adapter_manifest.get("source_commit"),
        },
        "merge_parity": merge_parity,
        "merged_checkpoint": {
            "filename": output_checkpoint.name,
            "sha256": merged_sha,
            "bytes": output_checkpoint.stat().st_size,
            "loadable_as": "orbitune-compound-hierarchical-gpt-v1",
        },
        "next_gate": "export V2 graphs, verify native/ORT/WASM parity, evaluate quality and rights, then approve a lora-premerged Web release",
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
