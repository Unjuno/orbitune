from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from safetensors.torch import load_file as load_safetensors

from orbitune.compound_lora import ADAPTER_TENSOR_FILE, sha256_file


SWEEP_SCHEMA = "orbitune-compound-lora-sweep-v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_spec(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("sweep spec schema_version must be 1")
    raw = payload.get("candidates")
    if not isinstance(raw, list) or not raw:
        raise ValueError("sweep spec requires a non-empty candidates list")
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"candidate {index} must be an object")
        candidate_id = str(item.get("id", "")).strip()
        if not candidate_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in candidate_id):
            raise ValueError(f"candidate {index} has an invalid id")
        if candidate_id in seen:
            raise ValueError(f"duplicate candidate id: {candidate_id}")
        seen.add(candidate_id)
        targets = item.get("target_modules")
        if not isinstance(targets, list) or not targets or not all(isinstance(value, str) and value for value in targets):
            raise ValueError(f"candidate {candidate_id} requires target_modules")
        rank = int(item.get("rank", 0))
        alpha = float(item.get("alpha", 0.0))
        dropout = float(item.get("dropout", 0.0))
        if rank <= 0:
            raise ValueError(f"candidate {candidate_id} rank must be positive")
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError(f"candidate {candidate_id} alpha must be finite and positive")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError(f"candidate {candidate_id} dropout must be in [0, 1)")
        candidates.append(
            {
                "id": candidate_id,
                "target_modules": list(targets),
                "rank": rank,
                "alpha": alpha,
                "dropout": dropout,
            }
        )
    return candidates


def _adapter_parameter_count(path: Path) -> int:
    tensors = load_safetensors(str(path / ADAPTER_TENSOR_FILE), device="cpu")
    return sum(tensor.numel() for tensor in tensors.values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled experimental Compound LoRA target/rank sweep. "
            "This ranks bounded ML candidates only; it does not establish musical preference."
        )
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-id", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.seq_len <= 0 or args.validation_batches <= 0:
        raise SystemExit("steps, batch-size, seq-len and validation-batches must be positive")
    if args.learning_rate <= 0.0:
        raise SystemExit("learning-rate must be positive")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(args.base_checkpoint)
    train_path = Path(args.train_jsonl)
    validation_path = Path(args.validation_jsonl)
    spec_path = Path(args.spec)
    candidates = _load_spec(spec_path)

    common = {
        "base_id": args.base_id,
        "base_checkpoint_sha256": sha256_file(base_path),
        "train_jsonl_sha256": sha256_file(train_path),
        "validation_jsonl_sha256": sha256_file(validation_path),
        "spec_sha256": sha256_file(spec_path),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "validation_batches": args.validation_batches,
        "seed": args.seed,
        "device": args.device,
        "source_commit": os.environ.get("ORBITUNE_SOURCE_COMMIT") or os.environ.get("GITHUB_SHA"),
    }

    results: list[dict[str, Any]] = []
    reference_initial: float | None = None
    for candidate in candidates:
        candidate_dir = output_dir / candidate["id"]
        command = [
            sys.executable,
            "scripts/compound_lora_sft.py",
            "--base-checkpoint", str(base_path),
            "--base-id", args.base_id,
            "--train-jsonl", str(train_path),
            "--validation-jsonl", str(validation_path),
            "--output-dir", str(candidate_dir),
            "--rank", str(candidate["rank"]),
            "--alpha", str(candidate["alpha"]),
            "--dropout", str(candidate["dropout"]),
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--seq-len", str(args.seq_len),
            "--learning-rate", str(args.learning_rate),
            "--weight-decay", str(args.weight_decay),
            "--validation-batches", str(args.validation_batches),
            "--seed", str(args.seed),
            "--device", args.device,
        ]
        for target in candidate["target_modules"]:
            command.extend(["--target-module", target])
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise SystemExit(
                f"candidate {candidate['id']} failed with exit {completed.returncode}:\n{completed.stderr}"
            )

        metrics = json.loads((candidate_dir / "metrics.json").read_text(encoding="utf-8"))
        adapter = json.loads((candidate_dir / "adapter.json").read_text(encoding="utf-8"))
        initial = float(metrics["initial_validation_loss"])
        final = float(metrics["final_validation_loss"])
        if reference_initial is None:
            reference_initial = initial
        elif not math.isclose(initial, reference_initial, rel_tol=1e-7, abs_tol=1e-8):
            raise RuntimeError(
                "candidate initial validation losses differ despite zero-init LoRA and identical validation protocol: "
                f"{candidate['id']}={initial}, reference={reference_initial}"
            )
        trainable_parameters = _adapter_parameter_count(candidate_dir)
        results.append(
            {
                "id": candidate["id"],
                "target_patterns": candidate["target_modules"],
                "resolved_targets": list(adapter["target_modules"]),
                "rank": candidate["rank"],
                "alpha": candidate["alpha"],
                "dropout": candidate["dropout"],
                "trainable_parameters": trainable_parameters,
                "adapter_bytes": (candidate_dir / ADAPTER_TENSOR_FILE).stat().st_size,
                "adapter_sha256": sha256_file(candidate_dir / ADAPTER_TENSOR_FILE),
                "initial_validation_loss": initial,
                "final_validation_loss": final,
                "validation_delta": final - initial,
                "loss_improvement": initial - final,
            }
        )

    ranked = sorted(results, key=lambda item: (item["final_validation_loss"], item["trainable_parameters"], item["id"]))
    for index, result in enumerate(ranked, start=1):
        result["ml_rank"] = index
    summary = {
        "schema": SWEEP_SCHEMA,
        "status": "experimental_ml_screen",
        "preference_claim": False,
        "selection_rule": "ascending final_validation_loss, then ascending trainable_parameters, then id",
        "common": common,
        "base_validation_loss": reference_initial,
        "candidates": ranked,
        "next_gate": (
            "Do not freeze a production Adapter target/rank from this loss screen alone. "
            "Run generated-MIDI structural checks and blinded human preference evaluation on the leading candidates."
        ),
    }
    _write_json(output_dir / "sweep.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
