from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch

from orbitune.compound_base import CompoundHierarchicalGPT
from orbitune.compound_lora import (
    CompoundLoRAConfig,
    assert_base_unchanged,
    assert_only_lora_trainable,
    base_parameter_digests,
    inject_lora,
    save_adapter,
    sha256_file,
    trainable_parameter_names,
)
from orbitune.compound_training import load_compound_jsonl, sample_compound_batch


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    return torch.device(name)


def _validation_loss(
    model: CompoundHierarchicalGPT,
    songs,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    seed: int,
    batches: int,
) -> float:
    rng = random.Random(seed)
    was_training = model.training
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for _ in range(batches):
            inputs, targets = sample_compound_batch(
                songs,
                batch_size=batch_size,
                seq_len=seq_len,
                rng=rng,
                device=device,
            )
            loss, _ = model(inputs, targets)
            values.append(float(loss.detach().cpu()))
    model.train(was_training)
    return sum(values) / len(values)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental Compound LoRA SFT. This is research tooling, not the frozen public "
            "Compound Adapter ABI."
        )
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-id", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target-module",
        action="append",
        dest="target_modules",
        required=True,
        help=(
            "Exact module name or shell-style pattern. Repeat for multiple targets, e.g. "
            "--target-module 'decoder.stack.blocks.*.attn.q_proj'. No default is intentional."
        ),
    )
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--dropout", type=float, default=0.0)
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
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.batch_size <= 0 or args.seq_len <= 0:
        raise SystemExit("--batch-size and --seq-len must be positive")
    if args.validation_batches <= 0:
        raise SystemExit("--validation-batches must be positive")
    if args.learning_rate <= 0.0:
        raise SystemExit("--learning-rate must be positive")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_path = Path(args.base_checkpoint)
    base_sha256 = sha256_file(base_path)
    model, checkpoint_payload = CompoundHierarchicalGPT.load_checkpoint(base_path, map_location=device)
    model.to(device)

    config = CompoundLoRAConfig(
        target_patterns=tuple(args.target_modules),
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    resolved_targets = inject_lora(model, config)
    assert_only_lora_trainable(model)
    frozen_before = base_parameter_digests(model)

    train_songs = load_compound_jsonl(args.train_jsonl)
    validation_songs = load_compound_jsonl(args.validation_jsonl)

    initial_validation = _validation_loss(
        model,
        validation_songs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
        seed=args.seed + 1,
        batches=args.validation_batches,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = random.Random(args.seed)
    model.train()
    losses: list[float] = []

    for step in range(1, args.steps + 1):
        inputs, targets = sample_compound_batch(
            train_songs,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            rng=rng,
            device=device,
        )
        loss, _ = model(inputs, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}: {float(loss.detach().cpu())}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    assert_only_lora_trainable(model)
    assert_base_unchanged(model, frozen_before)

    final_validation = _validation_loss(
        model,
        validation_songs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
        seed=args.seed + 1,
        batches=args.validation_batches,
    )

    source_commit = os.environ.get("ORBITUNE_SOURCE_COMMIT") or os.environ.get("GITHUB_SHA")
    training_config: dict[str, object] = {
        "base_checkpoint": str(base_path),
        "train_jsonl": str(args.train_jsonl),
        "train_jsonl_sha256": sha256_file(args.train_jsonl),
        "validation_jsonl": str(args.validation_jsonl),
        "validation_jsonl_sha256": sha256_file(args.validation_jsonl),
        "target_patterns": list(args.target_modules),
        "resolved_targets": resolved_targets,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "validation_batches": args.validation_batches,
        "seed": args.seed,
        "device": str(device),
    }
    manifest = save_adapter(
        model,
        output_dir,
        base_model_id=args.base_id,
        base_sha256=base_sha256,
        architecture_abi=model.architecture,
        tokenizer_abi=model.tokenizer,
        source_commit=source_commit,
        training_config=training_config,
    )

    metrics = {
        "initial_validation_loss": initial_validation,
        "final_validation_loss": final_validation,
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "training_loss_min": min(losses),
        "training_loss_mean": sum(losses) / len(losses),
    }
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "training.json",
        {
            "status": "experimental",
            "base_checkpoint_step": checkpoint_payload.get("step"),
            "base_checkpoint_source_commit": checkpoint_payload.get("source_commit"),
            "trainable_parameters": trainable_parameter_names(model),
            "config": training_config,
        },
    )

    print(json.dumps({"adapter": manifest, "metrics": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
