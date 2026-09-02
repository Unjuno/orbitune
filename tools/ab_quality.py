"""n_head=7 vs n_head=8 A/B quality gate.

Both runs use the same synthetic corpus, identical seed, identical
optimizer / LR / grad clip / seq_len / batch_size / step count, and the
same fixed validation plan. We compare:
  - per-component training loss (event_type, channel, delta, a1, a2,
    velocity, duration, control)
  - per-component validation loss on the same held-out windows

This is the closest A/B we can run on this machine; for a real decision
on quality the user must repeat the same A/B on real MIDI. The script
prints a side-by-side comparison at the end.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compound_cuda_train import (
    TensorSampler,
    config_from,
    precision_from,
    require_cuda,
    optimizer_for,
    train_step,
    autocast_for,
)
from scripts.compound_cfe_train import (
    install_causal_fastpath,
    validation_loss as cfe_validation_loss,
)
from orbitune.compound_base import CompoundHierarchicalGPT
from orbitune.compound_training import (
    capture_validation_window_plan,
    load_compound_jsonl,
)


def run_one(*, n_head: int, train_songs, validation_songs, args, device):
    base_cfg = config_from(args.config)
    cfg = replace(base_cfg, n_head=n_head)
    cfg.validate()

    install_causal_fastpath()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed + 7919)
    train_sampler = TensorSampler(train_songs)
    model = CompoundHierarchicalGPT(cfg).to(device)
    optimizer, fused = optimizer_for(model, args.learning_rate, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")

    train_history = []
    for step in range(1, args.steps + 1):
        x, y = train_sampler.sample(args.batch_size, args.seq_len, rng, device)
        loss, parts = train_step(model, optimizer, scaler, x, y, args.precision, args.grad_clip)
        train_history.append({"step": step, "loss": float(loss.detach()), "components": {k: float(v) for k, v in parts.items()}})

    val_plan = capture_validation_window_plan(
        validation_songs,
        validation_seed=args.validation_seed,
        batches=args.validation_batches,
        batch_size=min(args.batch_size, args.validation_batch_size),
        seq_len=args.seq_len,
    )
    val_loss, telemetry = cfe_validation_loss(
        model,
        validation_songs,
        plan_payload=val_plan,
        precision=args.precision,
        device=device,
    )

    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return {
        "n_head": n_head,
        "head_dim": cfg.d_model // cfg.n_head,
        "parameters": cfg.d_model * cfg.d_model // cfg.n_head * 4 + 0,  # informational
        "train_history": train_history,
        "validation_loss": val_loss,
        "validation_telemetry": telemetry,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/compound_hierarchical_9m.json")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--validation-seed", type=int, default=10001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="runs/ab_quality.json")
    args = parser.parse_args()

    args.precision = precision_from("auto")
    device = require_cuda()

    train_songs = load_compound_jsonl(args.train_jsonl)
    validation_songs = load_compound_jsonl(args.validation_jsonl)

    results = {"A": run_one(n_head=8, train_songs=train_songs, validation_songs=validation_songs, args=args, device=device),
               "B": run_one(n_head=7, train_songs=train_songs, validation_songs=validation_songs, args=args, device=device)}

    # Side-by-side summary.
    print("== Training loss comparison (mean over last 10 steps) ==")
    for label, run in results.items():
        tail = run["train_history"][-10:]
        mean_loss = sum(t["loss"] for t in tail) / len(tail)
        print(f"  {label} (n_head={run['n_head']}, head_dim={run['head_dim']}): mean_train_loss={mean_loss:.4f}")

    print("== Validation comparison (fixed plan) ==")
    for label, run in results.items():
        t = run["validation_telemetry"]
        print(f"  {label}: val_loss={run['validation_loss']:.4f}  window_hash={t['validation_window_hash'][:16]}...")

    # Validate same-window hash.
    assert results["A"]["validation_telemetry"]["validation_window_hash"] == \
        results["B"]["validation_telemetry"]["validation_window_hash"], "validation windows differ!"
    print("== Validation window hash identical between A and B ==")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()