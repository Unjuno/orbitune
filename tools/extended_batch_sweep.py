"""Extended CFE microbatch sweep on the RTX 3080 for head_dim=32.

The original CFE report only swept batch sizes up to 128. This script
extends the sweep to find the true throughput-vs-VRAM optimum while
enforcing the 0.92 VRAM safety ceiling.
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
    cuda_stats,
)
from scripts.compound_cfe_train import install_causal_fastpath
from orbitune.compound_base import CompoundHierarchicalGPT
from orbitune.compound_training import load_compound_jsonl


def measure(cfg, sampler, *, batch_size, seq_len, precision, warmup, measure, lr, wd, gc, seed):
    device = require_cuda()
    install_causal_fastpath()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    rng = random.Random(seed + 7919)
    model = CompoundHierarchicalGPT(cfg).to(device)
    optimizer, fused = optimizer_for(model, lr, wd)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    try:
        for _ in range(warmup):
            x, y = sampler.sample(batch_size, seq_len, rng, device)
            train_step(model, optimizer, scaler, x, y, precision, gc)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = __import__("time").perf_counter()
        last = None
        for _ in range(measure):
            x, y = sampler.sample(batch_size, seq_len, rng, device)
            last, _ = train_step(model, optimizer, scaler, x, y, precision, gc)
        torch.cuda.synchronize()
        elapsed = __import__("time").perf_counter() - started
        return {
            "status": "ok",
            "n_head": cfg.n_head,
            "head_dim": cfg.d_model // cfg.n_head,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "parameters": model.parameter_count(),
            "fused_adamw": fused,
            "steps_per_sec": measure / elapsed,
            "events_per_sec": batch_size * seq_len * measure / elapsed,
            "loss": None if last is None else float(last),
            **cuda_stats(),
        }
    except torch.OutOfMemoryError:
        return {
            "status": "oom",
            "n_head": cfg.n_head,
            "head_dim": cfg.d_model // cfg.n_head,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }
    finally:
        del model, optimizer, scaler
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", default="benchmarks/fixtures/cfe/synthetic_compound.jsonl")
    parser.add_argument("--config", default="configs/compound_hierarchical_9m_nhead7.json")
    parser.add_argument("--batch-sizes", default="128,136,144,152,160,168,176,192")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=50)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max-vram-fraction", type=float, default=0.92)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--out", default="runs/cfe_extended_batch.json")
    args = parser.parse_args()

    precision = precision_from("auto")
    base_cfg = config_from(args.config)
    sampler = TensorSampler(load_compound_jsonl(args.train_jsonl))

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    rows = []
    for bs in batch_sizes:
        for trial in range(args.trials):
            print(f"[bs={bs} trial={trial}]", flush=True)
            try:
                row = measure(
                    cfg=base_cfg,
                    sampler=sampler,
                    batch_size=bs,
                    seq_len=args.seq_len,
                    precision=precision,
                    warmup=args.warmup,
                    measure=args.measure,
                    lr=args.learning_rate,
                    wd=args.weight_decay,
                    gc=args.grad_clip,
                    seed=1 + trial,
                )
            except Exception as exc:
                row = {"status": "error", "batch_size": bs, "trial": trial, "error": str(exc).splitlines()[0][:200]}
            row["trial"] = trial
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if row.get("status") == "oom":
                break

    safe = [
        r for r in rows
        if r.get("status") == "ok"
        and r.get("peak_reserved_fraction", 0.0) <= args.max_vram_fraction
    ]
    summary = {
        "definition": "Extended microbatch sweep for the n_head=7 (head_dim=32) Compound base on RTX 3080 16GB.",
        "recommended": max(safe, key=lambda r: r["events_per_sec"]) if safe else None,
        "all": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()