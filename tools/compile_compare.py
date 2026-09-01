"""Compare torch.compile modes on the final CFE candidate (A: n_head=7, hd=32)."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
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
from scripts.compound_cfe_train import (
    install_causal_fastpath,
    uninstall_causal_fastpath,
)
from orbitune.compound_base import CompoundHierarchicalGPT
from orbitune.compound_training import load_compound_jsonl


def measure_with_compile(cfg, sampler, *, batch_size, seq_len, precision,
                         warmup, measure, lr, wd, gc, compile_mode):
    device = require_cuda()
    install_causal_fastpath()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(1)
    rng = random.Random(1 + 7919)
    model = CompoundHierarchicalGPT(cfg).to(device)
    if compile_mode:
        # capture compile time separately
        t0 = time.perf_counter()
        compiled = torch.compile(model, mode=compile_mode)
        # Do a dummy forward to trigger compilation
        x, y = sampler.sample(batch_size, seq_len, rng, device)
        loss, _ = train_step(compiled, optimizer_for(model, lr, wd)[0],
                              torch.amp.GradScaler("cuda", enabled=precision == "fp16"),
                              x, y, precision, gc)
        torch.cuda.synchronize()
        compile_time = time.perf_counter() - t0
        # After warmup steps we re-time training; the dummy already counts as warmup.
        optimizer, _ = optimizer_for(model, lr, wd)
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        # Use compiled for forward+backward
        def step():
            x, y = sampler.sample(batch_size, seq_len, rng, device)
            return train_step(compiled, optimizer, scaler, x, y, precision, gc)
    else:
        compile_time = 0.0
        optimizer, _ = optimizer_for(model, lr, wd)
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

        def step():
            x, y = sampler.sample(batch_size, seq_len, rng, device)
            return train_step(model, optimizer, scaler, x, y, precision, gc)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(measure):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "compile_mode": compile_mode,
        "compile_time_s": compile_time,
        "steps_per_sec": measure / elapsed,
        "events_per_sec": batch_size * seq_len * measure / elapsed,
        **cuda_stats(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/compound_hierarchical_9m.json")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--out", default="runs/compile_compare.json")
    args = parser.parse_args()

    precision = precision_from(args.precision)
    base_cfg = config_from(args.config)
    cfg = replace(base_cfg, n_head=args.n_head)
    cfg.validate()
    sampler = TensorSampler(load_compound_jsonl(args.train_jsonl))

    rows = []
    for mode in [None, "default", "reduce-overhead", "max-autotune"]:
        print(f"[compile_compare] mode={mode}", flush=True)
        try:
            row = measure_with_compile(
                cfg=cfg, sampler=sampler,
                batch_size=args.batch_size, seq_len=args.seq_len,
                precision=precision, warmup=args.warmup, measure=args.measure,
                lr=args.learning_rate, wd=args.weight_decay, gc=args.grad_clip,
                compile_mode=mode,
            )
            rows.append({"status": "ok", **row})
        except Exception as exc:
            rows.append({"status": "error", "compile_mode": mode, "error": str(exc).splitlines()[0][:240]})
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()