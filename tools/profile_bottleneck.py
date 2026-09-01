"""Profile the final candidate (n_head=7, hd=32, bs=128, seq=256) to find bottlenecks."""
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
    TensorSampler, config_from, precision_from, require_cuda,
    optimizer_for, train_step, cuda_stats,
)
from scripts.compound_cfe_train import install_causal_fastpath
from orbitune.compound_base import CompoundHierarchicalGPT
from orbitune.compound_training import load_compound_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/compound_hierarchical_9m.json")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--n-head", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=20)
    parser.add_argument("--out", default="runs/profile.json")
    args = parser.parse_args()

    install_causal_fastpath()
    precision = precision_from("auto")
    base_cfg = config_from(args.config)
    cfg = replace(base_cfg, n_head=args.n_head)
    cfg.validate()
    sampler = TensorSampler(load_compound_jsonl(args.train_jsonl))

    torch.manual_seed(1)
    rng = random.Random(1 + 7919)
    model = CompoundHierarchicalGPT(cfg).to(device := require_cuda())
    optimizer, _ = optimizer_for(model, 3e-4, 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    def step():
        x, y = sampler.sample(args.batch_size, args.seq_len, rng, device)
        return train_step(model, optimizer, scaler, x, y, precision, 1.0)

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    # PyTorch profiler
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(args.measure):
            step()
        torch.cuda.synchronize()

    # Aggregate by kernel name and op
    summary = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
    print(summary)
    print("---CPU---")
    cpu_summary = prof.key_averages().table(sort_by="cpu_time_total", row_limit=40)
    print(cpu_summary)

    # Save chrome trace
    Path("runs").mkdir(exist_ok=True)
    prof.export_chrome_trace("runs/trace.json")

    Path(args.out).write_text(
        json.dumps({
            "cuda": cuda_stats(),
            "top_cuda_ops": [{"name": e.key, "cuda_time_total_us": e.cuda_time_total, "count": e.count}
                             for e in prof.key_averages() if e.cuda_time_total > 0][:30],
            "top_cpu_ops": [{"name": e.key, "cpu_time_total_us": e.cpu_time_total, "count": e.count}
                            for e in prof.key_averages() if e.cpu_time_total > 0][:30],
        }, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()