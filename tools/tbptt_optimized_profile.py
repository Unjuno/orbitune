"""Short torch.profiler run on the optimized TBPTT path.

Trains a few steps at (batch, seq_len) on a resumed checkpoint, profiling
the inner optimized `tbptt_loss` and the optimizer step. Reports the top
hot ops by total device time and total CPU time, dumped as JSON for
follow-up analysis.

Used to characterise the e9fb567 "perf(tbptt): batch state-carry work
across lanes" optimization and to identify the next bottleneck for
state-carry TBPTT on RTX 3080.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orbitune.compound_base import CompoundHierarchicalGPT
from orbitune.compound_tbptt import (
    SequentialSongChunkSampler,
    detach_batch_stream_states,
    initial_batch_stream_states,
)
from orbitune.compound_tbptt_optimized import tbptt_loss as optimized_tbptt_loss
from orbitune.compound_training import load_compound_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    songs = load_compound_jsonl(args.train_jsonl)
    model, payload = CompoundHierarchicalGPT.load_checkpoint(args.resume, map_location="cpu")
    model.to(device).train()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    sampler = SequentialSongChunkSampler(
        songs, seq_len=args.seq_len, batch_size=args.batch_size, rng=random.Random(args.seed)
    )
    states = initial_batch_stream_states(model, args.batch_size)

    def step() -> tuple[torch.Tensor, object]:
        batch = sampler.sample(device)
        x = batch.inputs
        y = batch.targets
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, _parts, new_states = optimized_tbptt_loss(model, x, y, states)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        return loss, new_states

    # warmup (no profiler)
    for _ in range(2):
        _loss, states = step()
        if isinstance(states, tuple):
            states = states[2]
        states = detach_batch_stream_states(states)
    torch.cuda.synchronize()

    # profiled steps
    sched = schedule(wait=0, warmup=1, active=args.steps, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for i in range(args.steps):
            with record_function(f"step_{i}"):
                _loss, states = step()
                if isinstance(states, tuple):
                    states = states[2]
                states = detach_batch_stream_states(states)
            prof.step()

    torch.cuda.synchronize()

    # Aggregate by op name. prof.key_averages() returns aggregated
    # averages over the active steps (the wait/warmup/active schedule
    # configures which steps are counted).
    key_avg = prof.key_averages()
    rows = []
    for evt in key_avg:
        if evt.count == 0:
            continue
        rows.append({
            "name": evt.key,
            "count": int(evt.count),
            "cpu_time_total_us": float(evt.cpu_time_total),
            "device_time_total_us": float(evt.device_time_total),
            "self_cpu_time_total_us": float(evt.self_cpu_time_total),
            "self_device_time_total_us": float(evt.self_device_time_total),
        })
    rows.sort(key=lambda r: -r["device_time_total_us"])
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "events_per_step": args.batch_size * args.seq_len,
            "profiled_steps": args.steps,
            "precision": args.precision,
            "rows": rows,
        }, f, indent=2)
    # Top 20 by device time (CUDA + CPU)
    print(f"--- top 20 by device_time_total_us (over {args.steps} steps) ---")
    for r in rows[:20]:
        print(f"  dev={r['device_time_total_us']/1000:>9.1f}ms  cpu={r['cpu_time_total_us']/1000:>9.1f}ms  self_dev={r['self_device_time_total_us']/1000:>9.1f}ms  count={r['count']:>4}  {r['name'][:80]}")


if __name__ == "__main__":
    main()
