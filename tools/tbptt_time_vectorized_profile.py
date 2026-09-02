"""Short torch.profiler run on the time-vectorized TBPTT path."""
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
from orbitune.compound_tbptt_time_vectorized import tbptt_loss as time_vectorized_tbptt_loss
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
    model, _payload = CompoundHierarchicalGPT.load_checkpoint(args.resume, map_location="cpu")
    model.to(device).train()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    sampler = SequentialSongChunkSampler(
        songs, seq_len=args.seq_len, batch_size=args.batch_size, rng=random.Random(args.seed)
    )
    states = initial_batch_stream_states(model, args.batch_size)

    def step() -> tuple[torch.Tensor, list]:
        nonlocal states
        batch = sampler.sample(device)
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, _parts, states = time_vectorized_tbptt_loss(
                model, batch.inputs, batch.targets, states, reset_mask=batch.reset_mask
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        states = detach_batch_stream_states(states)
        return loss, states

    for _ in range(2):
        step()
    torch.cuda.synchronize()

    sched = schedule(wait=0, warmup=1, active=args.steps, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for i in range(args.steps):
            with record_function(f"step_{i}"):
                step()
            prof.step()

    torch.cuda.synchronize()
    rows = []
    for evt in prof.key_averages():
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
    Path(args.out_json).write_text(
        json.dumps({
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "events_per_step": args.batch_size * args.seq_len,
            "profiled_steps": args.steps,
            "precision": args.precision,
            "rows": rows,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"--- top 20 by device_time_total_us (over {args.steps} steps) ---")
    for row in rows[:20]:
        print(
            f"  dev={row['device_time_total_us']/1000:>9.1f}ms  "
            f"cpu={row['cpu_time_total_us']/1000:>9.1f}ms  "
            f"self_dev={row['self_device_time_total_us']/1000:>9.1f}ms  "
            f"count={row['count']:>4}  {row['name'][:80]}"
        )


if __name__ == "__main__":
    main()
