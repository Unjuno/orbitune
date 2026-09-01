"""Re-measure top CFE candidates with longer warmup/measure steps and multiple trials."""
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

from scripts.compound_cuda_train import (  # noqa: E402
    TensorSampler,
    config_from,
    precision_from,
    require_cuda,
    optimizer_for,
    train_step,
    autocast_for,
    cuda_stats,
)
from scripts.compound_cfe_train import (  # noqa: E402
    install_causal_fastpath,
    uninstall_causal_fastpath,
)
from orbitune.compound_base import (  # noqa: E402
    CompoundBaseConfig,
    CompoundHierarchicalGPT,
)
from orbitune.compound_training import load_compound_jsonl  # noqa: E402


def measure(cfg: CompoundBaseConfig, sampler, *, batch_size, seq_len, precision,
            warmup, measure, seed, causal_fastpath, lr, wd, gc) -> dict:
    device = require_cuda()
    if causal_fastpath:
        install_causal_fastpath()
    else:
        uninstall_causal_fastpath()
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
        started = time.perf_counter()
        last = None
        for _ in range(measure):
            x, y = sampler.sample(batch_size, seq_len, rng, device)
            last, _ = train_step(model, optimizer, scaler, x, y, precision, gc)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return {
            "status": "ok",
            "n_head": cfg.n_head,
            "head_dim": cfg.d_model // cfg.n_head,
            "causal_fastpath": causal_fastpath,
            "precision": precision,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "tokens_per_microbatch": batch_size * seq_len,
            "parameters": model.parameter_count(),
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
            "causal_fastpath": causal_fastpath,
            "precision": precision,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }
    finally:
        del model, optimizer, scaler
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/compound_hierarchical_9m.json")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=50)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--out", default="runs/cfe_top_repeat.json")
    args = parser.parse_args()

    precision = precision_from(args.precision)
    base_cfg = config_from(args.config)
    sampler = TensorSampler(load_compound_jsonl(args.train_jsonl))

    # Top candidates from the initial sweep, plus n_head=14 and head_dim=28 baselines.
    candidates = [
        # (n_head, causal_fastpath, batch_size, seq_len, label)
        (7,  True,  128, 256, "A: n_head=7 hd=32 fast True  bs=128 seq=256"),
        (7,  False, 128, 256, "A2:n_head=7 hd=32 fast False bs=128 seq=256"),
        (8,  True,  128, 256, "B: n_head=8 hd=28 fast True  bs=128 seq=256 (baseline)"),
        (8,  False, 128, 256, "B2:n_head=8 hd=28 fast False bs=128 seq=256"),
        (7,  True,   96, 256, "C: n_head=7 hd=32 fast True  bs=96  seq=256 (safer)"),
        (14, True,   96, 256, "D: n_head=14 hd=16 fast True bs=96 seq=256"),
        (7,  True,   64, 512, "E: n_head=7 hd=32 fast True  bs=64 seq=512"),
    ]

    rows = []
    for n_head, fast, bs, seq, label in candidates:
        cfg = replace(base_cfg, n_head=n_head)
        cfg.validate()
        for trial in range(args.trials):
            row = measure(
                cfg=cfg,
                sampler=sampler,
                batch_size=bs,
                seq_len=seq,
                precision=precision,
                warmup=args.warmup,
                measure=args.measure,
                seed=1 + trial,
                causal_fastpath=fast,
                lr=args.learning_rate,
                wd=args.weight_decay,
                gc=args.grad_clip,
            )
            row["label"] = label
            row["trial"] = trial
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()