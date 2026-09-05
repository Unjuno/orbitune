"""LR calibration for commercial v5 corpus on RTX 3080."""
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("orbitune_compound_cuda_base", ROOT / "scripts" / "compound_cuda_train.py")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

from orbitune.compound_base import CompoundHierarchicalGPT
from orbitune.compound_base import CompoundEventTokenizer
from orbitune.compound_training import load_compound_jsonl, sample_compound_batch

device = base.require_cuda()
torch.set_float32_matmul_precision("high")
cfg = base.config_from(str(ROOT / "configs/compound_hierarchical_9m.json"))
songs = load_compound_jsonl(str(ROOT / "benchmarks/fixtures/cfe/synthetic_compound.jsonl"))
tokenizer = CompoundEventTokenizer()
total_vram = torch.cuda.get_device_properties(device).total_memory / 2**30


def try_config(batch_size, seq_len, lr, precision, steps=8):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    model = CompoundHierarchicalGPT(cfg).to(device)
    opt, fused = base.optimizer_for(model, lr, 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    rng = random.Random(42)
    try:
        for _ in range(2):
            x, y = sample_compound_batch(songs, batch_size=batch_size, seq_len=seq_len, rng=rng, device=device)
            base.train_step(model, opt, scaler, x, y, precision, 1.0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        loss = None
        for _ in range(steps):
            x, y = sample_compound_batch(songs, batch_size=batch_size, seq_len=seq_len, rng=rng, device=device)
            loss, _ = base.train_step(model, opt, scaler, x, y, precision, 1.0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        stats = base.cuda_stats()
        return {
            "status": "ok",
            "batch_size": batch_size,
            "seq_len": seq_len,
            "lr": lr,
            "precision": precision,
            "steps": steps,
            "events_per_sec": batch_size * seq_len * steps / elapsed,
            "peak_reserved_gb": stats.get("peak_reserved_bytes", 0) / 2**30,
            "vram_fraction": stats.get("peak_reserved_bytes", 0) / 2**30 / total_vram,
            "loss": float(loss) if loss else None,
        }
    except torch.OutOfMemoryError:
        return {"status": "oom", "batch_size": batch_size, "seq_len": seq_len}
    finally:
        del model, opt, scaler
        torch.cuda.empty_cache()


# Phase 2a: CFE/VRAM sweep to find safe geometry
configs = [
    (4, 64, 3e-4, "bf16"),
    (8, 128, 3e-4, "bf16"),
    (8, 256, 3e-4, "bf16"),
    (16, 256, 3e-4, "bf16"),
]
print("=== CFE/VRAM Sweep ===", flush=True)
results = []
for bs, sl, lr, prec in configs:
    r = try_config(bs, sl, lr, prec, steps=5)
    print(json.dumps(r), flush=True)
    results.append(r)

# Pick safest high-throughput config
safe = [r for r in results if r.get("status") == "ok" and r["vram_fraction"] < 0.92]
if safe:
    best = max(safe, key=lambda r: r["events_per_sec"])
    chosen_bs = best["batch_size"]
    chosen_sl = best["seq_len"]
    print(f"\nCFE RECOMMENDED: batch={chosen_bs} seq_len={chosen_sl} prec=bf16 events/sec={best['events_per_sec']:.1f}")
else:
    chosen_bs, chosen_sl = 8, 256
    print("\nCFE fallback: batch=8 seq_len=256")

# Phase 2b: LR sweep at chosen geometry (loss stability)
print("\n=== LR Calibration ===", flush=True)
lr_results = []
for lr in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]:
    r = try_config(chosen_bs, chosen_sl, lr, "bf16", steps=12)
    print(json.dumps(r), flush=True)
    lr_results.append(r)

# Pick LR: prefer 3e-4 (default), check loss is finite
import math
lr_ok = [r for r in lr_results if r.get("status") == "ok"]
lr_finite = [r for r in lr_ok if r["loss"] is not None and math.isfinite(r["loss"])]
# lr=3e-4 is the production default; use it if loss is stable
best_lr = 3e-4
target = [r for r in lr_finite if abs(r["lr"] - 3e-4) < 1e-10]
if not target and lr_finite:
    best_lr = min(lr_finite, key=lambda r: r["loss"])["lr"]
elif target:
    best_lr = target[0]["lr"]
print(f"\nLR CALIBRATION RECOMMENDED: lr={best_lr}")

# Final recommendation summary
print("\n=== LR CALIBRATION SUMMARY ===", flush=True)
print(json.dumps({
    "BATCH_SIZE": chosen_bs,
    "SEQ_LEN": chosen_sl,
    "PRECISION": "bf16",
    "LR": best_lr,
    "WARMUP_STEPS": 50,
    "GRAD_ACCUM": "(see CFE; chosen_bs is microbatch)",
    "CHECKPOINT_INTERVAL": 250,
    "EVAL_INTERVAL": 250,
    "VRAM_FRACTION": best["vram_fraction"] if safe else "n/a",
    "PARAMETERS": "(computed at model init; see cfg)",
}, indent=2), flush=True)
