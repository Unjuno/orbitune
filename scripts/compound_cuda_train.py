from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from orbitune.compound import CompoundEventType
from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT, GaussianHead
from orbitune.compound_training import CompoundSong, load_compound_jsonl


class TensorSampler:
    def __init__(self, songs: list[CompoundSong]) -> None:
        self.songs = [torch.tensor(song.records, dtype=torch.long) for song in songs]

    def sample(self, batch: int, seq: int, rng: random.Random, device: torch.device):
        eligible = [song for song in self.songs if song.shape[0] >= seq + 1]
        if not eligible:
            raise ValueError("no song is long enough for requested seq_len")
        windows = []
        for _ in range(batch):
            song = rng.choice(eligible)
            start = rng.randrange(0, song.shape[0] - seq)
            windows.append(song[start : start + seq + 1])
        joined = torch.stack(windows).to(device)
        return joined[:, :-1], joined[:, 1:]


def config_from(path: str | None) -> CompoundBaseConfig:
    if not path:
        return CompoundBaseConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return CompoundBaseConfig(**raw)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    return torch.device("cuda")


def precision_from(name: str) -> str:
    if name == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if name == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16 unsupported")
    return name


def autocast_for(precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def masked_mean(values: torch.Tensor, mask: torch.Tensor):
    weight = mask.to(values.dtype)
    count = weight.sum()
    active = count.gt(0).to(values.dtype)
    return (values * weight).sum() / count.clamp_min(1), active


def gaussian_each(mean: torch.Tensor, log_scale: torch.Tensor, target: torch.Tensor):
    return 0.5 * (target - mean).square() * torch.exp(-2 * log_scale) + log_scale


def fast_loss(model: CompoundHierarchicalGPT, inputs: torch.Tensor, targets: torch.Tensor):
    out = model.decoder.forward_teacher(model.encode(inputs), targets)
    target = out["targets"]
    event = target["event_type"]
    losses: dict[str, torch.Tensor] = {}
    active: dict[str, torch.Tensor] = {}

    losses["event_type"] = F.cross_entropy(out["event_type"].reshape(-1, 10), event.reshape(-1))
    losses["channel"] = F.cross_entropy(out["channel"].reshape(-1, 16), target["channel"].reshape(-1))
    losses["delta"] = GaussianHead.loss(*out["delta"], target["delta"])
    for key in ("event_type", "channel", "delta"):
        active[key] = losses[key].new_ones(())

    a1_mask = torch.zeros_like(event, dtype=torch.bool)
    for kind in (0, 1, 2, 3, 4, 5, 8, 9):
        a1_mask |= event.eq(kind)
    a1_each = F.cross_entropy(
        out["a1"].reshape(-1, 1024), target["a1"].reshape(-1).clamp_max(1023), reduction="none"
    ).view_as(event)
    losses["a1"], active["a1"] = masked_mean(a1_each, a1_mask)

    a2_mask = event.eq(int(CompoundEventType.BANK)) | event.eq(int(CompoundEventType.TIME_SIGNATURE))
    a2_each = F.cross_entropy(
        out["a2"].reshape(-1, 1024), target["a2"].reshape(-1).clamp_max(1023), reduction="none"
    ).view_as(event)
    losses["a2"], active["a2"] = masked_mean(a2_each, a2_mask)

    note = event.eq(int(CompoundEventType.NOTE))
    losses["velocity"], active["velocity"] = masked_mean(
        gaussian_each(out["velocity"][0], out["velocity"][1], target["velocity"]), note
    )
    losses["duration"], active["duration"] = masked_mean(
        gaussian_each(out["duration"][0], out["duration"][1], target["duration"]), note
    )
    control = (
        event.eq(int(CompoundEventType.CC))
        | event.eq(int(CompoundEventType.PITCH_BEND))
        | event.eq(int(CompoundEventType.CHANNEL_PRESSURE))
        | event.eq(int(CompoundEventType.POLY_PRESSURE))
    )
    losses["control"], active["control"] = masked_mean(
        gaussian_each(out["control"][0], out["control"][1], target["control"]), control
    )
    num = torch.stack([losses[k] * active[k] for k in losses]).sum()
    den = torch.stack([active[k] for k in losses]).sum().clamp_min(1)
    return num / den, losses


def optimizer_for(model: torch.nn.Module, lr: float, wd: float):
    try:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=True), True
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd), False


def cuda_stats() -> dict[str, Any]:
    total = torch.cuda.get_device_properties(0).total_memory
    result: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "total_gib": total / 2**30,
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "peak_reserved_fraction": torch.cuda.max_memory_reserved() / max(1, total),
    }
    for name, key in (("utilization", "gpu_util_percent"), ("memory_usage", "memory_bw_percent")):
        fn = getattr(torch.cuda, name, None)
        if fn:
            try:
                result[key] = float(fn())
            except Exception:
                pass
    return result


def train_step(model, optimizer, scaler, inputs, targets, precision: str, grad_clip: float):
    optimizer.zero_grad(set_to_none=True)
    with autocast_for(precision):
        loss, parts = fast_loss(model, inputs, targets)
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    return loss, parts


def save_checkpoint(model, optimizer, scaler, path: Path, step: int, rng: random.Random, runtime: dict[str, Any]):
    payload = model.checkpoint_payload(
        optimizer=optimizer,
        step=step,
        source_commit=os.environ.get("ORBITUNE_SOURCE_COMMIT") or os.environ.get("GITHUB_SHA"),
        sampler_rng_state=rng.getstate(),
    )
    payload["amp_scaler_state_dict"] = scaler.state_dict() if scaler.is_enabled() else None
    payload["cuda_runtime"] = runtime
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def hardware(_: argparse.Namespace) -> None:
    require_cuda()
    print(json.dumps({
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "bf16": torch.cuda.is_bf16_supported(),
        "capability": list(torch.cuda.get_device_capability()),
        **cuda_stats(),
    }, indent=2, sort_keys=True))


def tune(args: argparse.Namespace) -> None:
    device = require_cuda(); precision = precision_from(args.precision)
    torch.set_float32_matmul_precision("high")
    songs = load_compound_jsonl(args.train_jsonl); sampler = TensorSampler(songs)
    results = []
    for batch in args.batch_sizes:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(args.seed); rng = random.Random(args.seed + 7919)
        model = CompoundHierarchicalGPT(config_from(args.config)).to(device)
        optimizer, fused = optimizer_for(model, args.learning_rate, args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        try:
            for _ in range(args.warmup_steps):
                x, y = sampler.sample(batch, args.seq_len, rng, device)
                train_step(model, optimizer, scaler, x, y, precision, args.grad_clip)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
            last = None
            for _ in range(args.measure_steps):
                x, y = sampler.sample(batch, args.seq_len, rng, device)
                last, _ = train_step(model, optimizer, scaler, x, y, precision, args.grad_clip)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - start
            row = {
                "status": "ok", "precision": precision, "batch_size": batch, "seq_len": args.seq_len,
                "fused_adamw": fused, "steps_per_sec": args.measure_steps / elapsed,
                "events_per_sec": batch * args.seq_len * args.measure_steps / elapsed,
                "loss": float(last) if last is not None else None, **cuda_stats(),
            }
        except torch.OutOfMemoryError:
            row = {"status": "oom", "precision": precision, "batch_size": batch, "seq_len": args.seq_len}
        results.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        del model, optimizer, scaler; torch.cuda.empty_cache()
        if row["status"] == "oom" or row.get("peak_reserved_fraction", 0) > args.max_vram_fraction:
            break
    safe = [r for r in results if r["status"] == "ok" and r["peak_reserved_fraction"] <= args.max_vram_fraction]
    if not safe:
        raise SystemExit("no safe candidate")
    best = max(safe, key=lambda r: r["events_per_sec"])
    summary = {"recommended": best, "underfilled_vram": best["peak_reserved_fraction"] < 0.60, "results": results}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)


def train(args: argparse.Namespace) -> None:
    device = require_cuda(); precision = precision_from(args.precision)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed); random.seed(args.seed); rng = random.Random(args.seed + 7919)
    sampler = TensorSampler(load_compound_jsonl(args.train_jsonl)); payload: dict[str, Any] = {}
    if args.resume:
        model, payload = CompoundHierarchicalGPT.load_checkpoint(args.resume, map_location=device); model.to(device)
        stored = (payload.get("cuda_runtime") or {}).get("precision")
        if args.precision == "auto" and stored in {"bf16", "fp16", "fp32"}: precision = precision_from(stored)
    else:
        model = CompoundHierarchicalGPT(config_from(args.config)).to(device)
    optimizer, fused = optimizer_for(model, args.learning_rate, args.weight_decay)
    if payload.get("optimizer_state_dict"): optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    if scaler.is_enabled() and payload.get("amp_scaler_state_dict"): scaler.load_state_dict(payload["amp_scaler_state_dict"])
    start_step = int(payload.get("step", 0))
    if payload.get("torch_rng_state") is not None: torch.set_rng_state(payload["torch_rng_state"].cpu())
    if payload.get("python_rng_state") is not None: random.setstate(payload["python_rng_state"])
    if payload.get("sampler_rng_state") is not None: rng.setstate(payload["sampler_rng_state"])
    if payload.get("cuda_rng_state_all") is not None: torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    if args.compile:
        model.compile(mode=args.compile_mode)
    runtime = {"precision": precision, "fused_adamw": fused, "compile": args.compile, "batch_size": args.batch_size, "seq_len": args.seq_len}
    checkpoint = Path(args.checkpoint); model.train(); interval_start = time.perf_counter(); interval_step = start_step
    torch.cuda.reset_peak_memory_stats()
    for step in range(start_step + 1, args.steps + 1):
        x, y = sampler.sample(args.batch_size, args.seq_len, rng, device)
        loss, parts = train_step(model, optimizer, scaler, x, y, precision, args.grad_clip)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            torch.cuda.synchronize(); now = time.perf_counter(); n = max(1, step - interval_step); elapsed = max(1e-9, now - interval_start)
            stats = cuda_stats(); message = {
                "step": step, "loss": float(loss), "components": {k: float(v) for k, v in parts.items()},
                "events_per_sec": n * args.batch_size * args.seq_len / elapsed, "runtime": runtime, "cuda": stats,
            }
            if step >= args.low_vram_warn_after and stats["peak_reserved_fraction"] < args.low_vram_fraction:
                message["warning"] = "low_vram_utilization"
            print(json.dumps(message, sort_keys=True), flush=True)
            interval_start = time.perf_counter(); interval_step = step; torch.cuda.reset_peak_memory_stats()
        if step % args.checkpoint_every == 0 or step == args.steps:
            save_checkpoint(model, optimizer, scaler, checkpoint, step, rng, runtime)
            interval_start = time.perf_counter(); interval_step = step


def csv_ints(value: str):
    return [int(x) for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CUDA tuning/training for Orbitune Compound Base")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("hardware"); p.set_defaults(func=hardware)
    p = sub.add_parser("tune"); p.add_argument("--train-jsonl", required=True); p.add_argument("--config")
    p.add_argument("--batch-sizes", type=csv_ints, default=csv_ints("4,8,16,32,64,96,128")); p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto"); p.add_argument("--warmup-steps", type=int, default=1); p.add_argument("--measure-steps", type=int, default=3)
    p.add_argument("--max-vram-fraction", type=float, default=0.92); p.add_argument("--learning-rate", type=float, default=3e-4); p.add_argument("--weight-decay", type=float, default=0.01); p.add_argument("--grad-clip", type=float, default=1.0); p.add_argument("--seed", type=int, default=1); p.add_argument("--out"); p.set_defaults(func=tune)
    p = sub.add_parser("train"); p.add_argument("--train-jsonl", required=True); p.add_argument("--config"); p.add_argument("--checkpoint", required=True); p.add_argument("--resume"); p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=8); p.add_argument("--seq-len", type=int, default=256); p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument("--learning-rate", type=float, default=3e-4); p.add_argument("--weight-decay", type=float, default=0.01); p.add_argument("--grad-clip", type=float, default=1.0); p.add_argument("--checkpoint-every", type=int, default=250); p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--compile", action="store_true"); p.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default"); p.add_argument("--low-vram-fraction", type=float, default=0.50); p.add_argument("--low-vram-warn-after", type=int, default=100); p.add_argument("--seed", type=int, default=1); p.set_defaults(func=train)
    return parser


def main() -> None:
    args = build_parser().parse_args(); args.func(args)


if __name__ == "__main__":
    main()
