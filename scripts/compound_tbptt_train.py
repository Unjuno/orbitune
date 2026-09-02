from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch

from orbitune.compound_base import CompoundHierarchicalGPT
from orbitune.compound_longrun import build_longrun_checkpoint, restore_longrun_rng, safe_backward_step
from orbitune.compound_tbptt import (
    SequentialSongChunkSampler,
    batch_stream_states_from_cpu,
    batch_stream_states_to_cpu,
    detach_batch_stream_states,
    initial_batch_stream_states,
    tbptt_loss,
)
from orbitune.compound_training import atomic_torch_save, load_compound_jsonl, parse_compound_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_script("orbitune_compound_cuda_base_tbptt", ROOT / "scripts" / "compound_cuda_train.py")
cfe = _load_script("orbitune_compound_cfe_tbptt", ROOT / "scripts" / "compound_cfe_train.py")


def _looks_like_synthetic(path: str | Path) -> bool:
    text = str(path).replace("\\", "/").lower()
    return any(token in text for token in ("synthetic", "fixture", "cfe/"))


def _save(
    *,
    target: Path,
    model: CompoundHierarchicalGPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    events_seen: int,
    runtime: dict[str, object],
    sampler_rng: random.Random,
    sampler: SequentialSongChunkSampler,
    stream_states,
    loss_history: list[float],
    grad_history: list[float],
    validation_history: list[dict[str, object]],
    best_validation_loss: float | None,
    best_step: int | None,
) -> None:
    payload = build_longrun_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=step,
        events_seen=events_seen,
        runtime=runtime,
        sampler_rng=sampler_rng,
        loss_history=loss_history,
        grad_norm_history=grad_history,
        best_validation_loss=best_validation_loss,
        best_step=best_step,
        last_healthy_step=step,
        last_healthy_events_seen=events_seen,
        validation_history=validation_history,
        validation_plan=None,
        source_commit=os.environ.get("ORBITUNE_SOURCE_COMMIT") or os.environ.get("GITHUB_SHA"),
    )
    payload["tbptt_sampler_state"] = sampler.state_dict()
    payload["tbptt_stream_states"] = batch_stream_states_to_cpu(stream_states)
    atomic_torch_save(payload, target)


@torch.no_grad()
def _validation_loss(
    model: CompoundHierarchicalGPT,
    songs,
    *,
    seq_len: int,
    max_songs: int,
    precision: str,
    device: torch.device,
) -> tuple[float, int]:
    model.eval()
    total = 0.0
    events = 0
    selected = songs if max_songs <= 0 else songs[:max_songs]
    for song in selected:
        state = initial_batch_stream_states(model, 1)
        offset = 0
        while offset + seq_len < len(song.records):
            window = torch.tensor(song.records[offset : offset + seq_len + 1], dtype=torch.long, device=device)
            x = window[:-1][None]
            y = window[1:][None]
            with base.autocast_for(precision):
                loss, _, state = tbptt_loss(model, x, y, state)
            value = float(loss.detach().float().cpu())
            if not math.isfinite(value):
                raise RuntimeError("non-finite TBPTT validation loss")
            total += value * seq_len
            events += seq_len
            state = detach_batch_stream_states(state)
            offset += seq_len
    model.train()
    if events == 0:
        raise ValueError("TBPTT validation produced zero events")
    return total / events, events


def train(args: argparse.Namespace) -> None:
    if _looks_like_synthetic(args.train_jsonl) and not args.allow_synthetic:
        raise SystemExit("synthetic/fixture training data requires --allow-synthetic")
    if _looks_like_synthetic(args.validation_jsonl) and not args.allow_synthetic:
        raise SystemExit("synthetic/fixture validation data requires --allow-synthetic")
    if args.override_resume_lr is not None and not args.resume:
        raise SystemExit("--override-resume-lr requires --resume; refusing to silently retune a fresh run")

    device = base.require_cuda()
    precision = base.precision_from(args.precision)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    sampler_rng = random.Random(args.seed + 7919)
    if args.causal_fastpath:
        cfe.install_causal_fastpath()

    train_songs = load_compound_jsonl(args.train_jsonl)
    validation_songs = load_compound_jsonl(args.validation_jsonl)

    payload: dict[str, Any] = {}
    source_training_mode = "fresh"
    if args.resume:
        model, raw = CompoundHierarchicalGPT.load_checkpoint(args.resume, map_location="cpu")
        payload = parse_compound_checkpoint(raw)
        model.to(device)
        source_training_mode = str((payload.get("runtime") or {}).get("training_mode") or "legacy_fixed_window")
        stored_precision = (payload.get("runtime") or {}).get("precision")
        if args.precision == "auto" and stored_precision in {"bf16", "fp16", "fp32"}:
            precision = base.precision_from(stored_precision)
    else:
        config = cfe.config_with_heads(args.config, args.n_head)
        model = CompoundHierarchicalGPT(config).to(device)

    transitioning_from_fixed = bool(payload) and source_training_mode != "state_carry_tbptt"
    if transitioning_from_fixed and args.override_resume_lr is None:
        raise SystemExit(
            "transitioning a fixed-window checkpoint into state-carry TBPTT requires "
            "--override-resume-lr so the schedule change is explicit and auditable"
        )

    optimizer, fused = base.optimizer_for(model, args.learning_rate, args.weight_decay)
    if payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if args.override_resume_lr is not None:
        old_lrs = sorted({float(group.get("lr", 0.0)) for group in optimizer.param_groups})
        for group in optimizer.param_groups:
            group["lr"] = float(args.override_resume_lr)
        print(json.dumps({"event": "resume_lr_override_applied", "old_lrs": old_lrs, "new_lr": float(args.override_resume_lr)}), flush=True)

    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    if scaler.is_enabled() and payload.get("amp_scaler_state_dict"):
        scaler.load_state_dict(payload["amp_scaler_state_dict"])
    if payload:
        restore_longrun_rng(payload, sampler_rng)

    sampler = SequentialSongChunkSampler(
        train_songs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        rng=sampler_rng,
    )
    if not transitioning_from_fixed and isinstance(payload.get("tbptt_sampler_state"), dict):
        sampler.load_state_dict(payload["tbptt_sampler_state"])

    if not transitioning_from_fixed and isinstance(payload.get("tbptt_stream_states"), list):
        stream_states = batch_stream_states_from_cpu(payload["tbptt_stream_states"], device)
        if len(stream_states) != args.batch_size:
            raise SystemExit("TBPTT checkpoint stream-state batch size does not match CLI")
    else:
        stream_states = initial_batch_stream_states(model, args.batch_size)
        if payload:
            print(json.dumps({
                "event": "tbptt_state_initialized_from_song_boundaries",
                "source_step": int(payload.get("step", 0)),
                "source_training_mode": source_training_mode,
            }), flush=True)

    start_step = int(payload.get("step", 0))
    start_events = int(payload.get("events_seen", 0))
    health = payload.get("health") if isinstance(payload.get("health"), dict) else {}
    loss_history = [float(v) for v in health.get("loss_history", []) if math.isfinite(float(v))]
    grad_history = [float(v) for v in health.get("grad_norm_history", []) if math.isfinite(float(v))]
    validation_history = list(payload.get("validation_history") or [])
    best_validation_loss = health.get("best_validation_loss")
    best_step = health.get("best_step")
    if transitioning_from_fixed:
        # Fixed-window validation and TBPTT streaming validation are different
        # metrics. Do not let the old best value suppress TBPTT best checkpoints.
        loss_history = []
        grad_history = []
        validation_history = []
        best_validation_loss = None
        best_step = None

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint.with_name(checkpoint.stem + ".best.pt")
    healthy_path = checkpoint.with_name(checkpoint.stem + ".healthy.pt")

    runtime: dict[str, object] = {
        "training_mode": "state_carry_tbptt",
        "precision": precision,
        "fused_adamw": fused,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "n_head": model.config.n_head,
        "head_dim": model.config.d_model // model.config.n_head,
        "causal_fastpath": args.causal_fastpath,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "tbptt_state_semantics": "generation_equivalent_advance_stream",
        "tbptt_source_training_mode": source_training_mode,
        "tbptt_source_step": start_step,
    }

    model.train()
    interval_start = time.perf_counter()
    interval_step = start_step
    torch.cuda.reset_peak_memory_stats()

    for step in range(start_step + 1, args.steps + 1):
        batch = sampler.sample(device)
        optimizer.zero_grad(set_to_none=True)
        with base.autocast_for(precision):
            loss, parts, stream_states = tbptt_loss(
                model,
                batch.inputs,
                batch.targets,
                stream_states,
                reset_mask=batch.reset_mask,
            )
        result = safe_backward_step(
            loss=loss,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
        )
        if not result.stepped:
            raise RuntimeError(f"unsafe TBPTT step {step}: {result.failure}")
        stream_states = detach_batch_stream_states(stream_states)

        loss_history.append(result.loss_value)
        grad_history.append(float(result.grad_norm or 0.0))
        loss_history = loss_history[-args.health_history_len :]
        grad_history = grad_history[-args.health_history_len :]
        events_seen = start_events + (step - start_step) * args.batch_size * args.seq_len

        if step == start_step + 1 or step % args.log_every == 0 or step == args.steps:
            torch.cuda.synchronize()
            now = time.perf_counter()
            elapsed = max(now - interval_start, 1e-9)
            n = max(1, step - interval_step)
            print(json.dumps({
                "step": step,
                "loss": result.loss_value,
                "components": parts,
                "grad_norm": result.grad_norm,
                "events_seen": events_seen,
                "events_per_sec": n * args.batch_size * args.seq_len / elapsed,
                "reset_lanes": int(batch.reset_mask.sum().item()),
                "runtime": runtime,
                "cuda": base.cuda_stats(),
            }, sort_keys=True), flush=True)
            interval_start = now
            interval_step = step
            torch.cuda.reset_peak_memory_stats()

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
            torch.cuda.synchronize()
            value, validation_events = _validation_loss(
                model,
                validation_songs,
                seq_len=args.seq_len,
                max_songs=args.validation_songs,
                precision=precision,
                device=device,
            )
            entry = {"step": step, "validation_loss": value, "validation_events": validation_events, "mode": "state_carry_tbptt"}
            validation_history.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)
            if best_validation_loss is None or value < float(best_validation_loss):
                best_validation_loss = value
                best_step = step
                _save(
                    target=best_path, model=model, optimizer=optimizer, scaler=scaler,
                    step=step, events_seen=events_seen, runtime=runtime,
                    sampler_rng=sampler_rng, sampler=sampler, stream_states=stream_states,
                    loss_history=loss_history, grad_history=grad_history,
                    validation_history=validation_history,
                    best_validation_loss=best_validation_loss, best_step=best_step,
                )
            model.train()
            interval_start = time.perf_counter()
            interval_step = step
            torch.cuda.reset_peak_memory_stats()

        if step % args.checkpoint_every == 0 or step == args.steps:
            common = dict(
                model=model, optimizer=optimizer, scaler=scaler, step=step,
                events_seen=events_seen, runtime=runtime, sampler_rng=sampler_rng,
                sampler=sampler, stream_states=stream_states,
                loss_history=loss_history, grad_history=grad_history,
                validation_history=validation_history,
                best_validation_loss=best_validation_loss, best_step=best_step,
            )
            _save(target=healthy_path, **common)
            _save(target=checkpoint, **common)
            interval_start = time.perf_counter()
            interval_step = step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="State-carry TBPTT trainer for Orbitune Compound Base")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--config", default="configs/compound_hierarchical_9m_nhead7.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=7)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--causal-fastpath", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--override-resume-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--validation-songs", type=int, default=2, help="0 means all validation songs")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--health-history-len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--allow-synthetic", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
