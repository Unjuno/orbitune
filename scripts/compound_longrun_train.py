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
from orbitune.compound_training import (
    assert_runtime_compatible,
    atomic_torch_save,
    capture_validation_window_plan,
    load_compound_jsonl,
    parse_compound_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_script("orbitune_compound_cuda_base", ROOT / "scripts" / "compound_cuda_train.py")
cfe = _load_script("orbitune_compound_cfe", ROOT / "scripts" / "compound_cfe_train.py")


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
    loss_history: list[float],
    grad_history: list[float],
    non_finite_loss_count: int,
    non_finite_grad_count: int,
    spike_events: list[dict[str, object]],
    best_validation_loss: float | None,
    best_step: int | None,
    last_healthy_step: int | None,
    last_healthy_events_seen: int | None,
    validation_history: list[dict[str, object]],
    validation_plan: dict[str, object],
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
        non_finite_loss_count=non_finite_loss_count,
        non_finite_grad_count=non_finite_grad_count,
        spike_events=spike_events,
        best_validation_loss=best_validation_loss,
        best_step=best_step,
        last_healthy_step=last_healthy_step,
        last_healthy_events_seen=last_healthy_events_seen,
        validation_history=validation_history,
        validation_plan=validation_plan,
        source_commit=os.environ.get("ORBITUNE_SOURCE_COMMIT") or os.environ.get("GITHUB_SHA"),
    )
    atomic_torch_save(payload, target)


def train(args: argparse.Namespace) -> None:
    if not args.allow_fixed_window_training:
        raise SystemExit(
            "fixed-window training intentionally resets all history before each sampled window. "
            "Pass --allow-fixed-window-training only after accepting docs/STATE_CARRY_AUDIT.md."
        )

    for label, raw in (("train", args.train_jsonl), ("validation", args.validation_jsonl)):
        if _looks_like_synthetic(raw) and not args.allow_synthetic:
            raise SystemExit(
                f"synthetic/fixture data detected in {label} path {raw!r}; "
                "use --allow-synthetic only for smoke or benchmark runs"
            )

    device = base.require_cuda()
    precision = base.precision_from(args.precision)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    sampler_rng = random.Random(args.seed + 7919)
    cfe.install_causal_fastpath() if args.causal_fastpath else cfe.uninstall_causal_fastpath()

    train_songs = load_compound_jsonl(args.train_jsonl)
    validation_songs = load_compound_jsonl(args.validation_jsonl)
    train_sampler = base.TensorSampler(train_songs)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    healthy_path = checkpoint_path.with_name(checkpoint_path.stem + ".healthy.pt")
    best_path = checkpoint_path.with_name(checkpoint_path.stem + ".best.pt")

    payload: dict[str, Any] = {}
    if args.resume:
        model, raw_payload = CompoundHierarchicalGPT.load_checkpoint(args.resume, map_location="cpu")
        payload = parse_compound_checkpoint(raw_payload)
        model.to(device)
        if args.n_head is not None and model.config.n_head != args.n_head:
            raise SystemExit(
                f"resume checkpoint n_head={model.config.n_head}, requested n_head={args.n_head}"
            )
        stored_precision = (payload.get("runtime") or {}).get("precision")
        if args.precision == "auto" and stored_precision in {"bf16", "fp16", "fp32"}:
            precision = base.precision_from(stored_precision)
    else:
        n_head = args.n_head if args.n_head is not None else base.config_from(args.config).n_head
        model = CompoundHierarchicalGPT(cfe.config_with_heads(args.config, n_head)).to(device)

    optimizer, fused = base.optimizer_for(model, args.learning_rate, args.weight_decay)
    if payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    if scaler.is_enabled() and payload.get("amp_scaler_state_dict"):
        scaler.load_state_dict(payload["amp_scaler_state_dict"])

    if payload:
        restore_longrun_rng(payload, sampler_rng)

    start_step = int(payload.get("step", 0))
    start_events = int(payload.get("events_seen", 0))
    health = payload.get("health") if isinstance(payload.get("health"), dict) else {}
    loss_history = [float(v) for v in health.get("loss_history", []) if math.isfinite(float(v))]
    grad_history = [float(v) for v in health.get("grad_norm_history", []) if math.isfinite(float(v))]
    non_finite_loss_count = int(health.get("non_finite_loss_count", 0))
    non_finite_grad_count = int(health.get("non_finite_grad_count", 0))
    spike_events = list(health.get("spike_events", []))
    best_validation_loss = health.get("best_validation_loss")
    best_step = health.get("best_step")
    last_healthy_step = health.get("last_healthy_step")
    last_healthy_events_seen = health.get("last_healthy_events_seen")
    validation_history = list(payload.get("validation_history") or [])
    validation_plan = payload.get("validation_plan")

    runtime: dict[str, object] = {
        "precision": precision,
        "fused_adamw": fused,
        "compile": False,
        "compile_mode": None,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "n_head": model.config.n_head,
        "head_dim": model.config.d_model // model.config.n_head,
        "causal_fastpath": args.causal_fastpath,
        "training_mode": "fixed_window_explicit_opt_in",
        "training_jsonl": str(args.train_jsonl),
        "validation_jsonl": str(args.validation_jsonl),
    }
    drift = assert_runtime_compatible(
        payload.get("runtime"),
        cli_runtime=runtime,
        allow_runtime_change=args.allow_runtime_change,
    )
    if drift:
        print(json.dumps({"event": "runtime_drift_acknowledged", "drift": drift}, sort_keys=True), flush=True)

    if validation_plan is None:
        validation_plan = capture_validation_window_plan(
            validation_songs,
            validation_seed=args.validation_seed,
            batches=args.validation_batches,
            batch_size=min(args.batch_size, args.validation_batch_size),
            seq_len=args.seq_len,
        )

    model.train()
    interval_start = time.perf_counter()
    interval_step = start_step
    torch.cuda.reset_peak_memory_stats()

    for step in range(start_step + 1, args.steps + 1):
        x, y = train_sampler.sample(args.batch_size, args.seq_len, sampler_rng, device)
        optimizer.zero_grad(set_to_none=True)
        with base.autocast_for(precision):
            loss, parts = base.fast_loss(model, x, y)
        result = safe_backward_step(
            loss=loss,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
        )
        if not result.stepped:
            if result.failure == "non_finite_loss":
                non_finite_loss_count += 1
            else:
                non_finite_grad_count += 1
            failure = {
                "event": "training_aborted_before_optimizer_step",
                "step": step,
                "reason": result.failure,
                "loss": result.loss_value,
                "grad_norm": result.grad_norm,
                "last_healthy_step": last_healthy_step,
            }
            print(json.dumps(failure, sort_keys=True), flush=True)
            raise RuntimeError(f"unsafe training step {step}: {result.failure}")

        loss_value = result.loss_value
        grad_norm = float(result.grad_norm if result.grad_norm is not None else 0.0)
        prior = loss_history[-args.health_history_len :]
        if len(prior) >= args.spike_min_samples:
            mean = sum(prior) / len(prior)
            var = sum((value - mean) ** 2 for value in prior) / len(prior)
            std = math.sqrt(var)
            if std > 1e-12:
                zscore = (loss_value - mean) / std
                if zscore > args.spike_z_threshold:
                    spike_events.append(
                        {"step": step, "kind": "loss", "loss": loss_value, "zscore": zscore, "gradient_norm": grad_norm}
                    )
        loss_history.append(loss_value)
        grad_history.append(grad_norm)
        loss_history = loss_history[-args.health_history_len :]
        grad_history = grad_history[-args.health_history_len :]

        events_seen = start_events + (step - start_step) * args.batch_size * args.seq_len
        should_log = step == start_step + 1 or step % args.log_every == 0 or step == args.steps
        if should_log:
            torch.cuda.synchronize()
            now = time.perf_counter()
            elapsed = max(1e-9, now - interval_start)
            n = max(1, step - interval_step)
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss_value,
                        "components": {key: float(value.detach().float().cpu()) for key, value in parts.items()},
                        "grad_norm": grad_norm,
                        "events_seen": events_seen,
                        "events_per_sec": n * args.batch_size * args.seq_len / elapsed,
                        "runtime": runtime,
                        "cuda": base.cuda_stats(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            interval_start = now
            interval_step = step
            torch.cuda.reset_peak_memory_stats()

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
            torch.cuda.synchronize()
            value, telemetry = cfe.validation_loss(
                model,
                validation_songs,
                plan_payload=validation_plan,
                precision=precision,
                device=device,
            )
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite validation loss at step {step}")
            entry: dict[str, object] = {"step": step, "validation_loss": value, **telemetry}
            validation_history.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)
            if best_validation_loss is None or value < float(best_validation_loss):
                best_validation_loss = value
                best_step = step
                _save(
                    target=best_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    events_seen=events_seen,
                    runtime=runtime,
                    sampler_rng=sampler_rng,
                    loss_history=loss_history,
                    grad_history=grad_history,
                    non_finite_loss_count=non_finite_loss_count,
                    non_finite_grad_count=non_finite_grad_count,
                    spike_events=spike_events,
                    best_validation_loss=best_validation_loss,
                    best_step=best_step,
                    last_healthy_step=last_healthy_step,
                    last_healthy_events_seen=last_healthy_events_seen,
                    validation_history=validation_history,
                    validation_plan=validation_plan,
                )
            interval_start = time.perf_counter()
            interval_step = step
            torch.cuda.reset_peak_memory_stats()

        if step % args.checkpoint_every == 0 or step == args.steps:
            last_healthy_step = step
            last_healthy_events_seen = events_seen
            common = dict(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                events_seen=events_seen,
                runtime=runtime,
                sampler_rng=sampler_rng,
                loss_history=loss_history,
                grad_history=grad_history,
                non_finite_loss_count=non_finite_loss_count,
                non_finite_grad_count=non_finite_grad_count,
                spike_events=spike_events,
                best_validation_loss=best_validation_loss,
                best_step=best_step,
                last_healthy_step=last_healthy_step,
                last_healthy_events_seen=last_healthy_events_seen,
                validation_history=validation_history,
                validation_plan=validation_plan,
            )
            _save(target=healthy_path, **common)
            _save(target=checkpoint_path, **common)
            interval_start = time.perf_counter()
            interval_step = step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production long-run trainer for Orbitune Compound Base")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--config", default="configs/compound_hierarchical_9m_nhead7.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=144)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=7)
    parser.add_argument("--causal-fastpath", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--validation-seed", type=int, default=10001)
    parser.add_argument("--health-history-len", type=int, default=200)
    parser.add_argument("--spike-min-samples", type=int, default=30)
    parser.add_argument("--spike-z-threshold", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--allow-runtime-change", action="store_true")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument(
        "--allow-fixed-window-training",
        action="store_true",
        help="Acknowledge the documented train/generation history-state gap. Required until state-carry TBPTT is implemented.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
