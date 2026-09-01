from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import orbitune.compound_base as compound_base
from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_training import (
    COMPOUND_CHECKPOINT_SCHEMA_VERSION,
    assert_runtime_compatible,
    atomic_torch_save,
    build_compound_checkpoint,
    capture_validation_window_plan,
    execute_validation_window_plan,
    load_compound_jsonl,
    parse_compound_checkpoint,
    restore_cuda_rng_state,
)

ROOT = Path(__file__).resolve().parents[1]
_BASE_PATH = ROOT / "scripts" / "compound_cuda_train.py"
_SPEC = importlib.util.spec_from_file_location("orbitune_compound_cuda_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load {_BASE_PATH}")
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)

_ORIGINAL_CAUSAL_BIAS = compound_base._causal_bias
_ORIGINAL_ATTN_FORWARD = compound_base.MultiheadSelfAttention.forward
_FASTPATH_INSTALLED = False


def _optimized_attention_forward(self, x: torch.Tensor, attention_bias: torch.Tensor | None) -> torch.Tensor:
    batch, steps, width = x.shape
    q = self.q_proj(x).view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(batch, steps, self.n_head, self.head_dim).transpose(1, 2)
    q, k = compound_base._apply_rope(q, k)
    y = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_bias,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=attention_bias is None,
    )
    return self.out_proj(y.transpose(1, 2).contiguous().view(batch, steps, width))


def install_causal_fastpath() -> None:
    """Use SDPA's causal flag whenever the requested mask is exactly full causal.

    This avoids materializing a dense triangular bias and lets PyTorch select
    fused Flash/cuDNN SDPA kernels when the device/dtype/head geometry qualify.
    Sliding-window local attention keeps the original explicit mask.
    """
    global _FASTPATH_INSTALLED
    if _FASTPATH_INSTALLED:
        return

    def optimized_causal_bias(length: int, device: torch.device, *, window: int | None = None):
        if window is None or window >= length:
            return None
        return _ORIGINAL_CAUSAL_BIAS(length, device, window=window)

    compound_base._causal_bias = optimized_causal_bias
    compound_base.MultiheadSelfAttention.forward = _optimized_attention_forward
    _FASTPATH_INSTALLED = True


def uninstall_causal_fastpath() -> None:
    global _FASTPATH_INSTALLED
    compound_base._causal_bias = _ORIGINAL_CAUSAL_BIAS
    compound_base.MultiheadSelfAttention.forward = _ORIGINAL_ATTN_FORWARD
    _FASTPATH_INSTALLED = False


def config_with_heads(path: str | None, n_head: int) -> CompoundBaseConfig:
    cfg = base.config_from(path)
    cfg = replace(cfg, n_head=n_head)
    cfg.validate()
    return cfg


def candidate_head_counts(d_model: int, requested: list[int]) -> list[int]:
    result: list[int] = []
    for n_head in requested:
        if n_head <= 0 or d_model % n_head:
            continue
        head_dim = d_model // n_head
        if head_dim % 2:
            continue
        if n_head not in result:
            result.append(n_head)
    if not result:
        raise ValueError("no valid head counts")
    return result


def _backend_probe_for(cfg: CompoundBaseConfig, precision: str) -> dict[str, Any]:
    device = base.require_cuda()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    batch, seq = 2, min(128, cfg.local_window)
    q = torch.randn(batch, cfg.n_head, seq, cfg.d_model // cfg.n_head, device=device, dtype=dtype)
    result: dict[str, Any] = {
        "n_head": cfg.n_head,
        "head_dim": cfg.d_model // cfg.n_head,
        "dtype": str(dtype),
    }
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception as exc:
        result["probe_error"] = f"sdpa_kernel unavailable: {exc}"
        return result

    for label, backend in (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("cudnn", getattr(SDPBackend, "CUDNN_ATTENTION", None)),
    ):
        if backend is None:
            continue
        try:
            with sdpa_kernel(backend):
                out = F.scaled_dot_product_attention(q, q, q, is_causal=True, dropout_p=0.0)
                out.sum().item()
            result[label] = True
        except Exception as exc:
            result[label] = False
            result[f"{label}_reason"] = str(exc).splitlines()[0][:240]
    return result


def hardware(args: argparse.Namespace) -> None:
    device = base.require_cuda()
    precision = base.precision_from(args.precision)
    cfg = base.config_from(args.config)
    heads = candidate_head_counts(cfg.d_model, args.head_counts)
    payload = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "total_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
        "bf16": torch.cuda.is_bf16_supported(),
        "precision": precision,
        "baseline": {"n_head": cfg.n_head, "head_dim": cfg.d_model // cfg.n_head},
        "head_candidates": [],
    }
    for n_head in heads:
        candidate = config_with_heads(args.config, n_head)
        payload["head_candidates"].append(_backend_probe_for(candidate, precision))
    print(json.dumps(payload, indent=2, sort_keys=True))


def _measure_candidate(
    *,
    sampler,
    cfg: CompoundBaseConfig,
    batch_size: int,
    seq_len: int,
    precision: str,
    warmup_steps: int,
    measure_steps: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    seed: int,
    causal_fastpath: bool,
    compile_mode: str | None,
) -> dict[str, Any]:
    device = base.require_cuda()
    if causal_fastpath:
        install_causal_fastpath()
    else:
        uninstall_causal_fastpath()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    rng = random.Random(seed + 7919)
    model = CompoundHierarchicalGPT(cfg).to(device)
    if compile_mode:
        model.compile(mode=compile_mode)
    optimizer, fused = base.optimizer_for(model, learning_rate, weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    try:
        for _ in range(warmup_steps):
            x, y = sampler.sample(batch_size, seq_len, rng, device)
            base.train_step(model, optimizer, scaler, x, y, precision, grad_clip)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        last = None
        for _ in range(measure_steps):
            x, y = sampler.sample(batch_size, seq_len, rng, device)
            last, _ = base.train_step(model, optimizer, scaler, x, y, precision, grad_clip)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return {
            "status": "ok",
            "n_head": cfg.n_head,
            "head_dim": cfg.d_model // cfg.n_head,
            "causal_fastpath": causal_fastpath,
            "compile_mode": compile_mode,
            "precision": precision,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "tokens_per_microbatch": batch_size * seq_len,
            "parameters": model.parameter_count(),
            "fused_adamw": fused,
            "steps_per_sec": measure_steps / elapsed,
            "events_per_sec": batch_size * seq_len * measure_steps / elapsed,
            "loss": None if last is None else float(last),
            **base.cuda_stats(),
        }
    except torch.OutOfMemoryError:
        return {
            "status": "oom",
            "n_head": cfg.n_head,
            "head_dim": cfg.d_model // cfg.n_head,
            "causal_fastpath": causal_fastpath,
            "compile_mode": compile_mode,
            "precision": precision,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }
    finally:
        del model, optimizer, scaler
        torch.cuda.empty_cache()


def cfe(args: argparse.Namespace) -> None:
    base.require_cuda()
    precision = base.precision_from(args.precision)
    torch.set_float32_matmul_precision("high")
    base_cfg = base.config_from(args.config)
    heads = candidate_head_counts(base_cfg.d_model, args.head_counts)
    sampler = base.TensorSampler(load_compound_jsonl(args.train_jsonl))
    results: list[dict[str, Any]] = []

    for causal_fastpath in args.fastpaths:
        for n_head in heads:
            cfg = config_with_heads(args.config, n_head)
            for seq_len in args.seq_lens:
                for batch_size in args.batch_sizes:
                    row = _measure_candidate(
                        sampler=sampler,
                        cfg=cfg,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        precision=precision,
                        warmup_steps=args.warmup_steps,
                        measure_steps=args.measure_steps,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        grad_clip=args.grad_clip,
                        seed=args.seed,
                        causal_fastpath=causal_fastpath,
                        compile_mode=args.compile_mode if args.compile else None,
                    )
                    results.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
                    if row["status"] == "oom" or row.get("peak_reserved_fraction", 0.0) > args.max_vram_fraction:
                        break

    safe = [
        r
        for r in results
        if r["status"] == "ok"
        and r["peak_reserved_fraction"] <= args.max_vram_fraction
    ]
    if not safe:
        raise SystemExit("no safe CFE candidate")
    best = max(safe, key=lambda r: r["events_per_sec"])
    frontier = sorted(
        safe,
        key=lambda r: (-r["events_per_sec"], r["peak_reserved_fraction"]),
    )[: min(10, len(safe))]
    summary = {
        "definition": "Context Fit Envelope: measured Pareto region over head geometry, causal kernel path, sequence length, microbatch and precision under a VRAM headroom constraint.",
        "recommended": best,
        "frontier": frontier,
        "results": results,
    }
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)


@torch.inference_mode()
def validation_loss(
    model: CompoundHierarchicalGPT,
    validation_songs,
    *,
    plan_payload: dict[str, object],
    precision: str,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    """Run the captured validation plan against the model in eval mode.

    Returns ``(mean_loss, telemetry)`` where telemetry carries the plan
    metadata so callers can record which windows were used. Each validation
    call uses the exact same windows for the lifetime of the run (and across
    resumes that share the plan), which removes the per-step window noise
    that contaminated the legacy ``seed = args.seed + step`` strategy.
    """
    model.eval()
    batches = execute_validation_window_plan(validation_songs, plan_payload, device=device)
    losses: list[float] = []
    events_total = 0
    for x, y in batches:
        with base.autocast_for(precision):
            loss, _ = base.fast_loss(model, x, y)
        losses.append(float(loss))
        events_total += int(y.numel())
    model.train()
    mean_loss = sum(losses) / max(1, len(losses))
    telemetry = {
        "validation_seed": int(plan_payload["validation_seed"]),
        "validation_batches": int(plan_payload["batches"]),
        "validation_events": int(events_total),
        "validation_window_hash": str(plan_payload["window_hash"]),
        "validation_batch_size": int(plan_payload["batch_size"]),
        "validation_seq_len": int(plan_payload["seq_len"]),
    }
    return mean_loss, telemetry


def _looks_like_synthetic(path: str | Path) -> bool:
    text = str(path).replace("\\", "/").lower()
    return any(token in text for token in ("synthetic", "fixture", "cfe/"))


def train(args: argparse.Namespace) -> None:
    device = base.require_cuda()
    precision = base.precision_from(args.precision)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed + 7919)
    install_causal_fastpath() if args.causal_fastpath else uninstall_causal_fastpath()

    train_path = Path(args.train_jsonl)
    validation_path = Path(args.validation_jsonl)
    for label, path in (("train", train_path), ("validation", validation_path)):
        if _looks_like_synthetic(path) and not args.allow_synthetic:
            raise SystemExit(
                f"synthetic data detected in {label} JSONL path ({path}). "
                "Use --allow-synthetic explicitly for benchmark / smoke runs only."
            )

    train_songs = load_compound_jsonl(train_path)
    validation_songs = load_compound_jsonl(validation_path)
    train_sampler = base.TensorSampler(train_songs)

    checkpoint_path = Path(args.checkpoint)
    healthy_path = checkpoint_path.with_name(checkpoint_path.stem + ".healthy.pt")
    best_path = checkpoint_path.with_name(checkpoint_path.stem + ".best.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {}
    start_step = 0
    validation_history: list[dict[str, Any]] = []
    loss_history: list[float] = []
    grad_history: list[float] = []
    non_finite_loss_count = 0
    non_finite_grad_count = 0
    spike_events: list[dict[str, Any]] = []
    best_validation_loss: float | None = None
    best_step: int | None = None
    last_healthy_step: int | None = None
    last_healthy_events_seen: int | None = None
    validation_plan: dict[str, Any] | None = None

    if args.resume:
        # Load to CPU so the saved cuda_rng_state_all arrives as CPU uint8
        # tensors (PyTorch 2.5+ rejects CUDA uint8 tensors in
        # torch.cuda.set_rng_state_all with "RNG state must be a
        # torch.ByteTensor"). We move the model explicitly afterwards.
        model, raw_payload = CompoundHierarchicalGPT.load_checkpoint(
            args.resume, map_location="cpu"
        )
        payload = parse_compound_checkpoint(raw_payload)
        model.to(device)
        if args.n_head is not None and model.config.n_head != args.n_head:
            raise SystemExit(
                f"resume checkpoint n_head={model.config.n_head}, requested {args.n_head}"
            )
        stored = (payload.get("runtime") or {}).get("precision")
        if args.precision == "auto" and stored in {"bf16", "fp16", "fp32"}:
            precision = base.precision_from(stored)
    else:
        n_head = args.n_head if args.n_head is not None else base.config_from(args.config).n_head
        model = CompoundHierarchicalGPT(config_with_heads(args.config, n_head)).to(device)

    optimizer, fused = base.optimizer_for(model, args.learning_rate, args.weight_decay)
    if payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    if scaler.is_enabled() and payload.get("amp_scaler_state_dict"):
        scaler.load_state_dict(payload["amp_scaler_state_dict"])

    start_step = int(payload.get("step", 0))
    start_events = int(payload.get("events_seen", 0))
    if payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"].cpu())
    if payload.get("python_rng_state") is not None:
        random.setstate(payload["python_rng_state"])
    if payload.get("sampler_rng_state") is not None:
        rng.setstate(payload["sampler_rng_state"])
    if payload.get("cuda_rng_state_all") is not None:
        # No workaround: validate dtype/device and let errors propagate.
        restore_cuda_rng_state(payload["cuda_rng_state_all"])

    health = payload.get("health") or {}
    if isinstance(health, dict):
        loss_history = [
            float(x) for x in health.get("loss_history", []) if math.isfinite(float(x))
        ]
        grad_history = [
            float(x) for x in health.get("grad_norm_history", []) if math.isfinite(float(x))
        ]
        non_finite_loss_count = int(health.get("non_finite_loss_count", 0))
        non_finite_grad_count = int(health.get("non_finite_grad_count", 0))
        spike_events = list(health.get("spike_events", []))
        best_validation_loss = health.get("best_validation_loss")
        best_step = health.get("best_step")
        last_healthy_step = health.get("last_healthy_step")
        last_healthy_events_seen = health.get("last_healthy_events_seen")
    validation_history = list(payload.get("validation_history") or [])
    validation_plan = payload.get("validation_plan")

    if args.compile:
        model.compile(mode=args.compile_mode)

    runtime = {
        "precision": precision,
        "fused_adamw": fused,
        "compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "n_head": model.config.n_head,
        "head_dim": model.config.d_model // model.config.n_head,
        "causal_fastpath": args.causal_fastpath,
        "cfe": True,
        "training_jsonl": str(args.train_jsonl),
        "validation_jsonl": str(args.validation_jsonl),
    }

    drift = assert_runtime_compatible(
        payload.get("runtime"),
        cli_runtime=runtime,
        allow_runtime_change=args.allow_runtime_change,
    )
    if drift:
        print(
            json.dumps({"event": "runtime_drift_acknowledged", "drift": drift}, sort_keys=True),
            flush=True,
        )

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

    def _save_atomic(
        *,
        step: int,
        events_seen: int,
        validation_history_local: list[dict[str, Any]],
        plan_payload: dict[str, Any],
        best_val: float | None,
        best_step_local: int | None,
        last_healthy_local: int | None,
        last_healthy_events_local: int | None,
        target: Path,
    ) -> None:
        ckpt_payload = build_compound_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            step=step,
            events_seen=events_seen,
            runtime=runtime,
            rng=rng,
            loss_history=loss_history,
            grad_norm_history=grad_history,
            non_finite_loss_count=non_finite_loss_count,
            non_finite_grad_count=non_finite_grad_count,
            spike_events=spike_events,
            best_validation_loss=best_val,
            best_step=best_step_local,
            last_healthy_step=last_healthy_local,
            last_healthy_events_seen=last_healthy_events_local,
            validation_history=validation_history_local,
            validation_plan=plan_payload,
            source_commit=os.environ.get("ORBITUNE_SOURCE_COMMIT") or os.environ.get("GITHUB_SHA"),
        )
        atomic_torch_save(ckpt_payload, target)

    for step in range(start_step + 1, args.steps + 1):
        x, y = train_sampler.sample(args.batch_size, args.seq_len, rng, device)
        loss, parts = base.train_step(
            model, optimizer, scaler, x, y, precision, args.grad_clip
        )
        loss_value = float(loss.detach())
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            args.grad_clip,
        )
        grad_norm_value = float(grad_norm_tensor.detach())

        if not math.isfinite(loss_value):
            non_finite_loss_count += 1
        if not math.isfinite(grad_norm_value):
            non_finite_grad_count += 1
        loss_history.append(loss_value)
        grad_history.append(grad_norm_value)
        if len(loss_history) > args.health_history_len:
            loss_history = loss_history[-args.health_history_len:]
        if len(grad_history) > args.health_history_len:
            grad_history = grad_history[-args.health_history_len:]

        if (
            len(loss_history) >= args.spike_min_samples
            and math.isfinite(loss_value)
        ):
            mean = sum(loss_history) / len(loss_history)
            var = sum((v - mean) ** 2 for v in loss_history) / len(loss_history)
            std = math.sqrt(var)
            if std > 1e-12:
                zscore = (loss_value - mean) / std
                if zscore > args.spike_z_threshold:
                    spike_events.append(
                        {
                            "step": step,
                            "kind": "loss",
                            "loss": loss_value,
                            "zscore": float(zscore),
                            "gradient_norm": grad_norm_value,
                        }
                    )

        should_log = step == start_step + 1 or step % args.log_every == 0 or step == args.steps
        if should_log:
            torch.cuda.synchronize()
            now = time.perf_counter()
            n = max(1, step - interval_step)
            elapsed = max(1e-9, now - interval_start)
            stats = base.cuda_stats()
            message: dict[str, Any] = {
                "step": step,
                "loss": loss_value,
                "components": {k: float(v) for k, v in parts.items()},
                "events_per_sec": n * args.batch_size * args.seq_len / elapsed,
                "runtime": runtime,
                "cuda": stats,
                "health": {
                    "non_finite_loss_count": non_finite_loss_count,
                    "non_finite_grad_count": non_finite_grad_count,
                    "loss_history_len": len(loss_history),
                    "spike_count": len(spike_events),
                },
            }
            print(json.dumps(message, sort_keys=True), flush=True)
            interval_start = time.perf_counter()
            interval_step = step
            torch.cuda.reset_peak_memory_stats()

        if (
            validation_songs is not None
            and args.eval_every > 0
            and (step % args.eval_every == 0 or step == args.steps)
        ):
            torch.cuda.synchronize()
            value, telemetry = validation_loss(
                model,
                validation_songs,
                plan_payload=validation_plan,
                precision=precision,
                device=device,
            )
            telemetry_full = {"step": step, "validation_loss": value, **telemetry}
            print(json.dumps(telemetry_full, sort_keys=True), flush=True)
            validation_history.append(telemetry_full)
            if best_validation_loss is None or value < best_validation_loss:
                best_validation_loss = value
                best_step = step
                _save_atomic(
                    step=step,
                    events_seen=start_events + (step - start_step) * args.batch_size * args.seq_len,
                    validation_history_local=validation_history,
                    plan_payload=validation_plan,
                    best_val=best_validation_loss,
                    best_step_local=best_step,
                    last_healthy_local=last_healthy_step,
                    last_healthy_events_local=last_healthy_events_seen,
                    target=best_path,
                )
            interval_start = time.perf_counter()
            interval_step = step
            torch.cuda.reset_peak_memory_stats()

        if step % args.checkpoint_every == 0 or step == args.steps:
            events_seen = start_events + (step - start_step) * args.batch_size * args.seq_len
            last_healthy_step = step
            last_healthy_events_seen = events_seen
            _save_atomic(
                step=step,
                events_seen=events_seen,
                validation_history_local=validation_history,
                plan_payload=validation_plan,
                best_val=best_validation_loss,
                best_step_local=best_step,
                last_healthy_local=last_healthy_step,
                last_healthy_events_local=last_healthy_events_seen,
                target=healthy_path,
            )
            _save_atomic(
                step=step,
                events_seen=events_seen,
                validation_history_local=validation_history,
                plan_payload=validation_plan,
                best_val=best_validation_loss,
                best_step_local=best_step,
                last_healthy_local=last_healthy_step,
                last_healthy_events_local=last_healthy_events_seen,
                target=checkpoint_path,
            )
            interval_start = time.perf_counter()
            interval_step = step


def csv_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def csv_bools(value: str) -> list[bool]:
    mapping = {"0": False, "1": True, "false": False, "true": True, "off": False, "on": True}
    values: list[bool] = []
    for part in value.split(","):
        key = part.strip().lower()
        if key:
            if key not in mapping:
                raise argparse.ArgumentTypeError(f"invalid bool {part!r}")
            values.append(mapping[key])
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Context Fit Envelope tuning for Orbitune Compound Base")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("hardware")
    p.add_argument("--config", default="configs/compound_hierarchical_9m.json")
    p.add_argument("--precision", choices=("auto", "bf16", "fp16"), default="auto")
    p.add_argument("--head-counts", type=csv_ints, default=csv_ints("8,7,14"))
    p.set_defaults(func=hardware)

    p = sub.add_parser("cfe")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--config", default="configs/compound_hierarchical_9m.json")
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument("--head-counts", type=csv_ints, default=csv_ints("8,7,14"))
    p.add_argument("--fastpaths", type=csv_bools, default=csv_bools("false,true"))
    p.add_argument("--seq-lens", type=csv_ints, default=csv_ints("256,512"))
    p.add_argument("--batch-sizes", type=csv_ints, default=csv_ints("4,8,16,32,64,96,128"))
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--measure-steps", type=int, default=20)
    p.add_argument("--max-vram-fraction", type=float, default=0.92)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out")
    p.set_defaults(func=cfe)

    p = sub.add_parser("train")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--validation-jsonl", required=True)
    p.add_argument("--config", default="configs/compound_hierarchical_9m.json")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--resume")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--n-head", type=int)
    p.add_argument("--causal-fastpath", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--validation-batches", type=int, default=4)
    p.add_argument("--validation-batch-size", type=int, default=4)
    p.add_argument("--validation-seed", type=int, default=10001,
                   help="Fixed seed used to precompute the validation window plan.")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    p.add_argument("--allow-runtime-change", action="store_true",
                   help="Allow resuming from a checkpoint whose runtime "
                        "(n_head, seq_len, batch_size, precision, causal_fastpath) "
                        "differs from the current CLI. Without this flag such "
                        "drift aborts resume to keep results reproducible.")
    p.add_argument("--allow-synthetic", action="store_true",
                   help="Permit training against paths that contain 'synthetic', "
                        "'fixture' or 'cfe/' tokens. Intended for benchmark / "
                        "smoke runs only; production training must point at a "
                        "real corpus.")
    p.add_argument("--health-history-len", type=int, default=200,
                   help="Maximum entries kept in the rolling loss / grad history.")
    p.add_argument("--spike-min-samples", type=int, default=30)
    p.add_argument("--spike-z-threshold", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=train)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
