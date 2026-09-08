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

from orbitune.compound_base import CompoundHierarchicalGPT, set_padded_head_dim_sdpa
from orbitune.compound_longrun import build_longrun_checkpoint, restore_longrun_rng, safe_backward_step
from orbitune.compound_tbptt import (
    SequentialSongChunkSampler,
    batch_stream_states_from_cpu,
    batch_stream_states_to_cpu,
    detach_batch_stream_states,
    initial_batch_stream_states,
    tbptt_loss,
)
from orbitune.compound_tbptt_hybrid import tbptt_loss_hybrid
from orbitune.compound_training import atomic_torch_save, parse_compound_checkpoint
from orbitune.indexed_sampling import IndexedSequentialSongChunkSampler
from orbitune.tbptt_run_support import (
    guard_transition_outputs,
    load_training_sources,
    target_final_step,
    validate_resume_contract,
    validation_identity,
)

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


def _load_source(path: str | Path):
    sources = load_training_sources(path)
    return sources.songs, sources if sources.indexed else None


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
    sampler,
    stream_states,
    loss_history: list[float],
    grad_history: list[float],
    validation_history: list[dict[str, object]],
    best_validation_loss: float | None,
    best_step: int | None,
) -> None:
    payload = build_longrun_checkpoint(
        model=model, optimizer=optimizer, scaler=scaler, step=step,
        events_seen=events_seen, runtime=runtime, sampler_rng=sampler_rng,
        loss_history=loss_history, grad_norm_history=grad_history,
        best_validation_loss=best_validation_loss, best_step=best_step,
        last_healthy_step=step, last_healthy_events_seen=events_seen,
        validation_history=validation_history, validation_plan=None,
        validation_corpus_identity=str(runtime["validation_corpus_identity"]),
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
    try:
        for song in selected:
            state = initial_batch_stream_states(model, 1)
            offset = 0
            while offset + seq_len < len(song.records):
                window = torch.tensor(song.records[offset : offset + seq_len + 1], dtype=torch.long, device=device)
                x, y = window[:-1][None], window[1:][None]
                with base.autocast_for(precision):
                    loss, _, state = tbptt_loss(model, x, y, state)
                value = float(loss.detach().float().cpu())
                if not math.isfinite(value):
                    raise RuntimeError("non-finite TBPTT validation loss")
                total += value * seq_len
                events += seq_len
                state = detach_batch_stream_states(state)
                offset += seq_len
    finally:
        model.train()
    if events == 0:
        raise ValueError("TBPTT validation produced zero events")
    return total / events, events


def _validate_arguments(args: argparse.Namespace) -> None:
    for name in ("batch_size", "seq_len", "checkpoint_every", "log_every", "health_history_len"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.eval_every < 0 or args.validation_songs < 0:
        raise ValueError("eval_every and validation_songs must be nonnegative")
    for name in ("learning_rate", "grad_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if args.override_resume_lr is not None:
        if not args.resume:
            raise ValueError("--override-resume-lr requires --resume")
        if not math.isfinite(args.override_resume_lr) or args.override_resume_lr <= 0:
            raise ValueError("--override-resume-lr must be finite and positive")
    target_final_step(start_step=0, start_events=0, batch_size=args.batch_size,
                      seq_len=args.seq_len, steps=args.steps, target_events=args.target_events)


def train(args: argparse.Namespace) -> None:
    _validate_arguments(args)
    for source in (args.train_jsonl, args.validation_jsonl):
        if _looks_like_synthetic(source) and not args.allow_synthetic:
            raise SystemExit("synthetic/fixture training or validation data requires --allow-synthetic")

    device = base.require_cuda()
    precision = base.precision_from(args.precision)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    sampler_rng = random.Random(args.seed + 7919)
    cfe.install_causal_fastpath() if args.causal_fastpath else cfe.uninstall_causal_fastpath()
    set_padded_head_dim_sdpa(bool(args.padded_sdpa))
    print(json.dumps({"event": "tbptt_impl_selected", "impl": args.impl,
                      "padded_head_dim_sdpa": bool(args.padded_sdpa)}), flush=True)

    train_sources = load_training_sources(args.train_jsonl)
    validation_sources = load_training_sources(args.validation_jsonl)
    train_songs, validation_songs = train_sources.songs, validation_sources.songs
    if args.weighted_corpus_sampling and not train_sources.indexed:
        raise SystemExit("--weighted-corpus-sampling requires indexed corpus sources")

    payload: dict[str, Any] = {}
    source_training_mode = "fresh"
    if args.resume:
        model, raw = CompoundHierarchicalGPT.load_checkpoint(args.resume, map_location="cpu")
        # parse_compound_checkpoint supplies legacy defaults; do not mistake
        # its fallback step count for measured cumulative event exposure.
        if "events_seen" not in raw:
            raise ValueError("resume checkpoint lacks measured events_seen; explicit accounting migration required")
        payload = parse_compound_checkpoint(raw)
        model.to(device)
        source_training_mode = str((payload.get("runtime") or {}).get("training_mode") or "legacy_fixed_window")
        stored_precision = (payload.get("runtime") or {}).get("precision")
        if args.precision == "auto" and stored_precision in {"bf16", "fp16", "fp32"}:
            precision = base.precision_from(stored_precision)
        if args.n_head is not None and model.config.n_head != args.n_head:
            raise ValueError("resume checkpoint n_head does not match explicit CLI n_head")
        if payload.get("optimizer_state_dict") is None:
            raise ValueError("training continuation requires optimizer state")
    else:
        n_head = args.n_head if args.n_head is not None else base.config_from(args.config).n_head
        model = CompoundHierarchicalGPT(cfe.config_with_heads(args.config, n_head)).to(device)

    transitioning_from_fixed = bool(payload) and source_training_mode != "state_carry_tbptt"
    if transitioning_from_fixed:
        guard_transition_outputs(args.resume, args.checkpoint)
        if args.override_resume_lr is None:
            raise SystemExit("transitioning a fixed-window checkpoint into state-carry TBPTT requires --override-resume-lr")

    optimizer, fused = base.optimizer_for(model, args.learning_rate, args.weight_decay)
    if payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if args.override_resume_lr is not None:
        old_lrs = sorted({float(group.get("lr", 0.0)) for group in optimizer.param_groups})
        for group in optimizer.param_groups:
            group["lr"] = float(args.override_resume_lr)
        print(json.dumps({"event": "resume_lr_override_applied", "old_lrs": old_lrs,
                          "new_lr": float(args.override_resume_lr)}), flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    if scaler.is_enabled() and payload.get("amp_scaler_state_dict"):
        scaler.load_state_dict(payload["amp_scaler_state_dict"])

    start_step = int(payload.get("step", 0))
    start_events = int(payload.get("events_seen", 0))
    final_step = target_final_step(start_step=start_step, start_events=start_events,
                                  batch_size=args.batch_size, seq_len=args.seq_len,
                                  steps=args.steps, target_events=args.target_events)
    previous_runtime = payload.get("runtime") or {}
    runtime: dict[str, Any] = {
        "training_mode": "state_carry_tbptt", "device_type": device.type,
        "precision": precision, "fused_adamw": fused,
        "batch_size": args.batch_size, "seq_len": args.seq_len,
        "tbptt_impl": args.impl, "padded_head_dim_sdpa": bool(args.padded_sdpa),
        "n_head": model.config.n_head, "head_dim": model.config.d_model // model.config.n_head,
        "causal_fastpath": args.causal_fastpath, "grad_clip": args.grad_clip,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": [float(group["weight_decay"]) for group in optimizer.param_groups],
        "tbptt_state_semantics": "generation_equivalent_advance_stream",
        "tbptt_source_training_mode": source_training_mode, "tbptt_source_step": start_step,
        "tbptt_origin_step": previous_runtime.get("tbptt_origin_step", start_step),
        "tbptt_origin_events_seen": previous_runtime.get("tbptt_origin_events_seen", start_events),
        "training_source": str(args.train_jsonl), "validation_source": str(args.validation_jsonl),
        "training_source_format": train_sources.source_format,
        "validation_source_format": validation_sources.source_format,
        "training_corpus_identity": train_sources.identity,
        "validation_corpus_identity": validation_sources.identity,
        "validation_protocol_identity": validation_identity(validation_sources, seq_len=args.seq_len,
                                                            max_songs=args.validation_songs),
        "weighted_corpus_sampling": bool(args.weighted_corpus_sampling),
        "target_events": args.target_events, "target_step": final_step,
        "exposure_unit": "sampled_next_event_position",
        "unique_coverage_established": False,
    }
    if payload and not transitioning_from_fixed:
        validate_resume_contract(payload, runtime)
        prev = payload.get("runtime") or {}
        for key, want in (("tbptt_impl", args.impl),
                          ("padded_head_dim_sdpa", bool(args.padded_sdpa))):
            if key not in prev:
                raise ValueError(
                    f"TBPTT resume missing runtime identity {key}; legacy TBPTT "
                    "checkpoints need an explicit migration audit, not silent resume")
            if prev[key] != want:
                raise ValueError(
                    f"TBPTT resume regime mismatch for {key}: "
                    f"checkpoint={prev[key]!r} requested={want!r}")
    if transitioning_from_fixed and payload.get("sampler_rng_state") is None:
        print(json.dumps({"event": "legacy_sampler_rng_missing",
                          "exact_fixed_window_resume": False,
                          "reason": "new TBPTT lanes start at song boundaries; historical RNG cannot be reconstructed"}), flush=True)
    if payload:
        restore_longrun_rng(payload, sampler_rng)

    if train_sources.indexed:
        sampler = IndexedSequentialSongChunkSampler(train_songs, batch_size=args.batch_size,
                                                     seq_len=args.seq_len, rng=sampler_rng,
                                                     weighted=args.weighted_corpus_sampling)
    else:
        sampler = SequentialSongChunkSampler(train_songs, batch_size=args.batch_size,
                                             seq_len=args.seq_len, rng=sampler_rng)
    if payload and not transitioning_from_fixed:
        sampler.load_state_dict(payload["tbptt_sampler_state"])
        # Validate saved offsets against actual song lengths before any update.
        for index, offset in zip(sampler.song_indices, sampler.offsets):
            if index >= 0 and offset > len(train_songs[index].records) - 1:
                raise ValueError("TBPTT resume sampler offset exceeds song length")
        stream_states = batch_stream_states_from_cpu(payload["tbptt_stream_states"], device)
    else:
        stream_states = initial_batch_stream_states(model, args.batch_size)
        if payload:
            print(json.dumps({"event": "tbptt_state_initialized_from_song_boundaries",
                              "source_step": start_step, "source_training_mode": source_training_mode}), flush=True)

    health = payload.get("health") or {}
    loss_history = [float(v) for v in health.get("loss_history", []) if math.isfinite(float(v))]
    grad_history = [float(v) for v in health.get("grad_norm_history", []) if math.isfinite(float(v))]
    validation_history = list(payload.get("validation_history") or [])
    best_validation_loss, best_step = health.get("best_validation_loss"), health.get("best_step")
    if transitioning_from_fixed:
        loss_history, grad_history, validation_history = [], [], []
        best_validation_loss, best_step = None, None

    if final_step == start_step:
        print(json.dumps({"event": "target_already_met", "step": start_step,
                          "events_seen": start_events, "target_events": args.target_events}), flush=True)
        return
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint.with_name(checkpoint.stem + ".best.pt")
    healthy_path = checkpoint.with_name(checkpoint.stem + ".healthy.pt")
    events_seen = start_events
    model.train()
    interval_start, interval_step = time.perf_counter(), start_step
    torch.cuda.reset_peak_memory_stats()

    for step in range(start_step + 1, final_step + 1):
        batch = sampler.sample(device)
        optimizer.zero_grad(set_to_none=True)
        with base.autocast_for(precision):
            if args.impl == "hybrid":
                loss, parts, stream_states = tbptt_loss_hybrid(
                    model, batch.inputs, batch.targets,
                    stream_states, reset_mask=batch.reset_mask)
            else:
                loss, parts, stream_states = tbptt_loss(model, batch.inputs, batch.targets,
                                                        stream_states, reset_mask=batch.reset_mask)
        result = safe_backward_step(loss=loss, model=model, optimizer=optimizer,
                                    scaler=scaler, grad_clip=args.grad_clip)
        if not result.stepped:
            raise RuntimeError(f"unsafe TBPTT step {step}: {result.failure}")
        stream_states = detach_batch_stream_states(stream_states)
        loss_history.append(result.loss_value)
        grad_history.append(float(result.grad_norm or 0.0))
        loss_history = loss_history[-args.health_history_len:]
        grad_history = grad_history[-args.health_history_len:]
        # Count next-event targets, NOT the twelve scalar fields per record.
        events_seen += int(batch.targets.shape[0] * batch.targets.shape[1])
        final = step == final_step
        if step == start_step + 1 or step % args.log_every == 0 or final:
            torch.cuda.synchronize()
            now = time.perf_counter()
            elapsed = max(now - interval_start, 1e-9)
            print(json.dumps({
                "step": step, "loss": result.loss_value, "components": parts,
                "grad_norm": result.grad_norm, "events_seen": events_seen,
                "tbptt_events_seen": events_seen - int(runtime["tbptt_origin_events_seen"]),
                "events_per_sec": (step - interval_step) * args.batch_size * args.seq_len / elapsed,
                "reset_lanes": int(batch.reset_mask.sum().item()), "runtime": runtime,
                "cuda": base.cuda_stats(),
            }, sort_keys=True), flush=True)
            interval_start, interval_step = now, step
            torch.cuda.reset_peak_memory_stats()

        if args.eval_every > 0 and (step % args.eval_every == 0 or final):
            torch.cuda.synchronize()
            value, validation_events = _validation_loss(model, validation_songs, seq_len=args.seq_len,
                                                        max_songs=args.validation_songs,
                                                        precision=precision, device=device)
            entry = {"step": step, "validation_loss": value, "validation_events": validation_events,
                     "mode": "state_carry_tbptt", "validation_corpus_identity": validation_sources.identity,
                     "validation_protocol_identity": runtime["validation_protocol_identity"]}
            validation_history.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)
            if best_validation_loss is None or value < float(best_validation_loss):
                best_validation_loss, best_step = value, step
                _save(target=best_path, model=model, optimizer=optimizer, scaler=scaler,
                      step=step, events_seen=events_seen, runtime=runtime, sampler_rng=sampler_rng,
                      sampler=sampler, stream_states=stream_states, loss_history=loss_history,
                      grad_history=grad_history, validation_history=validation_history,
                      best_validation_loss=best_validation_loss, best_step=best_step)
            interval_start, interval_step = time.perf_counter(), step
            torch.cuda.reset_peak_memory_stats()

        if step % args.checkpoint_every == 0 or final:
            common = dict(model=model, optimizer=optimizer, scaler=scaler, step=step,
                          events_seen=events_seen, runtime=runtime, sampler_rng=sampler_rng,
                          sampler=sampler, stream_states=stream_states, loss_history=loss_history,
                          grad_history=grad_history, validation_history=validation_history,
                          best_validation_loss=best_validation_loss, best_step=best_step)
            _save(target=healthy_path, **common)
            _save(target=checkpoint, **common)
            interval_start, interval_step = time.perf_counter(), step
    print(json.dumps({"event": "training_target_reached", "step": final_step, "events_seen": events_seen,
                      "target_events": args.target_events,
                      "event_target_overshoot": None if args.target_events is None else events_seen - args.target_events,
                      "tbptt_events_seen": events_seen - int(runtime["tbptt_origin_events_seen"])}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="State-carry TBPTT trainer for Orbitune Compound Base")
    parser.add_argument("--train-jsonl", "--train-source", dest="train_jsonl", required=True,
                        help="Record JSONL, flat index, or comma-separated flat/sharded indexes in fixed order.")
    parser.add_argument("--validation-jsonl", "--validation-source", dest="validation_jsonl", required=True)
    parser.add_argument("--config", default="configs/compound_hierarchical_9m.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--steps", type=int, help="Final global optimizer step (legacy target mode).")
    target.add_argument("--target-events", "--target-events-seen", dest="target_events", type=int,
                        help="Cumulative sampled next-event positions INCLUDING inherited exposure; round up to a full batch.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=None,
                        help="On resume, verify rather than override the checkpoint architecture.")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--causal-fastpath", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--override-resume-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weighted-corpus-sampling", action="store_true",
                        help="Use existing indexed sampler weight semantics; not no-replacement coverage.")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--validation-songs", type=int, default=2,
                        help="Ordered first N validation songs; 0 means all. Default is only a smoke-size evaluation.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--health-history-len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--impl", choices=("reference", "hybrid"), default="reference",
                        help="TBPTT step implementation: reference per-lane loop or hybrid "
                             "(A3 local windows + exact per-position lane-batched hierarchies). "
                             "Recorded in runtime identity; regime changes require a new run, not silent resume.")
    parser.add_argument("--padded-sdpa", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable the flag-gated padded head-dim fast-SDPA path. "
                             "Recorded in runtime identity; changes dropout-mask streams.")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
