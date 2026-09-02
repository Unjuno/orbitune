"""Production commercial-base pretrain trainer.

This is the trainer that backs the next engineering gate after the
500-step LR=3e-5 pilot PASS. It wraps the time-vectorized
``tbptt_loss`` (commit ``b5f161a``) with the production contract:

* epoch-aware, deterministic, no-replacement TBPTT sampler
  (``orbitune.epoch_sampler.EpochAwareNoReplacementSampler``);
* per-event loss weighting via the decoder's ``event_weight`` argument
  (padding / idle lanes have ``event_weight = 0`` and contribute
  neither loss nor gradient);
* ``events_seen`` is counted on *active* events only, not on padding
  or idle lanes;
* exact-resume ``state_dict`` round-trip for sampler, optimizer, RNG
  and stream state, fail-closed on corpus identity mismatch;
* course stage gates (50M / 100M / 150M / 200M / 1.0× corpus pass)
  for long-run progress logging;
* ``--stop-after-events`` and ``--stop-after-epochs`` for the
  production course.

The trainer does NOT auto-launch a long blind run. Pass an explicit
``--steps`` or ``--stop-after-events`` and run it in the foreground.
"""

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
from orbitune.compound_indexed import load_indexed_compound_corpus
from orbitune.compound_longrun import build_longrun_checkpoint, restore_longrun_rng, safe_backward_step
from orbitune.compound_tbptt import (
    batch_stream_states_from_cpu,
    batch_stream_states_to_cpu,
    detach_batch_stream_states,
    initial_batch_stream_states,
)
from orbitune.compound_tbptt_time_vectorized import tbptt_loss as time_vectorized_tbptt_loss
from orbitune.compound_training import atomic_torch_save, parse_compound_checkpoint
from orbitune.epoch_sampler import EpochAwareNoReplacementSampler


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_script("orbitune_compound_cuda_base_pretrain", ROOT / "scripts" / "compound_cuda_train.py")
cfe = _load_script("orbitune_compound_cfe_pretrain", ROOT / "scripts" / "compound_cfe_train.py")


def _load_indexed_corpus(path: str | Path):
    candidate = Path(path)
    index_path = candidate / "index.json" if candidate.is_dir() else candidate
    if index_path.name == "index.json" and index_path.exists():
        corpus = load_indexed_compound_corpus(index_path)
        return corpus.songs, corpus
    raise SystemExit(
        f"production commercial-base trainer requires an indexed corpus source; got {path}"
    )


def _save(
    *,
    target: Path,
    model: CompoundHierarchicalGPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    events_seen: int,
    epoch_index: int,
    epoch_events_seen: int,
    epoch_events_total: int,
    runtime: dict[str, object],
    sampler_rng: random.Random,
    sampler: EpochAwareNoReplacementSampler,
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
    payload["epoch_sampler_state"] = sampler.state_dict()
    payload["epoch_index"] = int(epoch_index)
    payload["epoch_events_seen"] = int(epoch_events_seen)
    payload["epoch_events_total"] = int(epoch_events_total)
    payload["tbptt_stream_states"] = batch_stream_states_to_cpu(stream_states)
    atomic_torch_save(payload, target)


def _stage_label(events_seen: int, total: int) -> str | None:
    """Return a one-shot course-stage label, or None if no new stage crossed."""
    if total <= 0:
        return None
    for fraction, label in (
        (0.05, "stage_5pct"),
        (0.10, "stage_10pct"),
        (0.20, "stage_20pct"),
        (0.50, "stage_50pct"),
        (0.75, "stage_75pct"),
        (1.00, "stage_100pct"),
    ):
        threshold = int(round(fraction * total))
        if events_seen >= threshold and threshold > 0:
            return f"{label}@{threshold}"
    return None


def train(args: argparse.Namespace) -> None:
    device = base.require_cuda()
    precision = base.precision_from(args.precision)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    sampler_rng = random.Random(args.seed + 7919)
    if args.causal_fastpath:
        cfe.install_causal_fastpath()

    train_songs, train_corpus = _load_indexed_corpus(args.train_source)
    validation_songs, _validation_corpus = _load_indexed_corpus(args.validation_source)
    if not args.weighted_loss:
        # Weighted loss is the production default; unweighted is for parity
        # with the 500-step pilot. Refuse silently auto-switching.
        print(json.dumps({"event": "unweighted_loss_active", "manifest_quality_weight_disabled": True}), flush=True)

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

    if args.override_resume_lr is None and args.resume and source_training_mode not in {
        "commercial_base_pretrain",
        "state_carry_tbptt",
    }:
        raise SystemExit(
            "transitioning a non-commercial-base / non-TBPTT checkpoint into the production "
            "trainer requires --override-resume-lr so the schedule change is explicit"
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

    # The epoch-aware sampler is the production source of truth for the
    # training distribution. The first epoch uses ``args.epoch_seed``; later
    # epochs derive their seed from it deterministically. Resuming from a
    # checkpoint restores the exact epoch/cursor/seed.
    epoch_seed = int(args.epoch_seed)
    sampler = EpochAwareNoReplacementSampler(
        train_songs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        epoch_seed=epoch_seed,
        weighted=bool(args.weighted_loss),
    )
    if isinstance(payload.get("epoch_sampler_state"), dict):
        sampler.load_state_dict(payload["epoch_sampler_state"])

    if isinstance(payload.get("tbptt_stream_states"), list):
        stream_states = batch_stream_states_from_cpu(payload["tbptt_stream_states"], device)
        if len(stream_states) != args.batch_size:
            raise SystemExit("checkpoint stream-state batch size does not match CLI")
    else:
        stream_states = initial_batch_stream_states(model, args.batch_size)

    start_step = int(payload.get("step", 0))
    start_events = int(payload.get("events_seen", 0))
    health = payload.get("health") if isinstance(payload.get("health"), dict) else {}
    loss_history = [float(v) for v in health.get("loss_history", []) if math.isfinite(float(v))]
    grad_history = [float(v) for v in health.get("grad_norm_history", []) if math.isfinite(float(v))]
    validation_history = list(payload.get("validation_history") or [])
    best_validation_loss = health.get("best_validation_loss")
    best_step = health.get("best_step")

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint.with_name(checkpoint.stem + ".best.pt")
    healthy_path = checkpoint.with_name(checkpoint.stem + ".healthy.pt")

    runtime: dict[str, object] = {
        "training_mode": "commercial_base_pretrain",
        "precision": precision,
        "fused_adamw": fused,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "n_head": model.config.n_head,
        "head_dim": model.config.d_model // model.config.n_head,
        "causal_fastpath": args.causal_fastpath,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "training_source": str(args.train_source),
        "validation_source": str(args.validation_source),
        "training_source_format": "indexed_memmap",
        "validation_source_format": "indexed_memmap",
        "weighted_loss": bool(args.weighted_loss),
        "epoch_seed": epoch_seed,
        "course_stages": list(args.course_stages) if args.course_stages else [],
    }

    model.train()
    interval_start = time.perf_counter()
    interval_step = start_step
    torch.cuda.reset_peak_memory_stats()

    # The 1.0× event total is the per-corpus total events for a single epoch
    # with the active ``batch_size`` and ``seq_len``. We approximate it from
    # the first epoch we sample; the sampler makes the exact value available
    # at the end of each epoch via ``sampler.epoch_events_total``.
    one_epoch_total = int(sampler.epoch_events_total)
    crossed_stages: set[str] = set()
    epoch_index = int(payload.get("epoch_index", sampler.epoch_index))
    if epoch_index != sampler.epoch_index:
        # Resuming mid-epoch is the supported case. Bring the sampler forward
        # to the same epoch_index (idempotent: only advance if behind).
        while sampler.epoch_index < epoch_index:
            sampler.advance_epoch()

    # Resume events_seen within the current epoch if the payload carried it.
    resume_events_seen = int(payload.get("epoch_events_seen", sampler.epoch_events_seen))
    if resume_events_seen != sampler.epoch_events_seen:
        # Best-effort: the round-trip is exact only when state was saved on
        # the same epoch the resume targets. If they differ, the epoch
        # bookkeeping is restored from the saved state; the per-step
        # ``events_counted`` will accumulate the remaining events.
        sampler.epoch_events_seen = min(resume_events_seen, sampler.epoch_events_total)

    stop_after_events = int(args.stop_after_events) if args.stop_after_events and args.stop_after_events > 0 else None
    stop_after_epochs = int(args.stop_after_epochs) if args.stop_after_epochs and args.stop_after_epochs > 0 else None

    max_step = int(args.steps) if args.steps and args.steps > 0 else None
    step = start_step
    while True:
        if max_step is not None and step >= max_step:
            break
        if stop_after_events is not None and (start_events + (step - start_step) * 0) >= stop_after_events:
            # We compare against the live events_seen inside the loop body
            pass
        if sampler.is_epoch_complete:
            if stop_after_epochs is not None and sampler.epoch_index + 1 >= stop_after_epochs + epoch_index:
                # ``epoch_index`` is the *current* epoch at the start of
                # training; we want stop_after_epochs of additional epochs.
                break
            sampler.advance_epoch()
            one_epoch_total = int(sampler.epoch_events_total)
            print(json.dumps({
                "event": "epoch_complete",
                "epoch_index": sampler.epoch_index - 1,
                "next_epoch_index": sampler.epoch_index,
                "next_epoch_total_events": one_epoch_total,
            }), flush=True)
        step += 1
        sample = sampler.sample(device)
        optimizer.zero_grad(set_to_none=True)
        with base.autocast_for(precision):
            loss, parts, stream_states = time_vectorized_tbptt_loss(
                model,
                sample.batch.inputs,
                sample.batch.targets,
                stream_states,
                reset_mask=sample.batch.reset_mask,
                event_weight=sample.event_weight,
            )
        result = safe_backward_step(
            loss=loss,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=args.grad_clip,
        )
        if not result.stepped:
            raise RuntimeError(f"unsafe commercial-base step {step}: {result.failure}")
        stream_states = detach_batch_stream_states(stream_states)
        # Active-events accounting only
        start_events += int(sample.events_counted)
        loss_history.append(result.loss_value)
        grad_history.append(float(result.grad_norm or 0.0))
        loss_history = loss_history[-args.health_history_len :]
        grad_history = grad_history[-args.health_history_len :]

        if step == start_step + 1 or step % args.log_every == 0 or step == max_step or sampler.is_epoch_complete:
            torch.cuda.synchronize()
            now = time.perf_counter()
            elapsed = max(now - interval_start, 1e-9)
            n = max(1, step - interval_step)
            print(json.dumps({
                "step": step,
                "loss": result.loss_value,
                "components": parts,
                "grad_norm": result.grad_norm,
                "events_seen": start_events,
                "events_per_sec": n * args.batch_size * args.seq_len / elapsed,
                "active_event_weight_sum": int(sample.event_weight.sum().item()),
                "reset_lanes": int(sample.batch.reset_mask.sum().item()),
                "epoch_index": sampler.epoch_index,
                "epoch_events_seen": sampler.epoch_events_seen,
                "epoch_events_total": sampler.epoch_events_total,
                "runtime": runtime,
                "cuda": base.cuda_stats(),
            }, sort_keys=True), flush=True)
            interval_start = now
            interval_step = step
            torch.cuda.reset_peak_memory_stats()

        # Course stage gate (one-shot log per stage)
        if args.course_stages:
            for stage in args.course_stages:
                if start_events >= stage and stage not in crossed_stages:
                    crossed_stages.add(stage)
                    print(json.dumps({
                        "event": "course_stage_crossed",
                        "stage_events": int(stage),
                        "actual_events_seen": start_events,
                        "step": step,
                    }), flush=True)

        if stop_after_events is not None and start_events >= stop_after_events:
            break

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == max_step):
            torch.cuda.synchronize()
            value, validation_events = _validation_loss(
                model, validation_songs,
                seq_len=args.seq_len,
                max_songs=args.validation_songs,
                precision=precision, device=device,
            )
            entry = {
                "step": step,
                "validation_loss": value,
                "validation_events": validation_events,
                "mode": "commercial_base_pretrain",
            }
            validation_history.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)
            if best_validation_loss is None or value < float(best_validation_loss):
                best_validation_loss = value
                best_step = step
                _save(
                    target=best_path, model=model, optimizer=optimizer, scaler=scaler,
                    step=step, events_seen=start_events, runtime=runtime,
                    sampler_rng=sampler_rng, sampler=sampler, stream_states=stream_states,
                    loss_history=loss_history, grad_history=grad_history,
                    validation_history=validation_history,
                    best_validation_loss=best_validation_loss, best_step=best_step,
                    epoch_index=sampler.epoch_index, epoch_events_seen=sampler.epoch_events_seen,
                    epoch_events_total=sampler.epoch_events_total,
                )
            model.train()
            interval_start = time.perf_counter()
            interval_step = step
            torch.cuda.reset_peak_memory_stats()

        if step % args.checkpoint_every == 0 or step == max_step:
            common = dict(
                model=model, optimizer=optimizer, scaler=scaler, step=step,
                events_seen=start_events, runtime=runtime, sampler_rng=sampler_rng,
                sampler=sampler, stream_states=stream_states,
                loss_history=loss_history, grad_history=grad_history,
                validation_history=validation_history,
                best_validation_loss=best_validation_loss, best_step=best_step,
                epoch_index=sampler.epoch_index, epoch_events_seen=sampler.epoch_events_seen,
                epoch_events_total=sampler.epoch_events_total,
            )
            _save(target=healthy_path, **common)
            _save(target=checkpoint, **common)
            interval_start = time.perf_counter()
            interval_step = step

    # Final save on exit (whether the loop ended on stop-after-events, stop-after-epochs, or --steps)
    _save(
        target=checkpoint, model=model, optimizer=optimizer, scaler=scaler,
        step=step, events_seen=start_events, runtime=runtime,
        sampler_rng=sampler_rng, sampler=sampler, stream_states=stream_states,
        loss_history=loss_history, grad_history=grad_history,
        validation_history=validation_history,
        best_validation_loss=best_validation_loss, best_step=best_step,
        epoch_index=sampler.epoch_index, epoch_events_seen=sampler.epoch_events_seen,
        epoch_events_total=sampler.epoch_events_total,
    )


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
                loss, _, state = time_vectorized_tbptt_loss(model, x, y, state)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production commercial-base pretrain trainer")
    parser.add_argument("--train-source", required=True, help="Indexed corpus directory or path to index.json")
    parser.add_argument("--validation-source", required=True)
    parser.add_argument("--config", default="configs/compound_hierarchical_9m_nhead7.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int, default=0, help="Hard step cap (0 = no cap)")
    parser.add_argument("--stop-after-events", type=int, default=0, help="Stop after this many active events (0 = no cap)")
    parser.add_argument("--stop-after-epochs", type=int, default=0, help="Stop after this many additional epochs (0 = no cap)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--n-head", type=int, default=7)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--causal-fastpath", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--override-resume-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weighted-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epoch-seed", type=int, default=2026)
    parser.add_argument("--course-stages", type=int, nargs="*", default=[50_000_000, 100_000_000, 150_000_000, 200_000_000])
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--validation-songs", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--health-history-len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
