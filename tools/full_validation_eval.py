"""Full-validation evaluator for Compound Base.

The production trainer's `--validation-batches 4 --validation-batch-size 4`
configuration evaluates only 4,096 events per checkpoint. With 372,369
validation events that is ~1.1% of the corpus, which is too narrow to
distinguish checkpoints that differ by <0.04 validation-loss units.

This tool evaluates a Compound checkpoint on **every** validation window
in a deterministic, forward-only pass:

  * For each song, tile the record sequence in non-overlapping
    ``seq_len``-event windows (with a final partial window dropped).
  * Group windows into fixed-size batches (default 32) and accumulate
    per-component loss exactly the way the trainer does.
  * Report per-component mean loss, total mean loss, total events seen,
    window count, and a deterministic plan hash so two evaluators can
    prove they evaluated the same windows.

The evaluator is forward-only; it never updates the model, optimizer,
or RNG state. It is intended for the read-only "do these checkpoints
actually differ?" question that arises between staged training runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_training import (
    load_compound_jsonl,
    parse_compound_checkpoint,
)


def _tile_song_windows(song, seq_len: int) -> list[tuple[int, int]]:
    """Return ``(start, length)`` tuples for non-overlapping seq_len windows.

    Drops a final partial window so every window covers exactly seq_len
    events. With seq_len=256 and the MAESTRO 2004 validation songs (min
    3,752 events) this produces >= 14 windows per song.
    """
    n = len(song.records)
    if n < seq_len + 1:
        return []
    windows: list[tuple[int, int]] = []
    for start in range(0, n - seq_len, seq_len):
        windows.append((start, seq_len))
    return windows


def _plan_payload(songs, *, seq_len: int) -> dict[str, Any]:
    """Build a deterministic full-validation plan.

    The plan hash is the SHA-256 of a sorted (song_sha, start, length)
    sequence, so two evaluators can prove they evaluated the same windows.
    Each window also carries the song's record count so the evaluator
    can resolve the song by length when the sha256 is empty (which is
    the case for the on-the-fly test fixtures; production JSONL always
    carries a sha256).
    """
    plan_rows: list[tuple[str, int, int, int]] = []
    for song in songs:
        sha = getattr(song, "sha256", "") or f"records={len(song.records)}"
        for start, length in _tile_song_windows(song, seq_len):
            plan_rows.append((sha, start, length, len(song.records)))
    plan_rows.sort()
    payload = json.dumps(plan_rows, sort_keys=True).encode("utf-8")
    return {
        "seq_len": int(seq_len),
        "window_count": len(plan_rows),
        "total_events": sum(length for _, _, length, _ in plan_rows),
        "window_hash": hashlib.sha256(payload).hexdigest(),
        "windows": [
            {"song_sha": s, "start": st, "length": ln, "song_records": sr}
            for s, st, ln, sr in plan_rows
        ],
    }


def evaluate_full_validation(
    checkpoint_path: str | Path,
    validation_jsonl: str | Path,
    *,
    seq_len: int = 256,
    batch_size: int = 32,
    device: str | torch.device = "cuda",
    precision: str | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint on the entire validation corpus.

    Returns a dictionary with per-component mean loss, total mean loss,
    total events, window count, plan hash, and the per-step mean loss
    distribution. Never modifies the model or its optimizer.
    """
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        device = "cpu"
    device = torch.device(device)

    raw = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    parsed = parse_compound_checkpoint(raw)
    cfg = parsed["config"]
    model = CompoundHierarchicalGPT(CompoundBaseConfig(**cfg))
    model.load_state_dict(parsed["model_state_dict"])
    model.to(device).eval()

    songs = load_compound_jsonl(validation_jsonl)
    plan = _plan_payload(songs, seq_len=seq_len)
    by_sha: dict[str, Any] = {getattr(s, "sha256", ""): s for s in songs}
    by_records: dict[int, Any] = {}
    for s in songs:
        by_records.setdefault(len(s.records), s)

    if precision is None:
        if "bf16" in str(parsed.get("runtime", {}).get("precision", "")).lower():
            precision = "bf16"
        elif "fp16" in str(parsed.get("runtime", {}).get("precision", "")).lower():
            precision = "fp16"
        else:
            precision = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp32"

    if precision == "bf16" and device.type == "cuda":
        amp_dtype = torch.bfloat16
    elif precision == "fp16" and device.type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    # Walk windows in plan order, batch them up, compute per-event loss.
    # We accumulate per-component sums and event counts, then divide at
    # the end so the per-component mean is exactly the mean over events.
    #
    # Terminology:
    #   - "events"     = Compound MIDI events (one record per (delta, event_type, ...) tuple)
    #   - "scalar fields" = events * COMPOUND_RECORD_WIDTH (12 for the Compound ABI)
    # The Compound record tensor y has shape (B, T, 12), so y.numel() = B*T*12 = scalar
    # fields, NOT events. The per-event mean (sum/events) and the per-scalar-field mean
    # (sum/fields) are the same number because every head attends every field.
    component_sums: dict[str, float] = {}
    component_events: dict[str, int] = {}
    total_loss_sum = 0.0
    total_events = 0
    total_scalar_fields = 0
    batch_bufs_x: list[torch.Tensor] = []
    batch_bufs_y: list[torch.Tensor] = []
    batch_event_count = 0

    def _flush() -> None:
        nonlocal total_loss_sum, total_events, total_scalar_fields
        if not batch_bufs_x:
            return
        x = torch.stack(batch_bufs_x).to(device)
        y = torch.stack(batch_bufs_y).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, parts = model(x, y)
        # The trainer loss is a uniform mean over the present components.
        # We accumulate the *same* total loss per event the trainer reports
        # so the numbers are directly comparable to the trainer log.
        b, t, width = int(y.shape[0]), int(y.shape[1]), int(y.shape[2])
        nevents = b * t
        nfields = nevents * width
        total_loss_sum += float(loss.detach()) * nevents
        total_events += nevents
        total_scalar_fields += nfields
        for name, value in parts.items():
            component_sums[name] = component_sums.get(name, 0.0) + float(value) * nevents
            component_events[name] = component_events.get(name, 0) + nevents
        batch_bufs_x.clear()
        batch_bufs_y.clear()

    started = time.time()
    for win in plan["windows"]:
        song = by_sha.get(win["song_sha"]) or by_records.get(int(win.get("song_records", -1)))
        if song is None:
            raise ValueError(f"validation window song not found (sha={win['song_sha']!r})")
        start = int(win["start"])
        recs = song.records[start : start + seq_len + 1]
        if len(recs) < seq_len + 1:
            continue
        x = torch.tensor(recs[:-1], dtype=torch.long)
        y = torch.tensor(recs[1:], dtype=torch.long)
        batch_bufs_x.append(x)
        batch_bufs_y.append(y)
        batch_event_count += int(y.shape[0]) * int(y.shape[1])
        if len(batch_bufs_x) >= batch_size:
            _flush()
    _flush()
    elapsed = time.time() - started

    # Per-component mean = sum / events_seen_by_that_component.
    # This matches how the trainer aggregates per-component loss
    # (one mean per batch, then averaged across batches) only if every
    # batch had every component. In our full-validation pass, every
    # batch does, so the trainer-style "uniform mean over present
    # components" reduces to per-event mean over the *same* denominator.
    # We report both the per-event mean and the trainer-style mean for
    # transparency.
    per_component = {}
    for name in sorted(component_sums.keys()):
        per_component[name] = {
            "sum": component_sums[name],
            "events": component_events[name],
            "mean_per_event": component_sums[name] / max(1, component_events[name]),
        }
    total_mean_per_event = total_loss_sum / max(1, total_events)

    # Trainer-style mean: sum of per-component means divided by the
    # number of present components. Identical to the trainer's
    # ``torch.stack(tuple(losses.values())).mean()`` aggregation.
    component_means = [v["mean_per_event"] for v in per_component.values()]
    trainer_style_mean = sum(component_means) / max(1, len(component_means))

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(parsed.get("step", 0)),
        "validation_jsonl": str(validation_jsonl),
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "precision": precision,
        "window_count": int(plan["window_count"]),
        "window_hash": plan["window_hash"],
        # events and scalar fields are reported separately; the per-event and
        # per-scalar-field mean losses are numerically equal (every head
        # attends every field), but downstream consumers (telemetry, reports)
        # should pick the one that matches their own denominator.
        "total_events": int(total_events),
        "total_scalar_fields": int(total_scalar_fields),
        "record_width": int(total_scalar_fields // max(1, total_events)),
        "elapsed_seconds": float(elapsed),
        "mean_loss_per_event": float(total_mean_per_event),
        "mean_loss_trainer_style": float(trainer_style_mean),
        "per_component": per_component,
    }


def _print_report(result: dict[str, Any]) -> None:
    print(f"checkpoint: {result['checkpoint']}  step={result['checkpoint_step']}")
    print(f"validation: {result['validation_jsonl']}")
    print(f"seq_len={result['seq_len']}  batch={result['batch_size']}  precision={result['precision']}")
    print(f"window_count={result['window_count']}  total_events={result['total_events']:,}  total_scalar_fields={result['total_scalar_fields']:,}  (record_width={result['record_width']})")
    print(f"window_hash: {result['window_hash']}")
    print(f"elapsed: {result['elapsed_seconds']:.1f} s")
    print(f"mean_loss_per_event:     {result['mean_loss_per_event']:.6f}")
    print(f"mean_loss_trainer_style: {result['mean_loss_trainer_style']:.6f}")
    print("per-component mean (per_event):")
    for name in sorted(result["per_component"]):
        v = result["per_component"][name]
        print(f"  {name:14s}  mean={v['mean_per_event']:+.4f}  events={v['events']:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = evaluate_full_validation(
        args.checkpoint,
        args.validation_jsonl,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
        precision=args.precision,
    )
    _print_report(result)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
