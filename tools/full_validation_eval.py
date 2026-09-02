"""Full-validation evaluator for Compound Base.

The production trainer's narrow validation plan is useful for frequent checks
but too small to rank nearby checkpoints. This tool deterministically tiles the
validation corpus into non-overlapping full windows and performs a forward-only
comparison.

Two different accounting concepts are reported explicitly:

* ``total_events`` / ``total_scalar_fields`` describe corpus coverage only.
* ``mean_trainer_loss_event_weighted`` is the historical checkpoint-ranking
  metric: each batch's model loss is weighted by the number of Compound events
  in that batch. ``mean_loss_per_event`` is retained as a deprecated alias so
  existing JSON consumers keep working.
* Per-head losses are means over the events on which that head is *active*.
  NOTE duration/velocity, control and a1/a2 do not share the same denominator,
  so they must not be described as all-event or scalar-field means.

The evaluator never updates model weights, optimizer state or RNG state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from orbitune.compound import CompoundEventType
from orbitune.compound_base import CompoundBaseConfig, CompoundHierarchicalGPT
from orbitune.compound_training import load_compound_jsonl, parse_compound_checkpoint


def _tile_song_windows(song, seq_len: int) -> list[tuple[int, int]]:
    """Return full non-overlapping ``(start, length)`` validation windows."""
    n = len(song.records)
    if n < seq_len + 1:
        return []
    return [(start, seq_len) for start in range(0, n - seq_len, seq_len)]


def _plan_payload(songs, *, seq_len: int) -> dict[str, Any]:
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


def _active_head_counts(target_records: torch.Tensor) -> dict[str, int]:
    """Return the exact number of target events used by each decoder head."""
    event_type = target_records[..., 0]
    nevents = int(event_type.numel())

    a1_active = torch.zeros_like(event_type, dtype=torch.bool)
    for kind in (0, 1, 2, 3, 4, 5, 8, 9):
        a1_active |= event_type.eq(kind)

    a2_active = event_type.eq(int(CompoundEventType.BANK)) | event_type.eq(
        int(CompoundEventType.TIME_SIGNATURE)
    )
    note = event_type.eq(int(CompoundEventType.NOTE))
    control = (
        event_type.eq(int(CompoundEventType.CC))
        | event_type.eq(int(CompoundEventType.PITCH_BEND))
        | event_type.eq(int(CompoundEventType.CHANNEL_PRESSURE))
        | event_type.eq(int(CompoundEventType.POLY_PRESSURE))
    )
    return {
        "event_type": nevents,
        "channel": nevents,
        "delta": nevents,
        "a1": int(a1_active.sum().item()),
        "a2": int(a2_active.sum().item()),
        "velocity": int(note.sum().item()),
        "duration": int(note.sum().item()),
        "control": int(control.sum().item()),
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
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        device = "cpu"
    device = torch.device(device)

    raw = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    parsed = parse_compound_checkpoint(raw)
    model = CompoundHierarchicalGPT(CompoundBaseConfig(**parsed["config"]))
    model.load_state_dict(parsed["model_state_dict"])
    model.to(device).eval()

    songs = load_compound_jsonl(validation_jsonl)
    plan = _plan_payload(songs, seq_len=seq_len)
    by_sha: dict[str, Any] = {getattr(song, "sha256", ""): song for song in songs}
    by_records: dict[int, Any] = {}
    for song in songs:
        by_records.setdefault(len(song.records), song)

    if precision is None:
        stored = str(parsed.get("runtime", {}).get("precision", "")).lower()
        if "bf16" in stored:
            precision = "bf16"
        elif "fp16" in stored:
            precision = "fp16"
        else:
            precision = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp32"

    if precision == "bf16" and device.type == "cuda":
        amp_dtype = torch.bfloat16
    elif precision == "fp16" and device.type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    component_sums: dict[str, float] = {}
    component_active_events: dict[str, int] = {}
    trainer_loss_event_sum = 0.0
    total_events = 0
    total_scalar_fields = 0
    batch_bufs_x: list[torch.Tensor] = []
    batch_bufs_y: list[torch.Tensor] = []

    def _flush() -> None:
        nonlocal trainer_loss_event_sum, total_events, total_scalar_fields
        if not batch_bufs_x:
            return
        x = torch.stack(batch_bufs_x).to(device)
        y = torch.stack(batch_bufs_y).to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            loss, parts = model(x, y)

        b, t, width = int(y.shape[0]), int(y.shape[1]), int(y.shape[2])
        nevents = b * t
        trainer_loss_event_sum += float(loss.detach().float().cpu()) * nevents
        total_events += nevents
        total_scalar_fields += nevents * width

        active_counts = _active_head_counts(y)
        for name, value in parts.items():
            count = int(active_counts.get(name, 0))
            if count <= 0:
                continue
            component_sums[name] = component_sums.get(name, 0.0) + float(value) * count
            component_active_events[name] = component_active_events.get(name, 0) + count

        batch_bufs_x.clear()
        batch_bufs_y.clear()

    started = time.time()
    for win in plan["windows"]:
        song = by_sha.get(win["song_sha"]) or by_records.get(int(win.get("song_records", -1)))
        if song is None:
            raise ValueError(f"validation window song not found (sha={win['song_sha']!r})")
        start = int(win["start"])
        records = song.records[start : start + seq_len + 1]
        if len(records) < seq_len + 1:
            continue
        tensor = torch.tensor(records, dtype=torch.long)
        batch_bufs_x.append(tensor[:-1])
        batch_bufs_y.append(tensor[1:])
        if len(batch_bufs_x) >= batch_size:
            _flush()
    _flush()
    elapsed = time.time() - started

    per_component: dict[str, dict[str, float | int]] = {}
    for name in sorted(component_sums):
        active = component_active_events[name]
        mean = component_sums[name] / max(1, active)
        per_component[name] = {
            "sum_over_active_events": component_sums[name],
            "active_events": active,
            "mean_on_active_events": mean,
            # Backward-compatible aliases. These names were historically
            # misleading; new consumers should use the fields above.
            "sum": component_sums[name],
            "events": active,
            "mean_per_event": mean,
        }

    mean_trainer_loss_event_weighted = trainer_loss_event_sum / max(1, total_events)
    global_head_means = [float(value["mean_on_active_events"]) for value in per_component.values()]
    uniform_mean_of_global_head_means = sum(global_head_means) / max(1, len(global_head_means))

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(parsed.get("step", 0)),
        "validation_jsonl": str(validation_jsonl),
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "precision": precision,
        "window_count": int(plan["window_count"]),
        "window_hash": plan["window_hash"],
        "total_events": int(total_events),
        "total_scalar_fields": int(total_scalar_fields),
        "record_width": int(total_scalar_fields // max(1, total_events)),
        "elapsed_seconds": float(elapsed),
        "mean_trainer_loss_event_weighted": float(mean_trainer_loss_event_weighted),
        "uniform_mean_of_global_head_means": float(uniform_mean_of_global_head_means),
        # Deprecated compatibility aliases for historical result files.
        "mean_loss_per_event": float(mean_trainer_loss_event_weighted),
        "mean_loss_trainer_style": float(mean_trainer_loss_event_weighted),
        "per_component": per_component,
    }


def _print_report(result: dict[str, Any]) -> None:
    print(f"checkpoint: {result['checkpoint']}  step={result['checkpoint_step']}")
    print(f"validation: {result['validation_jsonl']}")
    print(f"seq_len={result['seq_len']}  batch={result['batch_size']}  precision={result['precision']}")
    print(
        f"window_count={result['window_count']}  total_events={result['total_events']:,}  "
        f"total_scalar_fields={result['total_scalar_fields']:,}  (record_width={result['record_width']})"
    )
    print(f"window_hash: {result['window_hash']}")
    print(f"elapsed: {result['elapsed_seconds']:.1f} s")
    print(f"mean_trainer_loss_event_weighted: {result['mean_trainer_loss_event_weighted']:.6f}")
    print(f"uniform_mean_of_global_head_means: {result['uniform_mean_of_global_head_means']:.6f}")
    print("per-component mean on active events:")
    for name in sorted(result["per_component"]):
        value = result["per_component"][name]
        print(
            f"  {name:14s}  mean={value['mean_on_active_events']:+.4f}  "
            f"active_events={value['active_events']:,}"
        )


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
