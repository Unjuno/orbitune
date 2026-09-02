"""Streaming-state validation for fixed-window and TBPTT checkpoints.

This evaluator runs the same generation-equivalent carried state used by the
TBPTT training path. It can therefore evaluate the frozen step-1900 fixed-window
checkpoint and later TBPTT checkpoints on a common stateful metric.

It is forward-only and resets state only at song boundaries. Full partial tails
are dropped so every target chunk has exactly ``seq_len`` events.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orbitune.compound import CompoundEventType
from orbitune.compound_base import CompoundHierarchicalGPT
from orbitune.compound_tbptt import detach_batch_stream_states, initial_batch_stream_states, tbptt_loss
from orbitune.compound_training import load_compound_jsonl


def _active_counts(target_records: torch.Tensor) -> dict[str, int]:
    event_type = target_records[..., 0]
    nevents = int(event_type.numel())
    a1 = torch.zeros_like(event_type, dtype=torch.bool)
    for kind in (0, 1, 2, 3, 4, 5, 8, 9):
        a1 |= event_type.eq(kind)
    a2 = event_type.eq(int(CompoundEventType.BANK)) | event_type.eq(int(CompoundEventType.TIME_SIGNATURE))
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
        "a1": int(a1.sum().item()),
        "a2": int(a2.sum().item()),
        "velocity": int(note.sum().item()),
        "duration": int(note.sum().item()),
        "control": int(control.sum().item()),
    }


def evaluate_streaming_validation(
    checkpoint: str | Path,
    validation_jsonl: str | Path,
    *,
    seq_len: int = 32,
    device: str | torch.device = "cuda",
    precision: str = "auto",
    max_songs: int = 0,
) -> dict[str, Any]:
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        device = "cpu"
    device = torch.device(device)

    model, payload = CompoundHierarchicalGPT.load_checkpoint(checkpoint, map_location="cpu")
    model.to(device).eval()
    songs = load_compound_jsonl(validation_jsonl)
    if max_songs > 0:
        songs = songs[:max_songs]

    stored_precision = str((payload.get("runtime") or {}).get("precision", ""))
    if precision == "auto":
        if stored_precision in {"bf16", "fp16", "fp32"}:
            precision = stored_precision
        else:
            precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if precision == "bf16" and device.type == "cuda":
        amp_dtype = torch.bfloat16
    elif precision == "fp16" and device.type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = None

    total_loss_sum = 0.0
    total_events = 0
    component_sums: dict[str, float] = {}
    component_active: dict[str, int] = {}
    song_rows: list[dict[str, Any]] = []
    started = time.time()

    with torch.no_grad():
        for song_index, song in enumerate(songs):
            states = initial_batch_stream_states(model, 1)
            song_loss_sum = 0.0
            song_events = 0
            offset = 0
            while offset + seq_len < len(song.records):
                window = torch.tensor(
                    song.records[offset : offset + seq_len + 1],
                    dtype=torch.long,
                    device=device,
                )
                x = window[:-1][None]
                y = window[1:][None]
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_dtype is not None,
                ):
                    loss, parts, states = tbptt_loss(model, x, y, states)
                value = float(loss.detach().float().cpu())
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite streaming validation loss in song {song_index}")
                counts = _active_counts(y)
                for name, part in parts.items():
                    count = counts.get(name, 0)
                    if count <= 0:
                        continue
                    component_sums[name] = component_sums.get(name, 0.0) + float(part) * count
                    component_active[name] = component_active.get(name, 0) + count
                total_loss_sum += value * seq_len
                total_events += seq_len
                song_loss_sum += value * seq_len
                song_events += seq_len
                states = detach_batch_stream_states(states)
                offset += seq_len
            song_rows.append(
                {
                    "song_index": song_index,
                    "path": song.path,
                    "sha256": song.sha256,
                    "events_evaluated": song_events,
                    "trainer_loss_event_weighted": None if song_events == 0 else song_loss_sum / song_events,
                }
            )

    if total_events == 0:
        raise ValueError("streaming validation produced zero events")

    per_component = {
        name: {
            "active_events": component_active[name],
            "mean_on_active_events": component_sums[name] / component_active[name],
        }
        for name in sorted(component_sums)
    }
    elapsed = time.time() - started
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload.get("step", 0)),
        "source_training_mode": (payload.get("runtime") or {}).get("training_mode"),
        "validation_jsonl": str(validation_jsonl),
        "seq_len": int(seq_len),
        "precision": precision,
        "songs_evaluated": len(songs),
        "total_events": total_events,
        "trainer_loss_event_weighted": total_loss_sum / total_events,
        "per_component": per_component,
        "song_results": song_rows,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--max-songs", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()

    result = evaluate_streaming_validation(
        args.checkpoint,
        args.validation_jsonl,
        seq_len=args.seq_len,
        device=args.device,
        precision=args.precision,
        max_songs=args.max_songs,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "song_results"}, indent=2))
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
