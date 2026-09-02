"""MIDI A/B: generate N events from a checkpoint with a fixed seed.

Reproducibility: torch.manual_seed and random.seed are set BEFORE
loading the model. The model's generate_records uses torch RNG for
sampling. This makes the output byte-deterministic for the same
checkpoint, temperature, top_p, and events.

We don't decode to MIDI (slow). Instead we count events and check that
records have plausible types/values. The actual MIDI files would be
compared separately if needed; the headline metric here is "did the
model still produce valid outputs" and "do the event-type histograms
diverge sensibly between the A/B checkpoints".
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orbitune.compound_base import CompoundHierarchicalGPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--events", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    model, payload = CompoundHierarchicalGPT.load_checkpoint(args.checkpoint, map_location="cpu")
    model.to(device).eval()

    t0 = time.time()
    records = model.generate_records(
        primer=[],
        max_new_events=args.events,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    elapsed = time.time() - t0

    event_types = Counter(int(r.event_type) for r in records)
    channels = Counter(int(r.channel) for r in records)
    deltas = [int(r.delta_coarse) * 128 + int(r.delta_residual) for r in records]
    velocities = [int(r.a1) for r in records if int(r.event_type) in (1, 2, 3, 4, 5, 6, 7)]
    durations = [int(r.a1) for r in records if int(r.event_type) == 4]

    out = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": int(payload.get("step", 0)),
        "source_commit": payload.get("source_commit"),
        "seed": args.seed,
        "events_requested": args.events,
        "events_produced": len(records),
        "elapsed_seconds": elapsed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "event_type_counts": dict(event_types),
        "channel_counts": dict(channels),
        "delta_min": min(deltas) if deltas else None,
        "delta_max": max(deltas) if deltas else None,
        "delta_mean": sum(deltas) / len(deltas) if deltas else None,
        "velocity_min": min(velocities) if velocities else None,
        "velocity_max": max(velocities) if velocities else None,
        "velocity_mean": sum(velocities) / len(velocities) if velocities else None,
        "duration_min": min(durations) if durations else None,
        "duration_max": max(durations) if durations else None,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
