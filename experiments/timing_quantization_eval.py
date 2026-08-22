from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import mido

EDGES = [0, 24, 48, 96, 192, 384, 768, 1536]


def extract_timing(path: Path, steps_per_quarter: int = 96) -> tuple[list[int], list[int]]:
    midi = mido.MidiFile(path)
    scale = steps_per_quarter / midi.ticks_per_beat
    deltas: list[int] = []
    durations: list[int] = []

    for track in midi.tracks:
        time = 0
        onsets: list[int] = []
        active: dict[tuple[int, int], list[int]] = {}
        for message in track:
            time += message.time
            if message.type == "note_on" and message.velocity > 0:
                active.setdefault((message.channel, message.note), []).append(time)
                onsets.append(time)
            elif message.type in ("note_off", "note_on") and (
                message.type == "note_off" or message.velocity == 0
            ):
                key = (message.channel, message.note)
                starts = active.get(key)
                if starts:
                    start = starts.pop(0)
                    durations.append(max(1, round((time - start) * scale)))

        unique_onsets = sorted(set(onsets))
        deltas.extend(
            round((unique_onsets[i] - unique_onsets[i - 1]) * scale)
            for i in range(1, len(unique_onsets))
        )

    return deltas, durations


def coarse_residual_reconstruct(value: int, residual_levels: int) -> int:
    value = max(0, min(EDGES[-1], value))
    coarse = 0
    while coarse < len(EDGES) - 2 and value >= EDGES[coarse + 1]:
        coarse += 1
    lo, hi = EDGES[coarse], EDGES[coarse + 1]
    residual = round((value - lo) / (hi - lo) * (residual_levels - 1))
    residual = max(0, min(residual_levels - 1, residual))
    return round(lo + residual / (residual_levels - 1) * (hi - lo))


def log_levels(count: int, maximum: int = 1536) -> list[int]:
    levels = {0}
    if count <= 1:
        return [0]
    ratio = maximum ** (1 / max(1, count - 2))
    value = 1.0
    for _ in range(count - 1):
        levels.add(int(round(value)))
        value *= ratio
    levels.add(maximum)
    return sorted(levels)


def piecewise_levels() -> list[int]:
    levels = set(range(0, 97))
    levels.update(range(98, 193, 2))
    levels.update(range(196, 385, 4))
    levels.update(range(392, 769, 8))
    levels.update(range(784, 1537, 16))
    return sorted(levels)


def nearest(value: int, levels: list[int]) -> int:
    return min(levels, key=lambda candidate: abs(candidate - value))


def metrics(values: list[int], reconstruct) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mae_steps": None, "p95_steps": None, "exact_rate": None, "max_steps": None}
    errors = sorted(abs(reconstruct(value) - value) for value in values)
    return {
        "n": len(values),
        "mae_steps": mean(errors),
        "p95_steps": errors[int(0.95 * (len(errors) - 1))],
        "exact_rate": sum(error == 0 for error in errors) / len(errors),
        "max_steps": errors[-1],
    }


def evaluate(root: Path, steps_per_quarter: int = 96) -> dict:
    files = sorted({*root.rglob("*.mid"), *root.rglob("*.midi")})
    deltas: list[int] = []
    durations: list[int] = []
    failures: list[str] = []

    for path in files:
        try:
            d, u = extract_timing(path, steps_per_quarter=steps_per_quarter)
            deltas.extend(d)
            durations.extend(u)
        except Exception as exc:  # experiment should report bad corpus files, not abort
            failures.append(f"{path}: {exc}")

    schemes = {}
    for residual in (8, 12, 16, 24, 32):
        name = f"coarse7_res{residual}"
        reconstruct = lambda value, r=residual: coarse_residual_reconstruct(value, r)
        schemes[name] = {
            "vocab_per_attribute": 7 + residual,
            "attribute_heads": 2,
            "delta": metrics(deltas, reconstruct),
            "duration": metrics(durations, reconstruct),
        }

    for count in (64, 96, 128):
        levels = log_levels(count)
        schemes[f"log_{count}"] = {
            "vocab_per_attribute": len(levels),
            "attribute_heads": 1,
            "delta": metrics(deltas, lambda value, lv=levels: nearest(value, lv)),
            "duration": metrics(durations, lambda value, lv=levels: nearest(value, lv)),
        }

    levels = piecewise_levels()
    schemes["piecewise_sparse"] = {
        "vocab_per_attribute": len(levels),
        "attribute_heads": 1,
        "delta": metrics(deltas, lambda value: nearest(value, levels)),
        "duration": metrics(durations, lambda value: nearest(value, levels)),
    }

    return {
        "root": str(root),
        "steps_per_quarter": steps_per_quarter,
        "midi_files": len(files),
        "failed_files": failures,
        "delta_samples": len(deltas),
        "duration_samples": len(durations),
        "schemes": schemes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("midi_root", type=Path)
    parser.add_argument("--steps-per-quarter", type=int, default=96)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = evaluate(args.midi_root, args.steps_per_quarter)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
