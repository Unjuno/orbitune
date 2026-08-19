from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from orbitune.events import NoteEvent
from orbitune.midi import read_midi


def evaluate_events(events: list[NoteEvent]) -> dict[str, object]:
    if not events:
        return {
            "valid": False,
            "notes": 0,
            "bars": 0,
            "notes_per_bar_mean": 0.0,
            "pitch_min": None,
            "pitch_max": None,
            "velocity_mean": None,
            "duration_mean": None,
            "exact_event_repetition_ratio": 0.0,
        }

    bars = max(event.bar for event in events) + 1
    counts = Counter(event.bar for event in events)
    signatures = [(event.position, event.pitch, event.duration, event.velocity) for event in events]
    unique = len(set(signatures))
    repetition_ratio = 1.0 - unique / len(signatures)
    return {
        "valid": True,
        "notes": len(events),
        "bars": bars,
        "notes_per_bar_mean": len(events) / bars,
        "notes_per_bar_max": max(counts.values()),
        "pitch_min": min(event.pitch for event in events),
        "pitch_max": max(event.pitch for event in events),
        "velocity_mean": statistics.fmean(event.velocity for event in events),
        "duration_mean": statistics.fmean(event.duration for event in events),
        "exact_event_repetition_ratio": repetition_ratio,
    }


def evaluate_midi(path: str | Path) -> dict[str, object]:
    path = Path(path)
    events = read_midi(path)
    result = evaluate_events(events)
    result["path"] = str(path)
    return result


def write_evaluation(path: str | Path, out: str | Path) -> dict[str, object]:
    report = evaluate_midi(path)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
