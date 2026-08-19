from __future__ import annotations

import json
from pathlib import Path

from orbitune.midi import read_midi
from orbitune.tokenizer import TheoryRemiTokenizer


def find_midi_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted({*root.rglob("*.mid"), *root.rglob("*.midi")})


def prepare_corpus(source: str | Path, out_tokens: str | Path, out_report: str | Path, *, min_events: int = 1) -> dict[str, object]:
    tokenizer = TheoryRemiTokenizer()
    files = find_midi_files(source)
    if not files:
        raise ValueError(f"no MIDI files found under {source}")

    sequences: list[list[str]] = []
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    for midi_path in files:
        try:
            events = read_midi(midi_path)
            if len(events) < min_events:
                rejected.append({"path": str(midi_path), "reason": "too_few_events"})
                continue
            tokens = tokenizer.encode_events(events)
            if not tokens:
                rejected.append({"path": str(midi_path), "reason": "empty_token_sequence"})
                continue
            sequences.append(tokens)
            accepted.append({"path": str(midi_path), "events": len(events), "tokens": len(tokens)})
        except (ValueError, IndexError, OSError) as exc:
            rejected.append({"path": str(midi_path), "reason": f"{type(exc).__name__}: {exc}"})

    if not accepted:
        raise ValueError("no usable MIDI files found")

    merged: list[str] = []
    for index, tokens in enumerate(sequences):
        if index:
            merged.extend(["EOS", "BOS"])
        merged.extend(tokens)

    out_tokens = Path(out_tokens)
    out_report = Path(out_report)
    out_tokens.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.write_tokens(merged, out_tokens)

    report: dict[str, object] = {
        "source": str(source),
        "files_seen": len(files),
        "files_accepted": len(accepted),
        "files_rejected": len(rejected),
        "song_boundaries": max(0, len(accepted) - 1),
        "total_events": sum(int(item["events"]) for item in accepted),
        "total_tokens": len(merged),
        "accepted": accepted,
        "rejected": rejected,
    }
    out_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
