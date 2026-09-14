from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

import torch

from orbitune.compound import CompoundEventType
from orbitune.compound_base import CompoundHierarchicalGPT
from orbitune.tokenizer.compound_event import CompoundRecord


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _primer() -> CompoundRecord:
    # This matches the default primer used by CompoundHierarchicalGPT.generate_records
    # and the browser stream: a zero-delta 120 BPM tempo record.
    return CompoundRecord(
        event_type=int(CompoundEventType.TEMPO),
        channel=0,
        delta_coarse=0,
        delta_residual=0,
        a1=120,
        a2=0,
        a3=0,
        a4=0,
        duration_coarse=0,
        duration_residual=0,
        continuous_coarse=0,
        continuous_residual=0,
    )


def _audit_records(records: list[CompoundRecord]) -> dict[str, object]:
    programs = [0] * 16
    program_seen = [False] * 16
    banks = [(0, 0)] * 16

    event_types: collections.Counter[str] = collections.Counter()
    program_events: collections.Counter[int] = collections.Counter()
    bank_events: collections.Counter[str] = collections.Counter()
    note_channels: collections.Counter[int] = collections.Counter()
    effective_program_notes: collections.Counter[int] = collections.Counter()

    melodic_notes = 0
    drum_notes = 0
    default_program_notes = 0
    explicit_program0_notes = 0
    nonzero_program_notes = 0
    first_program_event_index: int | None = None

    for index, record in enumerate(records):
        kind = CompoundEventType(int(record.event_type))
        event_types[kind.name] += 1
        channel = int(record.channel)

        if kind == CompoundEventType.BANK:
            banks[channel] = (int(record.a1), int(record.a2))
            bank_events[f"{int(record.a1)}:{int(record.a2)}"] += 1
            continue

        if kind == CompoundEventType.PROGRAM:
            programs[channel] = int(record.a1)
            program_seen[channel] = True
            program_events[int(record.a1)] += 1
            if first_program_event_index is None:
                first_program_event_index = index
            continue

        if kind != CompoundEventType.NOTE:
            continue

        note_channels[channel] += 1
        if channel == 9:
            drum_notes += 1
            continue

        melodic_notes += 1
        current_program = programs[channel]
        effective_program_notes[current_program] += 1
        if not program_seen[channel]:
            default_program_notes += 1
        elif current_program == 0:
            explicit_program0_notes += 1
        else:
            nonzero_program_notes += 1

    return {
        "generated_events": len(records),
        "event_type_counts": dict(sorted(event_types.items())),
        "program_event_count": sum(program_events.values()),
        "program_event_counts": {str(k): v for k, v in sorted(program_events.items())},
        "bank_event_count": sum(bank_events.values()),
        "bank_event_counts": dict(sorted(bank_events.items())),
        "note_count": melodic_notes + drum_notes,
        "melodic_note_count": melodic_notes,
        "drum_note_count": drum_notes,
        "note_channel_counts": {str(k): v for k, v in sorted(note_channels.items())},
        "default_program_note_count": default_program_notes,
        "explicit_program0_note_count": explicit_program0_notes,
        "nonzero_program_note_count": nonzero_program_notes,
        "effective_program_note_counts": {str(k): v for k, v in sorted(effective_program_notes.items())},
        "unique_programs_emitted": sorted(program_events),
        "unique_nonzero_programs_emitted": sorted(p for p in program_events if p != 0),
        "first_program_event_index": first_program_event_index,
        "final_program_state": programs,
        "final_program_seen": program_seen,
        "final_bank_state": [list(pair) for pair in banks],
    }


def _merge_seed_results(per_seed: list[dict[str, object]]) -> dict[str, object]:
    numeric_keys = (
        "generated_events",
        "program_event_count",
        "bank_event_count",
        "note_count",
        "melodic_note_count",
        "drum_note_count",
        "default_program_note_count",
        "explicit_program0_note_count",
        "nonzero_program_note_count",
    )
    aggregate: dict[str, object] = {key: sum(int(item[key]) for item in per_seed) for key in numeric_keys}

    def merge_counter(key: str) -> dict[str, int]:
        counter: collections.Counter[str] = collections.Counter()
        for item in per_seed:
            counter.update({str(k): int(v) for k, v in dict(item[key]).items()})
        return dict(sorted(counter.items(), key=lambda pair: int(pair[0]) if pair[0].isdigit() else pair[0]))

    aggregate["event_type_counts"] = merge_counter("event_type_counts")
    aggregate["program_event_counts"] = merge_counter("program_event_counts")
    aggregate["bank_event_counts"] = merge_counter("bank_event_counts")
    aggregate["note_channel_counts"] = merge_counter("note_channel_counts")
    aggregate["effective_program_note_counts"] = merge_counter("effective_program_note_counts")

    melodic = int(aggregate["melodic_note_count"])
    notes = int(aggregate["note_count"])
    aggregate["default_program_note_fraction_of_melodic"] = (
        int(aggregate["default_program_note_count"]) / melodic if melodic else None
    )
    aggregate["explicit_program0_note_fraction_of_melodic"] = (
        int(aggregate["explicit_program0_note_count"]) / melodic if melodic else None
    )
    aggregate["nonzero_program_note_fraction_of_melodic"] = (
        int(aggregate["nonzero_program_note_count"]) / melodic if melodic else None
    )
    aggregate["drum_note_fraction_of_all_notes"] = int(aggregate["drum_note_count"]) / notes if notes else None
    aggregate["seeds_with_any_program_event"] = sum(int(item["program_event_count"]) > 0 for item in per_seed)
    aggregate["seeds_with_any_nonzero_program_note"] = sum(int(item["nonzero_program_note_count"]) > 0 for item in per_seed)
    aggregate["seeds_with_any_drum_note"] = sum(int(item["drum_note_count"]) > 0 for item in per_seed)
    aggregate["unique_programs_emitted"] = sorted({
        int(program)
        for item in per_seed
        for program in item["unique_programs_emitted"]
    })
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure instrumentation emitted by the frozen A2-512 Base.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--events-per-seed", type=int, default=512)
    parser.add_argument("--seeds", default="1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    expected_sha = args.checkpoint_sha256.lower()
    actual_sha = _sha256(checkpoint)
    if actual_sha != expected_sha:
        raise SystemExit(f"checkpoint SHA-256 mismatch: {actual_sha} != {expected_sha}")
    if args.events_per_seed <= 0:
        raise SystemExit("events-per-seed must be positive")

    seeds = [int(piece.strip()) for piece in args.seeds.split(",") if piece.strip()]
    if not seeds:
        raise SystemExit("at least one seed is required")

    model, payload = CompoundHierarchicalGPT.load_checkpoint(checkpoint, map_location="cpu")
    model.eval()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    per_seed: list[dict[str, object]] = []
    with torch.inference_mode():
        for seed in seeds:
            random.seed(seed)
            torch.manual_seed(seed)
            generated = model.generate_records(
                [_primer()],
                max_new_events=args.events_per_seed,
                temperature=args.temperature,
                top_p=args.top_p,
            )[1:]
            result = _audit_records(generated)
            result["seed"] = seed
            per_seed.append(result)
            print(json.dumps({
                "seed": seed,
                "program_events": result["program_event_count"],
                "notes": result["note_count"],
                "default_program_notes": result["default_program_note_count"],
                "explicit_program0_notes": result["explicit_program0_note_count"],
                "nonzero_program_notes": result["nonzero_program_note_count"],
                "drum_notes": result["drum_note_count"],
            }, sort_keys=True), flush=True)

    report = {
        "schema": "orbitune-a2-instrumentation-audit-v1",
        "checkpoint_sha256": actual_sha,
        "architecture": payload.get("architecture"),
        "tokenizer": payload.get("tokenizer"),
        "checkpoint_step": payload.get("step"),
        "model_config": payload.get("config"),
        "protocol": {
            "primer": "TEMPO 120 BPM, delta=0",
            "events_per_seed": args.events_per_seed,
            "seeds": seeds,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "aggregate": _merge_seed_results(per_seed),
        "per_seed": per_seed,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
