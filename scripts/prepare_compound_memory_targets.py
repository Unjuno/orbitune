from __future__ import annotations

import argparse
import json
from pathlib import Path

from orbitune.compound_memory_targets import (
    FAST_HORIZON_STEPS,
    MEDIUM_HORIZON_STEPS,
    MEMORY_TARGET_SCHEMA,
    derive_compound_memory_targets,
    target_cardinalities,
)
from orbitune.compound_training import load_compound_jsonl
from orbitune.tokenizer.compound_event import CompoundRecord


def prepare_memory_targets(
    inputs: list[str | Path],
    output: str | Path,
    report: str | Path,
) -> dict[str, object]:
    songs = load_compound_jsonl(inputs)
    target = Path(output)
    report_path = Path(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    events = 0
    with target.open("w", encoding="utf-8") as handle:
        for song in songs:
            records = [CompoundRecord(*values) for values in song.records]
            memory_targets = derive_compound_memory_targets(records)
            if len(memory_targets) != len(records):
                raise AssertionError("memory target alignment drifted from Compound records")
            payload = {
                "schema": MEMORY_TARGET_SCHEMA,
                "tokenizer_abi": song.tokenizer_abi,
                "path": song.path,
                "sha256": song.sha256,
                "records": len(records),
                "targets": [item.as_dict() for item in memory_targets],
            }
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            rows += 1
            events += len(records)

    summary: dict[str, object] = {
        "schema": MEMORY_TARGET_SCHEMA,
        "inputs": [str(path) for path in inputs],
        "output": str(target),
        "songs": rows,
        "events": events,
        "fast_horizon_steps": FAST_HORIZON_STEPS,
        "medium_horizon_steps": MEDIUM_HORIZON_STEPS,
        "slow_horizon": "causal composition prefix",
        "cardinalities": target_cardinalities(),
        "future_information_used": False,
    }
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive causal fast/medium/slow memory targets from Compound JSONL"
    )
    parser.add_argument("inputs", nargs="+", help="Compound train/validation JSONL files")
    parser.add_argument("--out", required=True, help="Output memory-target JSONL")
    parser.add_argument("--report", required=True, help="Output summary JSON")
    args = parser.parse_args()
    summary = prepare_memory_targets(args.inputs, args.out, args.report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
