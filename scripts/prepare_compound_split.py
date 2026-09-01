from __future__ import annotations

import argparse
import json

from orbitune.compound_dataset import prepare_compound_split_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare song-preserving Compound train/validation JSONL")
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--validation-out", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", default="orbitune-real-memory-smoke-v1")
    parser.add_argument("--min-events", type=int, default=32)
    args = parser.parse_args()
    report = prepare_compound_split_corpus(
        args.source,
        args.train_out,
        args.validation_out,
        args.report_out,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        min_events=args.min_events,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
