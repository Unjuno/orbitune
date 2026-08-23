from __future__ import annotations

import argparse
import json

from orbitune.compound_dataset import prepare_compound_split_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare experimental Hybrid Compound Event JSONL splits.")
    parser.add_argument("source")
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--validation-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", default="orbitune-compound-v0-experimental")
    parser.add_argument("--min-events", type=int, default=1)
    args = parser.parse_args()
    report = prepare_compound_split_corpus(
        args.source,
        args.train_out,
        args.validation_out,
        args.report,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        min_events=args.min_events,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
