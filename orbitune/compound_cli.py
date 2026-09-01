from __future__ import annotations

import argparse
import json
import sys

from orbitune.compound_base import generate_command, inspect_command, train_command
from orbitune.compound_dataset import prepare_compound_split_corpus


def _prepare_command(args: argparse.Namespace) -> None:
    report = prepare_compound_split_corpus(
        args.source,
        args.train_out,
        args.validation_out,
        args.report,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        min_events=args.min_events,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _add_training_args(parser: argparse.ArgumentParser, *, resume: bool) -> None:
    parser.add_argument("--train-jsonl", default="data/compound/train.jsonl")
    parser.add_argument("--validation-jsonl", default="data/compound/validation.jsonl")
    parser.add_argument("--checkpoint", default="models/compound-base.pt")
    if not resume:
        parser.add_argument("--config", default="configs/compound_hierarchical_9m.json")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")


def _add_info_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser], name: str, help_text: str) -> None:
    info = sub.add_parser(name, help=help_text)
    group = info.add_mutually_exclusive_group()
    group.add_argument("--config", default=None)
    group.add_argument("--checkpoint", default=None)
    info.set_defaults(func=inspect_command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orbitune-compound",
        description="Local-first Compound Transformer Base: prepare, train, resume and generate MIDI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="convert a MIDI directory to song-preserving Compound JSONL")
    prepare.add_argument("source")
    prepare.add_argument("--train-out", default="data/compound/train.jsonl")
    prepare.add_argument("--validation-out", default="data/compound/validation.jsonl")
    prepare.add_argument("--report", default="data/compound/report.json")
    prepare.add_argument("--validation-fraction", type=float, default=0.1)
    prepare.add_argument("--split-seed", default="orbitune-compound-base-v1")
    prepare.add_argument("--min-events", type=int, default=8)
    prepare.set_defaults(func=_prepare_command)

    train = sub.add_parser("train", help="start Compound Base training")
    _add_training_args(train, resume=False)
    train.set_defaults(func=train_command, resume=None)

    resume = sub.add_parser("resume", help="continue exactly from a saved Compound Base checkpoint")
    _add_training_args(resume, resume=True)
    resume.set_defaults(func=train_command, resume_from_checkpoint=True, config=None)

    _add_info_parser(sub, "info", "inspect a config or checkpoint")
    _add_info_parser(sub, "inspect", "compatibility alias for info")

    generate = sub.add_parser("generate", help="generate Standard MIDI from a trained checkpoint")
    generate.add_argument("--checkpoint", default="models/compound-base.pt")
    generate.add_argument("--out", default="generated.mid")
    generate.add_argument("--primer-midi")
    generate.add_argument("--events", type=int, default=512)
    generate.add_argument("--temperature", type=float, default=0.85)
    generate.add_argument("--top-p", type=float, default=0.92)
    generate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    generate.set_defaults(func=generate_command)
    return parser


def _validate(args: argparse.Namespace) -> None:
    for name in ("steps", "batch_size", "seq_len", "checkpoint_every", "log_every", "eval_every", "min_events", "events"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "learning_rate") and args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive")
    if hasattr(args, "weight_decay") and args.weight_decay < 0:
        raise SystemExit("--weight-decay must be non-negative")
    if hasattr(args, "temperature") and args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")
    if hasattr(args, "top_p") and not 0 < args.top_p <= 1:
        raise SystemExit("--top-p must be in (0, 1]")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    _validate(args)
    if getattr(args, "resume_from_checkpoint", False):
        args.resume = args.checkpoint
    args.func(args)


if __name__ == "__main__":
    main()
