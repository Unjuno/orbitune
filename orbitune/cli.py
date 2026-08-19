from __future__ import annotations

import argparse
import json
from pathlib import Path

from orbitune.adapter import package_adapter, validate_manifest_file
from orbitune.demo import make_demo_events
from orbitune.midi import read_midi, write_midi
from orbitune.tokenizer import TheoryRemiTokenizer


def _cmd_generate_demo(args: argparse.Namespace) -> None:
    events = make_demo_events(bars=args.bars, bpm=args.bpm)
    write_midi(events, args.out, bpm=args.bpm)
    print(f"wrote {args.out}")


def _cmd_tokenize(args: argparse.Namespace) -> None:
    tokenizer = TheoryRemiTokenizer()
    events = read_midi(args.midi)
    tokens = tokenizer.encode_events(events)
    tokenizer.write_tokens(tokens, args.out)
    print(f"wrote {len(tokens)} tokens to {args.out}")


def _cmd_detokenize(args: argparse.Namespace) -> None:
    tokenizer = TheoryRemiTokenizer()
    tokens = tokenizer.read_tokens(args.tokens)
    events = tokenizer.decode_events(tokens)
    write_midi(events, args.out, bpm=args.bpm)
    print(f"wrote {len(events)} events to {args.out}")


def _cmd_inspect(args: argparse.Namespace) -> None:
    path = Path(args.path)
    midi_files = sorted(path.rglob("*.mid")) if path.is_dir() else [path]
    tokenizer = TheoryRemiTokenizer()
    report = {"files": [], "total_files": 0, "total_events": 0, "total_tokens": 0}
    for midi_path in midi_files:
        events = read_midi(midi_path)
        tokens = tokenizer.encode_events(events)
        report["files"].append(
            {
                "path": str(midi_path),
                "events": len(events),
                "tokens": len(tokens),
            }
        )
        report["total_files"] += 1
        report["total_events"] += len(events)
        report["total_tokens"] += len(tokens)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


def _cmd_validate_adapter(args: argparse.Namespace) -> None:
    validate_manifest_file(args.manifest)
    print(f"valid adapter manifest: {args.manifest}")


def _cmd_package_adapter(args: argparse.Namespace) -> None:
    package_adapter(args.adapter_dir, args.manifest, args.out)
    print(f"wrote {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbitune")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_demo = subparsers.add_parser("generate-demo", help="write a deterministic demo MIDI file")
    generate_demo.add_argument("--out", required=True)
    generate_demo.add_argument("--bars", type=int, default=4)
    generate_demo.add_argument("--bpm", type=int, default=84)
    generate_demo.set_defaults(func=_cmd_generate_demo)

    tokenize = subparsers.add_parser("tokenize", help="convert MIDI to Theory-REMI tokens")
    tokenize.add_argument("midi")
    tokenize.add_argument("--out", required=True)
    tokenize.set_defaults(func=_cmd_tokenize)

    detokenize = subparsers.add_parser("detokenize", help="convert Theory-REMI tokens to MIDI")
    detokenize.add_argument("tokens")
    detokenize.add_argument("--out", required=True)
    detokenize.add_argument("--bpm", type=int, default=84)
    detokenize.set_defaults(func=_cmd_detokenize)

    inspect = subparsers.add_parser("inspect", help="inspect one MIDI file or a directory")
    inspect.add_argument("path")
    inspect.add_argument("--out", required=True)
    inspect.set_defaults(func=_cmd_inspect)

    validate_adapter = subparsers.add_parser("validate-adapter", help="validate an adapter manifest")
    validate_adapter.add_argument("manifest")
    validate_adapter.set_defaults(func=_cmd_validate_adapter)

    package_adapter_cmd = subparsers.add_parser("package-adapter", help="package an adapter directory")
    package_adapter_cmd.add_argument("adapter_dir")
    package_adapter_cmd.add_argument("--manifest", required=True)
    package_adapter_cmd.add_argument("--out", required=True)
    package_adapter_cmd.set_defaults(func=_cmd_package_adapter)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
