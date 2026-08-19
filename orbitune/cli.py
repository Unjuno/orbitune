from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from orbitune.adapter import package_adapter, validate_manifest_file
from orbitune.dataset import prepare_corpus
from orbitune.demo import make_demo_events
from orbitune.inference import generate_midi
from orbitune.lora import LoRAConfig
from orbitune.midi import read_midi, write_midi
from orbitune.model import OrbituneConfig, OrbituneGPT
from orbitune.tokenizer import TheoryRemiTokenizer
from orbitune.tokenizer.vocab import TheoryRemiVocab
from orbitune.training import TrainConfig, train_adapter, train_base


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


def _cmd_prepare_corpus(args: argparse.Namespace) -> None:
    report = prepare_corpus(args.source, args.out, args.report, min_events=args.min_events)
    print(json.dumps(report, indent=2))


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
        report["files"].append({"path": str(midi_path), "events": len(events), "tokens": len(tokens)})
        report["total_files"] += 1
        report["total_events"] += len(events)
        report["total_tokens"] += len(tokens)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


def _train_cfg(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
    )


def _cmd_train_base(args: argparse.Namespace) -> None:
    vocab = TheoryRemiVocab()
    cfg = OrbituneConfig(
        vocab_size=len(vocab),
        max_seq_len=args.max_seq_len,
        n_layer=4,
        n_embd=256,
        n_head=4,
        dropout=args.dropout,
    )
    report = train_base(args.tokens, args.out, model_cfg=cfg, train_cfg=_train_cfg(args))
    print(json.dumps(report, indent=2))


def _cmd_train_adapter(args: argparse.Namespace) -> None:
    lora_cfg = LoRAConfig(rank=args.rank, alpha=args.alpha, dropout=args.lora_dropout)
    report = train_adapter(args.base, args.tokens, args.out, lora_cfg=lora_cfg, train_cfg=_train_cfg(args))
    print(json.dumps(report, indent=2))


def _cmd_generate(args: argparse.Namespace) -> None:
    events = generate_midi(
        args.base,
        args.out,
        adapter=args.adapter,
        bpm=args.bpm,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    print(f"wrote {args.out} with {events} note events")


def _cmd_model_info(args: argparse.Namespace) -> None:
    if args.checkpoint:
        model = OrbituneGPT.load_checkpoint(args.checkpoint)
    else:
        vocab = TheoryRemiVocab()
        model = OrbituneGPT(OrbituneConfig(vocab_size=len(vocab)))
    print(json.dumps({"architecture": model.architecture, "parameters": model.parameter_count(), "config": asdict(model.config)}, indent=2))


def _cmd_validate_adapter(args: argparse.Namespace) -> None:
    validate_manifest_file(args.manifest)
    print(f"valid adapter manifest: {args.manifest}")


def _cmd_package_adapter(args: argparse.Namespace) -> None:
    package_adapter(args.adapter_dir, args.manifest, args.out)
    print(f"wrote {args.out}")


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tokens", nargs="+", required=True, help="one or more Theory-REMI token text files")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234, help="training reproducibility only; not exposed in the web UI")


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

    corpus = subparsers.add_parser("prepare-corpus", help="convert a MIDI directory into one training token corpus")
    corpus.add_argument("source")
    corpus.add_argument("--out", required=True, help="output Theory-REMI token text file")
    corpus.add_argument("--report", required=True, help="JSON data-quality report")
    corpus.add_argument("--min-events", type=int, default=1)
    corpus.set_defaults(func=_cmd_prepare_corpus)

    detokenize = subparsers.add_parser("detokenize", help="convert Theory-REMI tokens to MIDI")
    detokenize.add_argument("tokens")
    detokenize.add_argument("--out", required=True)
    detokenize.add_argument("--bpm", type=int, default=84)
    detokenize.set_defaults(func=_cmd_detokenize)

    inspect = subparsers.add_parser("inspect", help="inspect one MIDI file or a directory")
    inspect.add_argument("path")
    inspect.add_argument("--out", required=True)
    inspect.set_defaults(func=_cmd_inspect)

    train_base_cmd = subparsers.add_parser("train-base", help="train orbitune-tiny-v0 from Theory-REMI token files")
    _add_train_args(train_base_cmd)
    train_base_cmd.add_argument("--out", required=True)
    train_base_cmd.add_argument("--max-seq-len", type=int, default=512)
    train_base_cmd.add_argument("--dropout", type=float, default=0.1)
    train_base_cmd.set_defaults(func=_cmd_train_base)

    train_adapter_cmd = subparsers.add_parser("train-adapter", help="train a q_proj/v_proj LoRA adapter")
    _add_train_args(train_adapter_cmd)
    train_adapter_cmd.add_argument("--base", required=True)
    train_adapter_cmd.add_argument("--out", required=True)
    train_adapter_cmd.add_argument("--rank", type=int, default=4)
    train_adapter_cmd.add_argument("--alpha", type=float, default=8.0)
    train_adapter_cmd.add_argument("--lora-dropout", type=float, default=0.0)
    train_adapter_cmd.set_defaults(func=_cmd_train_adapter)

    generate = subparsers.add_parser("generate", help="generate MIDI from a trained base and optional adapter")
    generate.add_argument("--base", required=True)
    generate.add_argument("--adapter")
    generate.add_argument("--out", required=True)
    generate.add_argument("--bpm", type=int, default=84)
    generate.add_argument("--temperature", type=float, default=0.85)
    generate.add_argument("--top-p", type=float, default=0.92)
    generate.add_argument("--max-new-tokens", type=int, default=256)
    generate.add_argument("--device", default="cpu")
    generate.set_defaults(func=_cmd_generate)

    model_info = subparsers.add_parser("model-info", help="print model parameter count and configuration")
    model_info.add_argument("--checkpoint")
    model_info.set_defaults(func=_cmd_model_info)

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
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
