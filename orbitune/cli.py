from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from orbitune.adapter import create_adapter_scaffold, package_adapter, validate_manifest_file
from orbitune.compat import REFERENCE_MAX_SEQ_LEN, REFERENCE_N_EMBD, REFERENCE_N_HEAD, REFERENCE_N_LAYER
from orbitune.dataset import prepare_corpus, prepare_split_corpus
from orbitune.demo import make_demo_events
from orbitune.evaluation import write_evaluation
from orbitune.exporting import export_onnx, export_web_onnx
from orbitune.inference import generate_midi
from orbitune.lora import LoRAConfig
from orbitune.midi import read_midi, write_midi
from orbitune.model import OrbituneConfig, OrbituneGPT
from orbitune.tokenizer import TheoryRemiTokenizer
from orbitune.tokenizer.vocab import TheoryRemiVocab
from orbitune.training import TrainConfig, train_adapter, train_base


def _cmd_generate_demo(args: argparse.Namespace) -> None:
    events = make_demo_events(bars=args.bars, bpm=args.bpm); write_midi(events, args.out, bpm=args.bpm); print(f"wrote {args.out}")

def _cmd_tokenize(args: argparse.Namespace) -> None:
    tokenizer = TheoryRemiTokenizer(); events = read_midi(args.midi); tokens = tokenizer.encode_events(events); tokenizer.write_tokens(tokens, args.out); print(f"wrote {len(tokens)} tokens to {args.out}")

def _cmd_prepare_corpus(args: argparse.Namespace) -> None:
    print(json.dumps(prepare_corpus(args.source, args.out, args.report, min_events=args.min_events), indent=2))

def _cmd_prepare_split_corpus(args: argparse.Namespace) -> None:
    report = prepare_split_corpus(args.source, args.train_out, args.validation_out, args.report, validation_fraction=args.validation_fraction, split_seed=args.split_seed, min_events=args.min_events); print(json.dumps(report, indent=2))

def _cmd_detokenize(args: argparse.Namespace) -> None:
    tokenizer = TheoryRemiTokenizer(); tokens = tokenizer.read_tokens(args.tokens); events = tokenizer.decode_events(tokens); write_midi(events, args.out, bpm=args.bpm); print(f"wrote {len(events)} events to {args.out}")

def _cmd_inspect(args: argparse.Namespace) -> None:
    path = Path(args.path); midi_files = sorted(path.rglob("*.mid")) if path.is_dir() else [path]; tokenizer = TheoryRemiTokenizer(); report = {"files": [], "total_files": 0, "total_events": 0, "total_tokens": 0}
    for midi_path in midi_files:
        events = read_midi(midi_path); tokens = tokenizer.encode_events(events); report["files"].append({"path": str(midi_path), "events": len(events), "tokens": len(tokens)}); report["total_files"] += 1; report["total_events"] += len(events); report["total_tokens"] += len(tokens)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); print(f"wrote {args.out}")

def _cmd_eval_midi(args: argparse.Namespace) -> None:
    print(json.dumps(write_evaluation(args.midi, args.out), indent=2))

def _cmd_init_adapter(args: argparse.Namespace) -> None:
    root = create_adapter_scaffold(args.directory, name=args.name, display_name=args.display_name, adapter_family=args.family, rank=4, bpm=args.bpm, bars=args.bars, temperature=args.temperature); print(f"created adapter scaffold at {root}")

def _train_cfg(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len, learning_rate=args.learning_rate, weight_decay=args.weight_decay, device=args.device, seed=args.seed, validation_interval=args.validation_interval)

def _cmd_train_base(args: argparse.Namespace) -> None:
    vocab = TheoryRemiVocab(); cfg = OrbituneConfig(vocab_size=len(vocab), max_seq_len=args.max_seq_len, n_layer=REFERENCE_N_LAYER, n_embd=REFERENCE_N_EMBD, n_head=REFERENCE_N_HEAD, dropout=args.dropout)
    print(json.dumps(train_base(args.tokens, args.out, model_cfg=cfg, train_cfg=_train_cfg(args), validation_token_paths=args.validation_tokens), indent=2))

def _cmd_train_adapter(args: argparse.Namespace) -> None:
    lora_cfg = LoRAConfig(rank=4, alpha=args.alpha, dropout=args.lora_dropout); print(json.dumps(train_adapter(args.base, args.tokens, args.out, lora_cfg=lora_cfg, train_cfg=_train_cfg(args), validation_token_paths=args.validation_tokens), indent=2))

def _cmd_generate(args: argparse.Namespace) -> None:
    events = generate_midi(args.base, args.out, adapter=args.adapter, bpm=args.bpm, bars=args.bars, temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens, device=args.device); print(f"wrote {args.out} with {events} note events")

def _cmd_export_onnx(args: argparse.Namespace) -> None:
    print(f"wrote {export_onnx(args.base, args.out, example_seq_len=args.example_seq_len)}")

def _cmd_export_web_onnx(args: argparse.Namespace) -> None:
    print(f"wrote {export_web_onnx(args.base, args.out, example_seq_len=args.example_seq_len)}")

def _cmd_model_info(args: argparse.Namespace) -> None:
    model = OrbituneGPT.load_checkpoint(args.checkpoint) if args.checkpoint else OrbituneGPT(OrbituneConfig(vocab_size=len(TheoryRemiVocab()))); print(json.dumps({"architecture": model.architecture, "parameters": model.parameter_count(), "config": asdict(model.config)}, indent=2))

def _cmd_validate_adapter(args: argparse.Namespace) -> None:
    validate_manifest_file(args.manifest); print(f"valid adapter manifest: {args.manifest}")

def _cmd_package_adapter(args: argparse.Namespace) -> None:
    package_adapter(args.adapter_dir, args.manifest, args.out); print(f"wrote {args.out}")

def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tokens", nargs="+", required=True); parser.add_argument("--validation-tokens", nargs="+"); parser.add_argument("--validation-interval", type=int, default=0); parser.add_argument("--steps", type=int, default=100); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--seq-len", type=int, default=256); parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=0.01); parser.add_argument("--device", default="cpu"); parser.add_argument("--seed", type=int, default=1234)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbitune"); sub = parser.add_subparsers(dest="command", required=True)
    p=sub.add_parser("generate-demo"); p.add_argument("--out",required=True); p.add_argument("--bars",type=int,default=4); p.add_argument("--bpm",type=int,default=84); p.set_defaults(func=_cmd_generate_demo)
    p=sub.add_parser("tokenize"); p.add_argument("midi"); p.add_argument("--out",required=True); p.set_defaults(func=_cmd_tokenize)
    p=sub.add_parser("prepare-corpus"); p.add_argument("source"); p.add_argument("--out",required=True); p.add_argument("--report",required=True); p.add_argument("--min-events",type=int,default=1); p.set_defaults(func=_cmd_prepare_corpus)
    p=sub.add_parser("prepare-split-corpus"); p.add_argument("source"); p.add_argument("--train-out",required=True); p.add_argument("--validation-out",required=True); p.add_argument("--report",required=True); p.add_argument("--validation-fraction",type=float,default=0.1); p.add_argument("--split-seed",default="orbitune-v0"); p.add_argument("--min-events",type=int,default=1); p.set_defaults(func=_cmd_prepare_split_corpus)
    p=sub.add_parser("detokenize"); p.add_argument("tokens"); p.add_argument("--out",required=True); p.add_argument("--bpm",type=int,default=84); p.set_defaults(func=_cmd_detokenize)
    p=sub.add_parser("inspect"); p.add_argument("path"); p.add_argument("--out",required=True); p.set_defaults(func=_cmd_inspect)
    p=sub.add_parser("eval-midi"); p.add_argument("midi"); p.add_argument("--out",required=True); p.set_defaults(func=_cmd_eval_midi)
    p=sub.add_parser("init-adapter"); p.add_argument("directory"); p.add_argument("--name",required=True); p.add_argument("--display-name",required=True); p.add_argument("--family",default="style"); p.add_argument("--bpm",type=int,default=84); p.add_argument("--bars",type=int,default=8,choices=[4,8,16]); p.add_argument("--temperature",type=float,default=0.85); p.set_defaults(func=_cmd_init_adapter)
    p=sub.add_parser("train-base",help="train the current ~10M reference Base"); _add_train_args(p); p.add_argument("--out",required=True); p.add_argument("--max-seq-len",type=int,default=REFERENCE_MAX_SEQ_LEN); p.add_argument("--dropout",type=float,default=0.1); p.set_defaults(func=_cmd_train_base)
    p=sub.add_parser("train-adapter"); _add_train_args(p); p.add_argument("--base",required=True); p.add_argument("--out",required=True); p.add_argument("--alpha",type=float,default=8.0); p.add_argument("--lora-dropout",type=float,default=0.0); p.set_defaults(func=_cmd_train_adapter)
    p=sub.add_parser("generate"); p.add_argument("--base",required=True); p.add_argument("--adapter"); p.add_argument("--out",required=True); p.add_argument("--bpm",type=int,default=84); p.add_argument("--bars",type=int,default=8,choices=[4,8,16]); p.add_argument("--temperature",type=float,default=0.85); p.add_argument("--top-p",type=float,default=0.92); p.add_argument("--max-new-tokens",type=int,default=2048); p.add_argument("--device",default="cpu"); p.set_defaults(func=_cmd_generate)
    p=sub.add_parser("export-onnx"); p.add_argument("--base",required=True); p.add_argument("--out",required=True); p.add_argument("--example-seq-len",type=int,default=64); p.set_defaults(func=_cmd_export_onnx)
    p=sub.add_parser("export-web-onnx"); p.add_argument("--base",required=True); p.add_argument("--out",required=True); p.add_argument("--example-seq-len",type=int,default=64); p.set_defaults(func=_cmd_export_web_onnx)
    p=sub.add_parser("model-info"); p.add_argument("--checkpoint"); p.set_defaults(func=_cmd_model_info)
    p=sub.add_parser("validate-adapter"); p.add_argument("manifest"); p.set_defaults(func=_cmd_validate_adapter)
    p=sub.add_parser("package-adapter"); p.add_argument("adapter_dir"); p.add_argument("--manifest",required=True); p.add_argument("--out",required=True); p.set_defaults(func=_cmd_package_adapter)
    return parser

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args,"validation_interval",0) < 0: raise SystemExit("--validation-interval must be >= 0")
    if getattr(args,"validation_interval",0) > 0 and not getattr(args,"validation_tokens",None): raise SystemExit("--validation-interval requires --validation-tokens")
    args.func(args)

if __name__ == "__main__": main()
