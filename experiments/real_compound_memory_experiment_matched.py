from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import real_compound_memory_experiment as base  # noqa: E402


class SharedMatchedExact(base.SharedMatched):
    """Shared-memory control with exactly the routed model's parameter count.

    The base real-data harness is 194 parameters smaller than the routed
    multibank model (157,456 vs 157,650). This wrapper adds a 48->4 bias-free
    memory calibration probe (192 parameters) plus two learned memory gates.
    All 194 parameters are on the loss path and are classified as memory
    parameters by the staged optimizer policy.

    This module intentionally never mutates ``real_compound_memory_experiment``
    globals. Earlier versions patched ``base.SharedMatched``/``base.MODELS`` at
    import time, which made experiment results depend on pytest/import order.
    """

    def __init__(self) -> None:
        super().__init__()
        self.memory_capacity_probe = nn.Linear(base.D_MODEL, 4, bias=False)
        self.memory_capacity_gates = nn.Parameter(torch.tensor([0.1, 0.1]))

    def forward_chunk(self, records: torch.Tensor, state):  # type: ignore[no-untyped-def]
        fast, medium, slow, event, next_state = super().forward_chunk(records, state)
        hidden = self.embedding(records)
        probe = torch.tanh(self.memory_capacity_probe(hidden))
        gain = self.memory_capacity_gates[0]
        offset = self.memory_capacity_gates[1]

        def scale(logits: torch.Tensor, channel: int) -> torch.Tensor:
            temperature = 1.0 + gain * probe[:, :, channel : channel + 1]
            classes = torch.linspace(
                -1.0,
                1.0,
                logits.shape[-1],
                device=logits.device,
                dtype=logits.dtype,
            ).view(1, 1, -1)
            slope = offset * probe[:, :, channel : channel + 1] * classes
            return logits * temperature + slope

        fast = [scale(logits, 0) for logits in fast]
        medium = [scale(logits, 1) for logits in medium]
        slow = [scale(logits, 2) for logits in slow]
        event = scale(event, 3)
        return fast, medium, slow, event, next_state


SharedMatched = SharedMatchedExact
RoutedMultiBank = base.RoutedMultiBank
MODELS = {
    "shared_matched": SharedMatchedExact,
    "multibank_routed": RoutedMultiBank,
}

# Pure aliases are safe: none of these helpers resolve base.MODELS.
load_splits = base.load_splits
target_profile = base.target_profile
train_memory_stage = base.train_memory_stage
train_composer_stage = base.train_composer_stage
evaluate = base.evaluate
_configure_composer_optimizer = base._configure_composer_optimizer
D_MODEL = base.D_MODEL


def _parameter_counts() -> dict[str, int]:
    return {
        name: sum(parameter.numel() for parameter in model_type().parameters())
        for name, model_type in MODELS.items()
    }


def run(args) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """Run the real-Compound experiment without mutating the base module."""

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    train, validation = load_splits(
        args.train_jsonl,
        args.validation_jsonl,
        max_train_songs=args.max_train_songs,
        max_validation_songs=args.max_validation_songs,
    )
    if not any(len(song.records) > args.warmup_events for song in train):
        raise ValueError("no training song has events beyond --warmup-events")
    if not any(len(song.records) > args.warmup_events for song in validation):
        raise ValueError("no validation song has events beyond --warmup-events")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = MODELS[args.mode]().to(device)
    parameter_counts = _parameter_counts()
    if parameter_counts["shared_matched"] != parameter_counts["multibank_routed"]:
        raise AssertionError("shared/routed proxy models are not exactly parameter matched")

    train_memory_stage(
        model,
        train,
        epochs=args.memory_epochs,
        chunk_size=args.chunk_size,
        warmup_events=args.warmup_events,
        seed=args.seed,
        device=device,
        learning_rate=args.memory_lr,
    )
    before = evaluate(
        model,
        validation,
        chunk_size=args.chunk_size,
        warmup_events=args.warmup_events,
        device=device,
    )
    consolidated = copy.deepcopy(model)
    train_composer_stage(
        model,
        train,
        policy=args.composer_policy,
        epochs=args.composer_epochs,
        chunk_size=args.chunk_size,
        seed=args.seed,
        device=device,
        composer_lr=args.composer_lr,
        memory_lr_multiplier=args.memory_lr_multiplier,
    )
    after = evaluate(
        model,
        validation,
        chunk_size=args.chunk_size,
        warmup_events=args.warmup_events,
        device=device,
    )

    if args.checkpoint_out:
        checkpoint = Path(args.checkpoint_out)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "model": args.mode,
                "composer_policy": args.composer_policy,
                "consolidated_memory_state_dict": consolidated.state_dict(),
                "final_state_dict": model.state_dict(),
            },
            checkpoint,
        )

    return {
        "schema_version": 2,
        "device": str(device),
        "seed": args.seed,
        "model": args.mode,
        "composer_policy": args.composer_policy,
        "parameter_counts": parameter_counts,
        "split": {
            "train_jsonl_sha256": base._file_sha256(args.train_jsonl),
            "validation_jsonl_sha256": base._file_sha256(args.validation_jsonl),
            "train": base._corpus_summary(train),
            "validation": base._corpus_summary(validation),
            "split_leakage_check": "exact MIDI SHA-256 disjoint",
            "composition_family_near_dedup": "not yet implemented; required before production claims",
        },
        "target_profile": {
            "train": target_profile(train, args.warmup_events),
            "validation": target_profile(validation, args.warmup_events),
        },
        "training": {
            "memory_epochs": args.memory_epochs,
            "composer_epochs": args.composer_epochs,
            "chunk_size": args.chunk_size,
            "warmup_events": args.warmup_events,
            "memory_lr": args.memory_lr,
            "composer_lr": args.composer_lr,
            "memory_lr_multiplier": args.memory_lr_multiplier,
            "state_carry": "composition-local fixed-size state; detach at chunk boundaries",
        },
        "validation_before_composer": asdict(before),
        "validation_after_composer": asdict(after),
        "memory_delta": {
            "fast_macro_recall": after.fast_macro_recall - before.fast_macro_recall,
            "medium_macro_recall": after.medium_macro_recall - before.medium_macro_recall,
            "slow_macro_recall": after.slow_macro_recall - before.slow_macro_recall,
        },
        "scope": "real Compound JSONL experiment harness; corpus rights and composition-family dedup remain external gates",
        "reproducibility": "matched wrapper is import-side-effect-free; base module globals are not patched",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exactly parameter-matched state-carry memory experiment"
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--mode", choices=MODELS, required=True)
    parser.add_argument(
        "--composer-policy", choices=["frozen", "low_lr", "joint"], default="frozen"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--memory-epochs", type=int, default=1)
    parser.add_argument("--composer-epochs", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--warmup-events", type=int, default=32)
    parser.add_argument("--memory-lr", type=float, default=3e-3)
    parser.add_argument("--composer-lr", type=float, default=3e-3)
    parser.add_argument("--memory-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--max-train-songs", type=int, default=0)
    parser.add_argument("--max-validation-songs", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0 or args.warmup_events < 0:
        raise SystemExit("--chunk-size must be positive and --warmup-events non-negative")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    result = run(args)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
