from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from compound_memory_target_routing_proxy import (  # noqa: E402
    ACTIVE_START,
    FAST_CARDS,
    LATE_START,
    MEDIUM_CARDS,
    SEQ_LEN,
    SLOW_CARDS,
    RoutedMultiBank,
    _balanced_loss,
    _macro_recall,
    make_batch,
)


@dataclass(slots=True)
class MemoryMetrics:
    fast_macro_recall: float
    medium_macro_recall: float
    slow_macro_recall: float
    next_event_type_accuracy: float


@dataclass(slots=True)
class Result:
    policy: str
    seed: int
    stage1_steps: int
    stage2_steps: int
    before: MemoryMetrics
    after: MemoryMetrics


def evaluate(model: RoutedMultiBank, device: torch.device, seed: int) -> MemoryMetrics:
    records, fast, medium, slow = make_batch(48, device, 99991 + seed)
    late = torch.arange(SEQ_LEN, device=device) >= LATE_START
    model.eval()
    with torch.no_grad():
        fast_logits, medium_logits, slow_logits, event_logits = model(records)
        event_accuracy = float(
            (event_logits[:, :-1].argmax(-1) == records[:, 1:, 0]).float().mean()
        )
    return MemoryMetrics(
        fast_macro_recall=_macro_recall(fast_logits, fast, late),
        medium_macro_recall=_macro_recall(medium_logits, medium, late),
        slow_macro_recall=_macro_recall(slow_logits, slow, late),
        next_event_type_accuracy=event_accuracy,
    )


def pretrain_memory(
    seed: int,
    steps: int,
    batch: int,
    device: torch.device,
) -> RoutedMultiBank:
    torch.manual_seed(seed)
    random.seed(seed)
    model = RoutedMultiBank().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    active = torch.arange(SEQ_LEN, device=device) >= ACTIVE_START
    model.train()
    for step in range(steps):
        records, fast, medium, slow = make_batch(
            batch, device, seed * 10000 + step
        )
        fast_logits, medium_logits, slow_logits, _ = model(records)
        loss = (
            _balanced_loss(fast_logits, fast, FAST_CARDS, active)
            + _balanced_loss(medium_logits, medium, MEDIUM_CARDS, active)
            + _balanced_loss(slow_logits, slow, SLOW_CARDS, active)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model


def _composer_optimizer(
    model: RoutedMultiBank,
    policy: str,
) -> torch.optim.Optimizer:
    if policy == "frozen":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith("event_head") or name.startswith("event_mix")
            )
        return torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=3e-3,
        )

    if policy == "joint":
        return torch.optim.AdamW(model.parameters(), lr=3e-3)

    if policy == "low_lr":
        composer: list[torch.nn.Parameter] = []
        memory: list[torch.nn.Parameter] = []
        for name, parameter in model.named_parameters():
            if name.startswith("event_head") or name.startswith("event_mix"):
                composer.append(parameter)
            elif name.startswith("embedding") or "_memory" in name:
                memory.append(parameter)
            else:
                parameter.requires_grad_(False)
        return torch.optim.AdamW(
            [
                {"params": composer, "lr": 3e-3},
                {"params": memory, "lr": 3e-4},
            ]
        )

    raise ValueError(f"unsupported policy: {policy}")


def train_composer(
    base_model: RoutedMultiBank,
    seed: int,
    policy: str,
    steps: int,
    batch: int,
    device: torch.device,
) -> RoutedMultiBank:
    model = copy.deepcopy(base_model).to(device)
    optimizer = _composer_optimizer(model, policy)
    model.train()
    for step in range(steps):
        records, _, _, _ = make_batch(batch, device, seed * 30000 + step)
        _, _, _, event_logits = model(records)
        loss = F.cross_entropy(
            event_logits[:, :-1].reshape(-1, 10),
            records[:, 1:, 0].reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainable = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    return model


def run(
    policy: str,
    seed: int,
    stage1_steps: int,
    stage2_steps: int,
    batch: int,
    device: torch.device,
) -> Result:
    base = pretrain_memory(seed, stage1_steps, batch, device)
    before = evaluate(base, device, seed)
    composer = train_composer(
        base, seed, policy, stage2_steps, batch, device
    )
    after = evaluate(composer, device, seed)
    return Result(
        policy=policy,
        seed=seed,
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        before=before,
        after=after,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["frozen", "low_lr", "joint"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage1-steps", type=int, default=60)
    parser.add_argument("--stage2-steps", type=int, default=80)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    torch.set_num_threads(min(4, torch.get_num_threads()))
    result = run(
        args.policy,
        args.seed,
        args.stage1_steps,
        args.stage2_steps,
        args.batch,
        device,
    )
    payload = {
        "schema_version": 1,
        "device": str(device),
        "task": {
            "stage1": "memory-target consolidation only",
            "stage2": "next-event-type composer objective only",
            "scope": "gradient-interference proxy; final local Transformer is not modeled here",
        },
        "result": {
            **asdict(result),
            "before": asdict(result.before),
            "after": asdict(result.after),
        },
    }
    Path(args.out).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
