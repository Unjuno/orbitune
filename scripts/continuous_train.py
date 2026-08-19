from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from orbitune.compat import REFERENCE_PARAMETER_COUNT
from orbitune.model import OrbituneConfig, OrbituneGPT
from orbitune.tokenizer.vocab import TheoryRemiVocab
from orbitune.training import _sample_batch, evaluate_token_loss, read_token_ids


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue Orbitune reference Base training from a durable state")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--validation", nargs="+", required=True)
    parser.add_argument("--state", default=".orbitune-ci-state/state.pt")
    parser.add_argument("--best", default=".orbitune-ci-state/best.pt")
    parser.add_argument("--report", default=".orbitune-ci-state/report.json")
    parser.add_argument("--max-seconds", type=int, default=18_000)
    parser.add_argument("--max-steps", type=int, default=1_000_000_000)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_seconds <= 0 or args.validation_interval <= 0:
        raise SystemExit("max-seconds and validation-interval must be positive")

    vocab = TheoryRemiVocab()
    train_ids = read_token_ids(args.train, vocab)
    validation_ids = read_token_ids(args.validation, vocab)
    device = torch.device(args.device)
    state_path = Path(args.state)
    best_path = Path(args.best)
    report_path = Path(args.report)

    cfg = OrbituneConfig(vocab_size=len(vocab))
    model = OrbituneGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    global_step = 0
    best_validation_loss = float("inf")
    best_step = 0

    if state_path.is_file():
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        if payload.get("architecture") != model.architecture or payload.get("config") != vars(cfg):
            raise ValueError("continuous training state architecture/config mismatch")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        global_step = int(payload.get("global_step", 0))
        best_validation_loss = float(payload.get("best_validation_loss", float("inf")))
        best_step = int(payload.get("best_step", 0))
        if "rng_state" in payload:
            rng.setstate(payload["rng_state"])
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"])
    else:
        torch.manual_seed(args.seed)

    if model.parameter_count() != REFERENCE_PARAMETER_COUNT:
        raise RuntimeError(f"reference parameter count drifted: {model.parameter_count()} != {REFERENCE_PARAMETER_COUNT}")

    parameters = list(model.parameters())
    start = time.monotonic()
    run_start_step = global_step
    losses: list[float] = []
    validation_history: list[dict[str, float | int]] = []

    while global_step < args.max_steps and time.monotonic() - start < args.max_seconds:
        model.train()
        x, y = _sample_batch(train_ids, batch_size=args.batch_size, seq_len=args.seq_len, device=device, rng=rng)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        global_step += 1
        losses.append(float(loss.detach().cpu()))

        if global_step % args.validation_interval == 0:
            validation_loss = evaluate_token_loss(model, validation_ids, seq_len=args.seq_len, device=args.device)
            validation_history.append({"step": global_step, "validation_loss": validation_loss})
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_step = global_step
                model.save_checkpoint(best_path)

    if not validation_history or validation_history[-1]["step"] != global_step:
        validation_loss = evaluate_token_loss(model, validation_ids, seq_len=args.seq_len, device=args.device)
        validation_history.append({"step": global_step, "validation_loss": validation_loss})
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_step = global_step
            model.save_checkpoint(best_path)

    elapsed = time.monotonic() - start
    state = {
        "architecture": model.architecture,
        "config": vars(cfg),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "best_step": best_step,
        "rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    _atomic_torch_save(state, state_path)
    report = {
        "parameters": model.parameter_count(),
        "run_start_step": run_start_step,
        "global_step": global_step,
        "steps_this_run": global_step - run_start_step,
        "elapsed_seconds": elapsed,
        "last_loss": losses[-1] if losses else None,
        "best_validation_loss": best_validation_loss,
        "best_step": best_step,
        "validation_history": validation_history,
        "train_tokens": len(train_ids),
        "validation_tokens": len(validation_ids),
        "config": vars(cfg),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
