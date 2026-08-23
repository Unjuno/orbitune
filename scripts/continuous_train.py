from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import torch

from orbitune.compat import REFERENCE_PARAMETER_COUNT, TOKENIZER_ABI
from orbitune.model import OrbituneConfig, OrbituneGPT
from orbitune.tokenizer.vocab import TheoryRemiVocab
from orbitune.training import _sample_batch, evaluate_token_loss, read_token_ids

CONTINUOUS_STATE_FORMAT = 1


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _capture_state(
    model: OrbituneGPT,
    optimizer: torch.optim.Optimizer,
    *,
    config: dict[str, object],
    global_step: int,
    tokens_seen: int,
    best_validation_loss: float,
    best_step: int,
    rng: random.Random,
    loss_history: list[float] | None = None,
    consecutive_spikes: int = 0,
) -> dict[str, object]:
    return {
        "state_format": CONTINUOUS_STATE_FORMAT,
        "architecture": model.architecture,
        "tokenizer": model.tokenizer,
        "config": config,
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "best_validation_loss": best_validation_loss,
        "best_step": best_step,
        "rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "loss_history": list(loss_history or []),
        "consecutive_spikes": consecutive_spikes,
    }


def _restore_state(
    payload: dict[str, object],
    model: OrbituneGPT,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    *,
    expected_config: dict[str, object],
) -> tuple[int, int, float, int]:
    if not isinstance(payload, dict):
        raise ValueError("continuous training state must be a mapping")
    state_format = int(payload.get("state_format", 0))
    if state_format not in (0, CONTINUOUS_STATE_FORMAT):
        raise ValueError(f"unsupported continuous training state format: {state_format}")
    if payload.get("architecture") != model.architecture or payload.get("config") != expected_config:
        raise ValueError("continuous training state architecture/config mismatch")
    if payload.get("tokenizer", TOKENIZER_ABI) != model.tokenizer:
        raise ValueError("continuous training state tokenizer mismatch")
    model_state = payload.get("model_state")
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
        raise ValueError("continuous training state is missing model/optimizer mappings")
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    if "rng_state" in payload:
        rng.setstate(payload["rng_state"])
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])
    return (
        int(payload.get("global_step", 0)),
        int(payload.get("tokens_seen", 0)),
        float(payload.get("best_validation_loss", float("inf"))),
        int(payload.get("best_step", 0)),
    )


def _restored_health_state(payload: dict[str, object], *, maxlen: int) -> tuple[deque[float], int]:
    raw_history = payload.get("loss_history", [])
    if not isinstance(raw_history, (list, tuple)):
        raise ValueError("continuous training loss_history must be a sequence")
    history: deque[float] = deque(maxlen=maxlen)
    for raw in raw_history[-maxlen:]:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("continuous training loss_history contains a non-finite value")
        history.append(value)
    consecutive = int(payload.get("consecutive_spikes", 0))
    if consecutive < 0:
        raise ValueError("continuous training consecutive_spikes must be non-negative")
    return history, consecutive


def _load_training_state(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("continuous training state must be a mapping")
    return payload


def _loss_zscore(history: deque[float], value: float, *, min_samples: int) -> float | None:
    if len(history) < min_samples:
        return None
    mean = statistics.fmean(history)
    std = statistics.pstdev(history)
    if std <= 1e-12:
        return 0.0 if value <= mean else float("inf")
    return (value - mean) / std


def _snapshot_boundary(previous_tokens: int, current_tokens: int, interval: int) -> list[int]:
    if interval <= 0 or current_tokens <= previous_tokens:
        return []
    first = ((previous_tokens // interval) + 1) * interval
    return list(range(first, current_tokens + 1, interval))


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue Orbitune reference Base training from a durable state")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--validation", nargs="+", required=True)
    parser.add_argument("--state", default=".orbitune-ci-state/state.pt")
    parser.add_argument("--healthy-state", default=".orbitune-ci-state/healthy.pt")
    parser.add_argument("--best", default=".orbitune-ci-state/best.pt")
    parser.add_argument("--snapshots-dir", default=".orbitune-ci-state/snapshots")
    parser.add_argument("--report", default=".orbitune-ci-state/report.json")
    parser.add_argument("--max-seconds", type=int, default=18_000)
    parser.add_argument("--max-steps", type=int, default=1_000_000_000)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--snapshot-token-interval", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--spike-window", type=int, default=100)
    parser.add_argument("--spike-min-samples", type=int, default=30)
    parser.add_argument("--spike-z-threshold", type=float, default=5.0)
    parser.add_argument("--spike-consecutive-limit", type=int, default=3)
    parser.add_argument("--gradient-spike-threshold", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_seconds <= 0 or args.max_steps <= 0 or args.validation_interval <= 0:
        raise SystemExit("max-seconds, max-steps and validation-interval must be positive")
    if args.snapshot_token_interval <= 0 or args.batch_size <= 0 or args.seq_len <= 0:
        raise SystemExit("snapshot-token-interval, batch-size and seq-len must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise SystemExit("learning-rate must be positive and weight-decay non-negative")
    if args.spike_window < 2 or not 2 <= args.spike_min_samples <= args.spike_window:
        raise SystemExit("spike window/min-samples are inconsistent")
    if args.spike_z_threshold <= 0 or args.spike_consecutive_limit <= 0 or args.gradient_spike_threshold <= 0:
        raise SystemExit("spike thresholds must be positive")

    vocab = TheoryRemiVocab()
    train_ids = read_token_ids(args.train, vocab)
    validation_ids = read_token_ids(args.validation, vocab)
    device = torch.device(args.device)
    state_path = Path(args.state)
    healthy_path = Path(args.healthy_state)
    best_path = Path(args.best)
    snapshots_dir = Path(args.snapshots_dir)
    report_path = Path(args.report)

    cfg = OrbituneConfig(vocab_size=len(vocab))
    config_dict = asdict(cfg)
    # Seed before parameter construction. A resume immediately overwrites RNG
    # and model/optimizer state from the checkpoint, while a fresh run now has
    # deterministic initial weights as promised by --seed.
    torch.manual_seed(args.seed)
    model = OrbituneGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    global_step = 0
    tokens_seen = 0
    best_validation_loss = float("inf")
    best_step = 0
    rolling_losses: deque[float] = deque(maxlen=args.spike_window)
    consecutive_spikes = 0

    if state_path.is_file():
        payload = _load_training_state(state_path)
        global_step, tokens_seen, best_validation_loss, best_step = _restore_state(
            payload, model, optimizer, rng, expected_config=config_dict
        )
        rolling_losses, consecutive_spikes = _restored_health_state(payload, maxlen=args.spike_window)

    if model.parameter_count() != REFERENCE_PARAMETER_COUNT:
        raise RuntimeError(f"reference parameter count drifted: {model.parameter_count()} != {REFERENCE_PARAMETER_COUNT}")

    parameters = list(model.parameters())
    start = time.monotonic()
    run_start_step = global_step
    run_start_tokens = tokens_seen
    losses: list[float] = []
    validation_history: list[dict[str, float | int]] = []
    gradient_norms: list[float] = []
    spike_events: list[dict[str, float | int | str]] = []
    rolled_back = False
    rollback_reason: str | None = None
    created_snapshots: list[str] = []
    last_healthy_step = global_step
    last_healthy_tokens = tokens_seen

    # Ensure there is always a durable rollback point before this run mutates weights.
    initial_state = _capture_state(
        model,
        optimizer,
        config=config_dict,
        global_step=global_step,
        tokens_seen=tokens_seen,
        best_validation_loss=best_validation_loss,
        best_step=best_step,
        rng=rng,
        loss_history=list(rolling_losses),
        consecutive_spikes=consecutive_spikes,
    )
    _atomic_torch_save(initial_state, healthy_path)

    while global_step < args.max_steps and time.monotonic() - start < args.max_seconds:
        model.train()
        x, y = _sample_batch(train_ids, batch_size=args.batch_size, seq_len=args.seq_len, device=device, rng=rng)
        _, loss = model(x, y)
        assert loss is not None
        loss_value = float(loss.detach().cpu())
        if not math.isfinite(loss_value):
            rolled_back = True
            rollback_reason = "non_finite_loss"
            break

        zscore = _loss_zscore(rolling_losses, loss_value, min_samples=args.spike_min_samples)
        is_loss_spike = zscore is not None and zscore > args.spike_z_threshold

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        grad_norm = float(grad_norm_tensor.detach().cpu())
        if not math.isfinite(grad_norm):
            rolled_back = True
            rollback_reason = "non_finite_gradient"
            break
        is_gradient_spike = grad_norm > args.gradient_spike_threshold

        if is_loss_spike:
            consecutive_spikes += 1
            spike_events.append({
                "step": global_step + 1,
                "kind": "loss",
                "loss": loss_value,
                "zscore": float(zscore),
                "gradient_norm": grad_norm,
            })
        else:
            consecutive_spikes = 0
        if is_gradient_spike:
            spike_events.append({
                "step": global_step + 1,
                "kind": "gradient",
                "loss": loss_value,
                "zscore": float(zscore) if zscore is not None else 0.0,
                "gradient_norm": grad_norm,
            })

        if consecutive_spikes >= args.spike_consecutive_limit or (is_loss_spike and is_gradient_spike):
            rolled_back = True
            rollback_reason = "persistent_training_spike"
            break

        optimizer.step()
        previous_tokens = tokens_seen
        global_step += 1
        tokens_seen += int(y.numel())
        losses.append(loss_value)
        gradient_norms.append(grad_norm)
        rolling_losses.append(loss_value)

        boundaries = _snapshot_boundary(previous_tokens, tokens_seen, args.snapshot_token_interval)
        for boundary in boundaries:
            snapshot_path = snapshots_dir / f"tokens-{boundary:012d}.pt"
            snapshot_state = _capture_state(
                model,
                optimizer,
                config=config_dict,
                global_step=global_step,
                tokens_seen=tokens_seen,
                best_validation_loss=best_validation_loss,
                best_step=best_step,
                rng=rng,
                loss_history=list(rolling_losses),
                consecutive_spikes=consecutive_spikes,
            )
            _atomic_torch_save(snapshot_state, snapshot_path)
            created_snapshots.append(str(snapshot_path))

        if global_step % args.validation_interval == 0:
            validation_loss = evaluate_token_loss(model, validation_ids, seq_len=args.seq_len, device=args.device)
            if not math.isfinite(validation_loss):
                rolled_back = True
                rollback_reason = "non_finite_validation_loss"
                break
            validation_history.append({"step": global_step, "tokens_seen": tokens_seen, "validation_loss": validation_loss})
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_step = global_step
                model.save_checkpoint(best_path)
            healthy_state = _capture_state(
                model,
                optimizer,
                config=config_dict,
                global_step=global_step,
                tokens_seen=tokens_seen,
                best_validation_loss=best_validation_loss,
                best_step=best_step,
                rng=rng,
                loss_history=list(rolling_losses),
                consecutive_spikes=consecutive_spikes,
            )
            _atomic_torch_save(healthy_state, healthy_path)
            last_healthy_step = global_step
            last_healthy_tokens = tokens_seen

    if rolled_back:
        healthy_payload = _load_training_state(healthy_path)
        global_step, tokens_seen, best_validation_loss, best_step = _restore_state(
            healthy_payload, model, optimizer, rng, expected_config=config_dict
        )
        rolling_losses, consecutive_spikes = _restored_health_state(healthy_payload, maxlen=args.spike_window)
    else:
        if not validation_history or validation_history[-1]["step"] != global_step:
            validation_loss = evaluate_token_loss(model, validation_ids, seq_len=args.seq_len, device=args.device)
            if not math.isfinite(validation_loss):
                raise RuntimeError("final validation loss is non-finite")
            validation_history.append({"step": global_step, "tokens_seen": tokens_seen, "validation_loss": validation_loss})
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_step = global_step
                model.save_checkpoint(best_path)
        healthy_state = _capture_state(
            model,
            optimizer,
            config=config_dict,
            global_step=global_step,
            tokens_seen=tokens_seen,
            best_validation_loss=best_validation_loss,
            best_step=best_step,
            rng=rng,
            loss_history=list(rolling_losses),
            consecutive_spikes=consecutive_spikes,
        )
        _atomic_torch_save(healthy_state, healthy_path)
        last_healthy_step = global_step
        last_healthy_tokens = tokens_seen

    elapsed = time.monotonic() - start
    state = _capture_state(
        model,
        optimizer,
        config=config_dict,
        global_step=global_step,
        tokens_seen=tokens_seen,
        best_validation_loss=best_validation_loss,
        best_step=best_step,
        rng=rng,
        loss_history=list(rolling_losses),
        consecutive_spikes=consecutive_spikes,
    )
    _atomic_torch_save(state, state_path)

    report = {
        "status": "rolled_back" if rolled_back else "healthy",
        "rollback_reason": rollback_reason,
        "state_format": CONTINUOUS_STATE_FORMAT,
        "tokenizer": model.tokenizer,
        "parameters": model.parameter_count(),
        "run_start_step": run_start_step,
        "global_step": global_step,
        "steps_this_run": global_step - run_start_step,
        "run_start_tokens": run_start_tokens,
        "tokens_seen": tokens_seen,
        "tokens_this_run": tokens_seen - run_start_tokens,
        "elapsed_seconds": elapsed,
        "last_loss": losses[-1] if losses else None,
        "loss_mean": statistics.fmean(losses) if losses else None,
        "gradient_norm_max": max(gradient_norms) if gradient_norms else None,
        "gradient_norm_mean": statistics.fmean(gradient_norms) if gradient_norms else None,
        "spike_count": len(spike_events),
        "spike_events": spike_events[-100:],
        "health_history_samples": len(rolling_losses),
        "consecutive_spikes": consecutive_spikes,
        "best_validation_loss": best_validation_loss,
        "best_step": best_step,
        "last_healthy_step": last_healthy_step,
        "last_healthy_tokens": last_healthy_tokens,
        "validation_history": validation_history,
        "created_snapshots": created_snapshots,
        "snapshot_token_interval": args.snapshot_token_interval,
        "train_tokens": len(train_ids),
        "validation_tokens": len(validation_ids),
        "config": config_dict,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
