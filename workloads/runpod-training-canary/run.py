from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from orbitune.compat import REFERENCE_PARAMETER_COUNT, TOKENIZER_ABI
from orbitune.model import OrbituneConfig, OrbituneGPT
from orbitune.tokenizer.vocab import TheoryRemiVocab


SCHEMA_VERSION = 1
WORKLOAD_ID = "orbitune-runpod-training-canary-v1"
DEFAULT_STEPS = 250
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQ_LEN = 256
DEFAULT_VALIDATION_INTERVAL = 50
DEFAULT_SEED = 20260824


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_ids(vocab: TheoryRemiVocab, *, repetitions: int = 4096) -> list[int]:
    # Deterministic Theory-REMI phrases. This is deliberately synthetic: the
    # canary validates GPU/container/training/checkpoint infrastructure, not
    # musical quality or corpus suitability.
    phrases = [
        ["BAR", "POSITION_0", "NOTE_PITCH_60", "NOTE_DURATION_4", "VELOCITY_16", "POSITION_4", "NOTE_PITCH_64", "NOTE_DURATION_4", "VELOCITY_18", "POSITION_8", "NOTE_PITCH_67", "NOTE_DURATION_4", "VELOCITY_20", "POSITION_12", "NOTE_PITCH_72", "NOTE_DURATION_4", "VELOCITY_18"],
        ["BAR", "POSITION_0", "NOTE_PITCH_48", "NOTE_DURATION_8", "VELOCITY_14", "POSITION_8", "NOTE_PITCH_55", "NOTE_DURATION_8", "VELOCITY_14", "POSITION_12", "NOTE_PITCH_60", "NOTE_DURATION_4", "VELOCITY_16"],
        ["BAR", "POSITION_0", "NOTE_PITCH_57", "NOTE_DURATION_4", "VELOCITY_15", "POSITION_4", "NOTE_PITCH_60", "NOTE_DURATION_4", "VELOCITY_17", "POSITION_8", "NOTE_PITCH_64", "NOTE_DURATION_8", "VELOCITY_19"],
        ["BAR", "POSITION_0", "NOTE_PITCH_50", "NOTE_DURATION_8", "VELOCITY_13", "POSITION_8", "NOTE_PITCH_57", "NOTE_DURATION_4", "VELOCITY_15", "POSITION_12", "NOTE_PITCH_62", "NOTE_DURATION_4", "VELOCITY_17"],
    ]
    ids: list[int] = [vocab.token_to_id["BOS"]]
    for index in range(repetitions):
        phrase = phrases[index % len(phrases)]
        ids.extend(vocab.token_to_id[token] for token in phrase)
    ids.append(vocab.token_to_id["EOS"])
    return ids


def _sample_batch(
    ids: list[int],
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(ids) <= seq_len + 1:
        raise ValueError("synthetic corpus is too small for requested sequence length")
    maximum = len(ids) - seq_len - 1
    starts = [rng.randint(0, maximum) for _ in range(batch_size)]
    x = torch.tensor([ids[s : s + seq_len] for s in starts], dtype=torch.long, device=device)
    y = torch.tensor([ids[s + 1 : s + seq_len + 1] for s in starts], dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def _validation_loss(model: OrbituneGPT, ids: list[int], *, seq_len: int, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    stride = seq_len
    # Fixed windows make validation deterministic and cheap.
    for start in range(0, min(len(ids) - seq_len - 1, seq_len * 8), stride):
        x = torch.tensor([ids[start : start + seq_len]], dtype=torch.long, device=device)
        y = torch.tensor([ids[start + 1 : start + seq_len + 1]], dtype=torch.long, device=device)
        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("validation produced no loss")
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("validation produced no windows")
    return sum(losses) / len(losses)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded Orbitune GPU training canary")
    parser.add_argument("--output-dir", default="/outputs")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.seq_len <= 1 or args.validation_interval <= 0:
        raise SystemExit("steps/batch-size/validation-interval must be positive and seq-len must exceed 1")

    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        raise SystemExit("CUDA is required for this run but torch.cuda.is_available() is false")
    if args.device == "cuda" and not cuda_available:
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if (args.device == "auto" and cuda_available) else args.device if args.device != "auto" else "cpu")

    torch.manual_seed(args.seed)
    if cuda_available:
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    vocab = TheoryRemiVocab()
    ids = _synthetic_ids(vocab)
    cfg = OrbituneConfig(vocab_size=len(vocab), dropout=0.0)
    model = OrbituneGPT(cfg).to(device)
    if model.parameter_count() != REFERENCE_PARAMETER_COUNT:
        raise RuntimeError(f"reference parameter count drifted: {model.parameter_count()} != {REFERENCE_PARAMETER_COUNT}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "canary-base.pt"
    result_path = output_dir / "result.json"

    validation_history: list[dict[str, float | int]] = []
    first_loss: float | None = None
    final_loss: float | None = None
    start_time = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, args.steps + 1):
        model.train()
        x, y = _sample_batch(ids, batch_size=args.batch_size, seq_len=args.seq_len, device=device, rng=rng)
        _, loss = model(x, y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        value = float(loss.detach().cpu())
        first_loss = value if first_loss is None else first_loss
        final_loss = value

        if step % args.validation_interval == 0 or step == args.steps:
            validation = _validation_loss(model, ids, seq_len=args.seq_len, device=device)
            if not math.isfinite(validation):
                raise RuntimeError(f"non-finite validation loss at step {step}")
            validation_history.append({"step": step, "loss": validation})

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time

    model.cpu().save_checkpoint(checkpoint)
    checkpoint_bytes = checkpoint.stat().st_size
    checkpoint_sha256 = _sha256(checkpoint)
    tokens_processed = args.steps * args.batch_size * args.seq_len
    final_validation = float(validation_history[-1]["loss"])
    initial_validation = float(validation_history[0]["loss"])
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    peak_vram_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0

    passed = (
        final_loss is not None
        and math.isfinite(final_loss)
        and math.isfinite(final_validation)
        and checkpoint_bytes > 0
        and final_validation < initial_validation
    )
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "workload_id": WORKLOAD_ID,
        "status": "pass" if passed else "fail",
        "purpose": "GPU/container/training/checkpoint infrastructure canary; not a musical-quality benchmark",
        "architecture": model.architecture,
        "tokenizer": TOKENIZER_ABI,
        "parameters": model.parameter_count(),
        "device_type": device.type,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "tokens_processed": tokens_processed,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_processed / elapsed if elapsed > 0 else 0.0,
        "first_training_loss": first_loss,
        "final_training_loss": final_loss,
        "validation_history": validation_history,
        "peak_vram_bytes": peak_vram_bytes,
        "checkpoint": {
            "name": checkpoint.name,
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_sha256,
        },
    }
    _atomic_json(payload, result_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
