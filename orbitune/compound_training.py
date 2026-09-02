from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from orbitune.compound import COMPOUND_TOKENIZER_ABI
from orbitune.tokenizer.compound_event import COMPOUND_RECORD_WIDTH


@dataclass(frozen=True, slots=True)
class CompoundSong:
    path: str
    sha256: str
    tokenizer_abi: str
    records: tuple[tuple[int, ...], ...]


def _validate_record(record: tuple[int, ...], *, source: str | Path, line_number: int) -> None:
    if len(record) != COMPOUND_RECORD_WIDTH:
        raise ValueError(
            f"{source}:{line_number}: each compound record must have width {COMPOUND_RECORD_WIDTH}"
        )
    if any(value < 0 for value in record):
        raise ValueError(f"{source}:{line_number}: compound record values must be non-negative")
    event_type, channel, delta_coarse, delta_residual = record[:4]
    duration_coarse, duration_residual, continuous_coarse, continuous_residual = record[8:12]
    if not 0 <= event_type <= 9:
        raise ValueError(f"{source}:{line_number}: invalid compound event type {event_type}")
    if not 0 <= channel <= 15:
        raise ValueError(f"{source}:{line_number}: invalid MIDI channel {channel}")
    if not 0 <= delta_coarse < 7 or not 0 <= duration_coarse < 7:
        raise ValueError(f"{source}:{line_number}: invalid time coarse index")
    if not 0 <= delta_residual < 16 or not 0 <= duration_residual < 16:
        raise ValueError(f"{source}:{line_number}: invalid time residual index")
    if not 0 <= continuous_coarse < 8 or not 0 <= continuous_residual < 8:
        raise ValueError(f"{source}:{line_number}: invalid continuous factor index")


def load_compound_jsonl(paths: str | Path | Iterable[str | Path]) -> list[CompoundSong]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    songs: list[CompoundSong] = []
    for source in paths:
        with Path(source).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                tokenizer_abi = str(payload.get("tokenizer_abi", ""))
                if tokenizer_abi != COMPOUND_TOKENIZER_ABI:
                    raise ValueError(
                        f"{source}:{line_number}: tokenizer ABI {tokenizer_abi!r} does not match "
                        f"{COMPOUND_TOKENIZER_ABI!r}"
                    )
                if int(payload.get("record_width", -1)) != COMPOUND_RECORD_WIDTH:
                    raise ValueError(
                        f"{source}:{line_number}: record_width does not match {COMPOUND_RECORD_WIDTH}"
                    )
                raw_records = payload.get("records")
                if not isinstance(raw_records, list) or not raw_records:
                    raise ValueError(f"{source}:{line_number}: missing compound records")
                records: list[tuple[int, ...]] = []
                for raw_record in raw_records:
                    if not isinstance(raw_record, list):
                        raise ValueError(f"{source}:{line_number}: compound record must be a list")
                    record = tuple(int(value) for value in raw_record)
                    _validate_record(record, source=source, line_number=line_number)
                    records.append(record)
                songs.append(
                    CompoundSong(
                        path=str(payload.get("path", "")),
                        sha256=str(payload.get("sha256", "")),
                        tokenizer_abi=tokenizer_abi,
                        records=tuple(records),
                    )
                )
    if not songs:
        raise ValueError("no compound songs loaded")
    return songs


def sample_compound_batch(
    songs: list[CompoundSong],
    *,
    batch_size: int,
    seq_len: int,
    rng: random.Random | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample song-local next-event windows without crossing composition boundaries."""

    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size and seq_len must be positive")
    eligible = [song for song in songs if len(song.records) >= seq_len + 1]
    if not eligible:
        raise ValueError("no song is long enough for the requested seq_len")
    rng = rng or random.Random()
    inputs: list[list[tuple[int, ...]]] = []
    targets: list[list[tuple[int, ...]]] = []
    for _ in range(batch_size):
        song = rng.choice(eligible)
        start = rng.randrange(0, len(song.records) - seq_len)
        window = song.records[start : start + seq_len + 1]
        inputs.append(list(window[:-1]))
        targets.append(list(window[1:]))
    return (
        torch.tensor(inputs, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


# ---------------------------------------------------------------------------
# Compound checkpoint / long-run helpers
#
# These helpers are shared by the Compound CFE trainer, the legacy CUDA
# trainer and the production launcher. They provide:
#   - atomic torch.save via temp-file + os.replace (same-filesystem only)
#   - schema-versioned payload assembly / parsing
#   - correct CUDA RNG state restore (the saved tensors must arrive as CPU
#     uint8 ByteTensors, otherwise torch.cuda.set_rng_state_all raises
#     "RNG state must be a torch.ByteTensor" on PyTorch 2.5+).
#   - validation-window determinism: a fixed seed precomputes the exact
#     windows so every eval run sees the same inputs.
#   - long-run health telemetry (loss history, grad norms, non-finite counts,
#     validation history, best/last-healthy step).
# ---------------------------------------------------------------------------


COMPOUND_CHECKPOINT_SCHEMA_VERSION = 2
COMPOUND_CHECKPOINT_SCHEMA_LEGACY = 1


def atomic_torch_save(payload: object, path: str | Path) -> None:
    """Write ``payload`` to ``path`` atomically within the same filesystem.

    A crash, power loss or process kill between torch.save() and
    os.replace() can leave a stale ``.tmp`` file behind but never a
    partially-written production checkpoint. ``os.replace`` is atomic on
    the same filesystem on both POSIX and Windows.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def normalize_cuda_rng_state(states: object) -> list[torch.Tensor]:
    """Return ``states`` as a list of CPU uint8 contiguous tensors.

    ``torch.cuda.get_rng_state_all()`` returns a list of CUDA uint8 tensors.
    After ``torch.save`` / ``torch.load(map_location="cpu")`` the tensors
    become CPU uint8 and ``torch.cuda.set_rng_state_all`` accepts them.

    However, when the caller loads a checkpoint with
    ``map_location="cuda"`` the tensors stay on CUDA. PyTorch 2.5+'s C++
    binding then rejects them because ``isinstance(state, torch.ByteTensor)``
    fails for CUDA uint8 tensors (only CPU uint8 satisfies the alias).
    Moving to CPU first is the documented fix.
    """
    if states is None:
        return []
    if not isinstance(states, list):
        raise ValueError("cuda_rng_state_all must be a list of per-device tensors")
    normalized: list[torch.Tensor] = []
    for index, state in enumerate(states):
        if not isinstance(state, torch.Tensor):
            raise ValueError(f"cuda_rng_state_all[{index}] is not a tensor: {type(state).__name__}")
        cpu_state = state.detach().to(device="cpu", dtype=torch.uint8)
        if not cpu_state.is_contiguous():
            cpu_state = cpu_state.contiguous()
        normalized.append(cpu_state)
    return normalized


def restore_cuda_rng_state(states: object) -> None:
    """Restore CUDA RNG state from a checkpoint payload.

    No-op when CUDA is unavailable. Raises on shape/dtype problems rather
    than silently dropping the restore (the previous workaround script
    silently swallowed this error, which violated reproducibility).
    """
    if not torch.cuda.is_available():
        return
    normalized = normalize_cuda_rng_state(states)
    if not normalized:
        return
    torch.cuda.set_rng_state_all(normalized)


def capture_validation_window_plan(
    songs: list[CompoundSong],
    *,
    validation_seed: int,
    batches: int,
    batch_size: int,
    seq_len: int,
) -> dict[str, object]:
    """Pre-sample the exact evaluation windows used by every validation call.

    ``TensorSampler.sample`` uses an internal Python RNG so two runs at the
    same step would otherwise pick different windows. This helper consumes
    ``batches`` windows from a private RNG seeded with ``validation_seed``
    and records each window's song identity + start offset. The trainer
    then iterates this plan deterministically.
    """
    eligible = [song for song in songs if len(song.records) >= seq_len + 1]
    if not eligible:
        raise ValueError("no song is long enough for the requested seq_len")

    private_rng = random.Random(validation_seed)
    plan: list[dict[str, object]] = []
    for batch_index in range(batches):
        windows: list[dict[str, object]] = []
        for sample_index in range(batch_size):
            song = private_rng.choice(eligible)
            start = private_rng.randrange(0, len(song.records) - seq_len)
            windows.append(
                {
                    "song_sha": getattr(song, "sha256", ""),
                    "song_records": int(len(song.records)),
                    "start": int(start),
                    "seq_len": int(seq_len),
                }
            )
        plan.append({"batch_index": batch_index, "windows": windows})

    payload = json.dumps(plan, sort_keys=True).encode("utf-8")
    window_hash = hashlib.sha256(payload).hexdigest()
    return {
        "validation_seed": int(validation_seed),
        "batches": int(batches),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "window_hash": window_hash,
        "plan": plan,
    }


def execute_validation_window_plan(
    songs: list[CompoundSong],
    plan_payload: dict[str, object],
    *,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Materialize validation windows from a captured plan, deterministically."""
    seq_len = int(plan_payload["seq_len"])
    by_sha: dict[str, CompoundSong] = {getattr(song, "sha256", ""): song for song in songs}
    by_records: dict[int, CompoundSong] = {}
    for song in songs:
        by_records.setdefault(len(song.records), song)

    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch in plan_payload["plan"]:
        tensors: list[torch.Tensor] = []
        for window in batch["windows"]:
            sha = window.get("song_sha", "")
            song = by_sha.get(sha) or by_records.get(int(window["song_records"]))
            if song is None:
                raise ValueError(
                    f"validation window song not found (sha={sha!r}, "
                    f"records={int(window['song_records'])})"
                )
            start = int(window["start"])
            tensor = torch.tensor(
                song.records[start : start + seq_len + 1], dtype=torch.long
            )
            tensors.append(tensor)
        joined = torch.stack(tensors).to(device)
        out.append((joined[:, :-1], joined[:, 1:]))
    return out


def build_compound_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    step: int,
    events_seen: int,
    runtime: dict[str, object],
    rng: random.Random,
    loss_history: list[float] | None = None,
    grad_norm_history: list[float] | None = None,
    non_finite_loss_count: int = 0,
    non_finite_grad_count: int = 0,
    spike_events: list[dict[str, object]] | None = None,
    best_validation_loss: float | None = None,
    best_step: int | None = None,
    last_healthy_step: int | None = None,
    last_healthy_events_seen: int | None = None,
    validation_history: list[dict[str, object]] | None = None,
    validation_plan: dict[str, object] | None = None,
    source_commit: str | None = None,
) -> dict[str, object]:
    """Assemble a versioned, device-independent Compound checkpoint payload."""
    scaler_state = None
    if scaler is not None and scaler.is_enabled():
        scaler_state = scaler.state_dict()
    cuda_states: object = None
    if torch.cuda.is_available():
        # Move to CPU up front so the saved payload is device-independent.
        cuda_states = [
            state.detach().to(device="cpu", dtype=torch.uint8)
            for state in torch.cuda.get_rng_state_all()
        ]
    return {
        "schema_version": COMPOUND_CHECKPOINT_SCHEMA_VERSION,
        "architecture": getattr(model, "architecture", None),
        "tokenizer": getattr(model, "tokenizer", None),
        "config": _config_to_dict(getattr(model, "config", None)),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "amp_scaler_state_dict": scaler_state,
        "step": int(step),
        "events_seen": int(events_seen),
        "runtime": dict(runtime),
        "torch_rng_state": torch.get_rng_state().cpu(),
        "cuda_rng_state_all": cuda_states,
        "python_rng_state": rng.getstate(),
        "sampler_rng_state": None,
        "source_commit": source_commit,
        "health": {
            "loss_history": list(loss_history or []),
            "grad_norm_history": list(grad_norm_history or []),
            "non_finite_loss_count": int(non_finite_loss_count),
            "non_finite_grad_count": int(non_finite_grad_count),
            "spike_events": list(spike_events or []),
            "best_validation_loss": best_validation_loss,
            "best_step": best_step,
            "last_healthy_step": last_healthy_step,
            "last_healthy_events_seen": last_healthy_events_seen,
        },
        "validation_history": list(validation_history or []),
        "validation_plan": validation_plan,
    }


def _config_to_dict(config: object) -> object:
    if config is None:
        return None
    if hasattr(config, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return config


def parse_compound_checkpoint(payload: dict[str, object]) -> dict[str, object]:
    """Validate and normalize a checkpoint payload.

    Accepts both legacy schema_version=1 (produced before long-run
    helpers existed) and the current schema_version=2. Returns a new dict
    with ``schema_version`` always set to the current value and missing
    fields filled with safe defaults.
    """
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    schema = int(payload.get("schema_version", 0))
    if schema not in (COMPOUND_CHECKPOINT_SCHEMA_LEGACY, COMPOUND_CHECKPOINT_SCHEMA_VERSION):
        raise ValueError(f"unsupported checkpoint schema_version={schema}")
    out = dict(payload)
    out["schema_version"] = COMPOUND_CHECKPOINT_SCHEMA_VERSION

    if "health" not in out or not isinstance(out["health"], dict):
        out["health"] = {}
    health = out["health"]
    health.setdefault("loss_history", [])
    health.setdefault("grad_norm_history", [])
    health.setdefault("non_finite_loss_count", 0)
    health.setdefault("non_finite_grad_count", 0)
    health.setdefault("spike_events", [])
    health.setdefault("best_validation_loss", None)
    health.setdefault("best_step", None)
    health.setdefault("last_healthy_step", None)
    health.setdefault("last_healthy_events_seen", None)

    out.setdefault("validation_history", [])
    out.setdefault("validation_plan", None)
    out.setdefault("amp_scaler_state_dict", None)
    out.setdefault("events_seen", int(out.get("step", 0)))
    return out


def assert_runtime_compatible(
    checkpoint_runtime: dict[str, object] | None,
    *,
    cli_runtime: dict[str, object],
    allow_runtime_change: bool,
) -> list[str]:
    """Compare checkpoint ``cuda_runtime`` to current CLI flags.

    Returns the list of drift fields. Empty list means compatible.
    Raises when drift is detected and ``allow_runtime_change`` is False.
    """
    if not checkpoint_runtime:
        return []
    drift: list[str] = []
    for key in ("n_head", "seq_len", "batch_size", "precision", "causal_fastpath"):
        if key not in checkpoint_runtime:
            continue
        old = checkpoint_runtime.get(key)
        new = cli_runtime.get(key)
        if old != new:
            drift.append(f"{key}: checkpoint={old!r} cli={new!r}")
    if not allow_runtime_change and drift:
        raise SystemExit(
            "runtime drift between checkpoint and CLI detected; "
            "pass --allow-runtime-change to override. drift=" + ", ".join(drift)
        )
    return drift
