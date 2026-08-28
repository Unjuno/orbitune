from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True, slots=True)
class ScanResult:
    states: torch.Tensor
    normalizers: torch.Tensor
    final_state: torch.Tensor
    final_normalizer: torch.Tensor


def _powers(decay: torch.Tensor, length: int, dtype: torch.dtype, device: torch.device):
    index = torch.arange(length, dtype=dtype, device=device)
    if decay.ndim == 0:
        inverse = decay.pow(-index)
        forward = decay.pow(index)
        initial = decay.pow(index + 1)
    elif decay.ndim == 1:
        inverse = decay[None, :].pow(-index[:, None])
        forward = decay[None, :].pow(index[:, None])
        initial = decay[None, :].pow(index[:, None] + 1)
    else:
        raise ValueError("decay must be scalar or [slots]")
    return inverse, forward, initial


def chunkwise_discounted_scan(
    contributions: torch.Tensor,
    normalizer_contributions: torch.Tensor,
    decay: torch.Tensor,
    *,
    chunk_size: int = 128,
    initial_state: torch.Tensor | None = None,
    initial_normalizer: torch.Tensor | None = None,
) -> ScanResult:
    """Exact discounted recurrence with bounded exponent range inside each chunk.

    The represented recurrence is::

        S_t = decay * S_(t-1) + C_t
        Z_t = decay * Z_(t-1) + z_t

    ``contributions`` has shape ``[batch,time,slots,width]`` and normalizer
    contributions ``[batch,time,slots]``. The chunk loop is over coarse
    segments, never individual events; state crossing a chunk boundary is the
    same fixed-size state used by streaming inference.
    """

    if contributions.ndim != 4:
        raise ValueError("contributions must have shape [batch,time,slots,width]")
    if normalizer_contributions.shape != contributions.shape[:3]:
        raise ValueError("normalizer_contributions shape mismatch")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    batch, total, slots, width = contributions.shape
    if decay.ndim == 1 and decay.shape != (slots,):
        raise ValueError("vector decay must have one value per slot")
    if torch.any((decay <= 0) | (decay > 1)):
        raise ValueError("decay must be in (0, 1]")

    state = (
        contributions.new_zeros((batch, slots, width))
        if initial_state is None
        else initial_state
    )
    normalizer = (
        normalizer_contributions.new_zeros((batch, slots))
        if initial_normalizer is None
        else initial_normalizer
    )
    if state.shape != (batch, slots, width):
        raise ValueError("initial_state shape mismatch")
    if normalizer.shape != (batch, slots):
        raise ValueError("initial_normalizer shape mismatch")

    state_chunks: list[torch.Tensor] = []
    normalizer_chunks: list[torch.Tensor] = []
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        chunk = contributions[:, start:stop]
        z_chunk = normalizer_contributions[:, start:stop]
        length = stop - start
        inverse, forward, initial = _powers(
            decay, length, contributions.dtype, contributions.device
        )
        if decay.ndim == 0:
            weighted = torch.cumsum(
                chunk * inverse[None, :, None, None], dim=1
            ) * forward[None, :, None, None]
            weighted_z = torch.cumsum(
                z_chunk * inverse[None, :, None], dim=1
            ) * forward[None, :, None]
            states = weighted + state[:, None] * initial[None, :, None, None]
            normalizers = weighted_z + normalizer[:, None] * initial[None, :, None]
        else:
            weighted = torch.cumsum(
                chunk * inverse[None, :, :, None], dim=1
            ) * forward[None, :, :, None]
            weighted_z = torch.cumsum(
                z_chunk * inverse[None, :, :], dim=1
            ) * forward[None, :, :]
            states = weighted + state[:, None] * initial[None, :, :, None]
            normalizers = weighted_z + normalizer[:, None] * initial[None, :, :]
        state = states[:, -1]
        normalizer = normalizers[:, -1]
        state_chunks.append(states)
        normalizer_chunks.append(normalizers)

    if total == 0:
        states_all = contributions
        normalizers_all = normalizer_contributions
    else:
        states_all = torch.cat(state_chunks, dim=1)
        normalizers_all = torch.cat(normalizer_chunks, dim=1)
    return ScanResult(states_all, normalizers_all, state, normalizer)


def sequential_discounted_scan(
    contributions: torch.Tensor,
    normalizer_contributions: torch.Tensor,
    decay: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    initial_normalizer: torch.Tensor | None = None,
) -> ScanResult:
    batch, total, slots, width = contributions.shape
    state = contributions.new_zeros((batch, slots, width)) if initial_state is None else initial_state
    normalizer = normalizer_contributions.new_zeros((batch, slots)) if initial_normalizer is None else initial_normalizer
    states: list[torch.Tensor] = []
    normalizers: list[torch.Tensor] = []
    for index in range(total):
        if decay.ndim == 0:
            state = decay * state + contributions[:, index]
            normalizer = decay * normalizer + normalizer_contributions[:, index]
        else:
            state = decay[None, :, None] * state + contributions[:, index]
            normalizer = decay[None, :] * normalizer + normalizer_contributions[:, index]
        states.append(state)
        normalizers.append(normalizer)
    if total == 0:
        states_all = contributions
        normalizers_all = normalizer_contributions
    else:
        states_all = torch.stack(states, dim=1)
        normalizers_all = torch.stack(normalizers, dim=1)
    return ScanResult(states_all, normalizers_all, state, normalizer)


def run_demo(length: int, chunk_size: int, device: torch.device) -> dict[str, object]:
    torch.manual_seed(20260829)
    contributions = torch.randn(1, length, 3, 8, device=device) * 0.05
    normalizers = torch.rand(1, length, 3, device=device) * 0.1
    decay = torch.tensor([0.90, 0.97, 0.995], device=device)
    chunked = chunkwise_discounted_scan(
        contributions, normalizers, decay, chunk_size=chunk_size
    )
    reference_length = min(length, 1024)
    reference = sequential_discounted_scan(
        contributions[:, :reference_length],
        normalizers[:, :reference_length],
        decay,
    )
    comparison = chunkwise_discounted_scan(
        contributions[:, :reference_length],
        normalizers[:, :reference_length],
        decay,
        chunk_size=chunk_size,
    )
    return {
        "device": str(device),
        "length": length,
        "chunk_size": chunk_size,
        "finite": bool(torch.isfinite(chunked.states).all()),
        "final_state_finite": bool(torch.isfinite(chunked.final_state).all()),
        "reference_length": reference_length,
        "max_abs_error_vs_sequential": float(
            (comparison.states - reference.states).abs().max().cpu()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.length <= 0:
        raise SystemExit("--length must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    )
    result = run_demo(args.length, args.chunk_size, device)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
