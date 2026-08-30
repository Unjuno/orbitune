from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import recurrent_memory_cuda_benchmark as base  # noqa: E402


def _read_cold_state(
    linear: base.LinearMemoryBench,
    x: torch.Tensor,
    state: torch.Tensor,
    normalizer: torch.Tensor,
) -> torch.Tensor:
    h = linear.norm(x)
    q = F.elu(linear.q(h)) + 1
    read = torch.einsum("bk,bkd->bd", q.float(), state) / (
        torch.einsum("bk,bk->b", q.float(), normalizer).unsqueeze(-1) + 1e-5
    )
    return linear.out(read.to(x.dtype))


def benchmark_hot_cold_stream(
    linear: base.LinearMemoryBench,
    sdpa: base.SDPABench,
    x: torch.Tensor,
    device: torch.device,
    *,
    hot_window: int,
) -> base.BenchResult:
    """Benchmark the current reference streaming partition.

    Events remain exact in a bounded hot KV cache. Once an event falls out of
    that window it is committed exactly once to the fixed-size recurrent cold
    state. The cold memory read conditions the current representation before
    bounded local attention, matching the CPU architecture ablation.
    """

    if hot_window <= 0:
        raise ValueError("hot_window must be positive")
    batch, length, d_model = x.shape
    state = torch.zeros(
        batch, linear.slots, d_model, device=device, dtype=torch.float32
    )
    normalizer = torch.zeros(
        batch, linear.slots, device=device, dtype=torch.float32
    )
    cache_length = min(hot_window, length)
    k_cache = torch.empty(
        batch,
        sdpa.heads,
        cache_length,
        sdpa.head_dim,
        device=device,
        dtype=x.dtype,
    )
    v_cache = torch.empty_like(k_cache)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(length):
            if index >= hot_window:
                evicted = x[:, index - hot_window]
                _, state, normalizer = linear.recurrent_step(
                    evicted, state, normalizer
                )

            current = x[:, index]
            if index >= hot_window:
                current = current + _read_cold_state(
                    linear, current, state, normalizer
                )

            q, k, v = sdpa.project(current.unsqueeze(1))
            slot = index % cache_length
            k_cache[:, :, slot : slot + 1] = k
            v_cache[:, :, slot : slot + 1] = v
            filled = min(index + 1, cache_length)
            if index + 1 <= cache_length:
                keys = k_cache[:, :, :filled]
                values = v_cache[:, :, :filled]
            else:
                # For this kernel benchmark the cache is circular. Absolute
                # chronological order does not affect one-query dot-product
                # attention; production local-position semantics are handled
                # outside this microbenchmark.
                keys = k_cache
                values = v_cache
            y = F.scaled_dot_product_attention(
                q, keys, values, is_causal=False
            )
            sdpa.out(y.transpose(1, 2).reshape(batch, 1, d_model))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = None
    milliseconds = (time.perf_counter() - started) * 1000
    state_bytes = (state.numel() + normalizer.numel()) * state.element_size()
    hot_kv_bytes = (k_cache.numel() + v_cache.numel()) * k_cache.element_size()
    return base.BenchResult(
        kernel="hot_cold_memory_first_stream",
        length=length,
        status="ok",
        milliseconds=milliseconds,
        tokens_per_second=batch * length * 1000 / milliseconds,
        peak_memory_bytes=peak,
        state_or_cache_bytes=int(state_bytes + hot_kv_bytes),
    )


def run(args) -> dict[str, object]:
    result = base.run(args)
    device = torch.device(result["device"])
    dtype = base._dtype(args.dtype, device)
    linear = base.LinearMemoryBench(
        args.d_model, args.slots, args.chunk_size
    ).to(device=device, dtype=dtype).eval()
    sdpa = base.SDPABench(args.d_model, args.heads).to(
        device=device, dtype=dtype
    ).eval()

    composite: list[base.BenchResult] = []
    for length in args.lengths:
        x = torch.randn(
            args.batch,
            length,
            args.d_model,
            device=device,
            dtype=dtype,
        )
        composite.append(
            benchmark_hot_cold_stream(
                linear,
                sdpa,
                x,
                device,
                hot_window=args.hot_window,
            )
        )

    result["hot_window"] = args.hot_window
    result["results"].extend(asdict(item) for item in composite)
    result["notes"]["hot_cold_memory_first_stream"] = (
        "fixed recurrent cold state + bounded hot KV cache; cold memory read "
        "conditions the current representation before local attention"
    )
    result["notes"]["hot_cold_state_scaling"] = (
        "state/cache bytes are independent of total generated length once "
        "length exceeds hot_window"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hot/cold recurrent-memory versus full-KV benchmark"
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument(
        "--dtype", choices=["fp32", "fp16", "bf16"], default="fp16"
    )
    parser.add_argument(
        "--lengths",
        type=lambda value: tuple(int(v) for v in value.split(",")),
        default=(256, 512, 1024, 2048, 4096),
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--hot-window", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hot_window <= 0:
        raise SystemExit("--hot-window must be positive")
    if args.device == "cpu" and args.dtype == "fp16":
        args.dtype = "fp32"
    result = run(args)
    Path(args.out).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
