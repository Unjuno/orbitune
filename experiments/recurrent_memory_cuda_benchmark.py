from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from recurrent_memory_chunkwise_scan import chunkwise_discounted_scan  # noqa: E402


@dataclass(slots=True)
class BenchResult:
    kernel: str
    length: int
    status: str
    milliseconds: float | None
    tokens_per_second: float | None
    peak_memory_bytes: int | None
    state_or_cache_bytes: int | None


def _dtype(name: str, device: torch.device) -> torch.dtype:
    mapping = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = mapping[name]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("CPU benchmark does not support fp16")
    return dtype


def _time(callable_, device: torch.device, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        callable_()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            callable_()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end)) / iterations
    started = time.perf_counter()
    for _ in range(iterations):
        callable_()
    return (time.perf_counter() - started) * 1000 / iterations


class LinearMemoryBench(nn.Module):
    def __init__(self, d_model: int, slots: int, chunk_size: int) -> None:
        super().__init__()
        self.slots = slots
        self.chunk_size = chunk_size
        self.norm = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, slots, bias=False)
        self.k = nn.Linear(d_model, slots, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.write = nn.Linear(d_model, 1)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.logit_decay = nn.Parameter(torch.tensor(5.3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = F.elu(self.q(h)) + 1
        k = F.elu(self.k(h)) + 1
        v = self.v(h)
        write = torch.sigmoid(self.write(h))
        decay = torch.sigmoid(self.logit_decay).clamp(0.9, 0.9999)
        contributions = write.unsqueeze(-1) * torch.einsum("btk,btd->btkd", k, v)
        normalizers = write * k
        scan = chunkwise_discounted_scan(
            contributions.float(),
            normalizers.float(),
            decay.float(),
            chunk_size=self.chunk_size,
        )
        read = torch.einsum("btk,btkd->btd", q.float(), scan.states) / (
            torch.einsum("btk,btk->bt", q.float(), scan.normalizers).unsqueeze(-1) + 1e-5
        )
        return self.out(read.to(x.dtype))

    def recurrent_step(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        normalizer: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.norm(x)
        q = F.elu(self.q(h)) + 1
        k = F.elu(self.k(h)) + 1
        v = self.v(h)
        write = torch.sigmoid(self.write(h))
        decay = torch.sigmoid(self.logit_decay).clamp(0.9, 0.9999).float()
        state = decay * state + write.float().unsqueeze(-1) * torch.einsum(
            "bk,bd->bkd", k.float(), v.float()
        )
        normalizer = decay * normalizer + write.float() * k.float()
        read = torch.einsum("bk,bkd->bd", q.float(), state) / (
            torch.einsum("bk,bk->b", q.float(), normalizer).unsqueeze(-1) + 1e-5
        )
        return self.out(read.to(x.dtype)), state, normalizer


class SDPABench(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def project(self, x: torch.Tensor):
        batch, length, _ = x.shape
        qkv = self.qkv(self.norm(x)).view(
            batch, length, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.project(x)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(x.shape)
        return self.out(y)


def _peak_memory(device: torch.device, callable_) -> int | None:
    if device.type != "cuda":
        callable_()
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    callable_()
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def benchmark_parallel(
    kernel: str,
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
    state_or_cache_bytes: int | None,
) -> BenchResult:
    try:
        with torch.inference_mode():
            peak = _peak_memory(device, lambda: model(x))
            ms = _time(lambda: model(x), device, warmup=warmup, iterations=iterations)
        return BenchResult(
            kernel,
            x.shape[1],
            "ok",
            ms,
            x.shape[0] * x.shape[1] * 1000 / ms,
            peak,
            state_or_cache_bytes,
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return BenchResult(kernel, x.shape[1], "oom", None, None, None, state_or_cache_bytes)


def benchmark_streaming(
    linear: LinearMemoryBench,
    sdpa: SDPABench,
    x: torch.Tensor,
    device: torch.device,
) -> list[BenchResult]:
    batch, length, d_model = x.shape
    state = torch.zeros(batch, linear.slots, d_model, device=device, dtype=torch.float32)
    normalizer = torch.zeros(batch, linear.slots, device=device, dtype=torch.float32)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(length):
            _, state, normalizer = linear.recurrent_step(x[:, index], state, normalizer)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        linear_peak = int(torch.cuda.max_memory_allocated(device))
    else:
        linear_peak = None
    linear_ms = (time.perf_counter() - started) * 1000
    linear_state_bytes = (state.numel() + normalizer.numel()) * state.element_size()

    head_dim = sdpa.head_dim
    k_cache = torch.empty(batch, sdpa.heads, length, head_dim, device=device, dtype=x.dtype)
    v_cache = torch.empty_like(k_cache)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(length):
            q, k, v = sdpa.project(x[:, index : index + 1])
            k_cache[:, :, index : index + 1] = k
            v_cache[:, :, index : index + 1] = v
            y = F.scaled_dot_product_attention(
                q,
                k_cache[:, :, : index + 1],
                v_cache[:, :, : index + 1],
                is_causal=False,
            )
            sdpa.out(y.transpose(1, 2).reshape(batch, 1, d_model))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        sdpa_peak = int(torch.cuda.max_memory_allocated(device))
    else:
        sdpa_peak = None
    sdpa_ms = (time.perf_counter() - started) * 1000
    kv_bytes = (k_cache.numel() + v_cache.numel()) * k_cache.element_size()

    return [
        BenchResult(
            "linear_recurrent_stream",
            length,
            "ok",
            linear_ms,
            batch * length * 1000 / linear_ms,
            linear_peak,
            int(linear_state_bytes),
        ),
        BenchResult(
            "sdpa_kv_stream",
            length,
            "ok",
            sdpa_ms,
            batch * length * 1000 / sdpa_ms,
            sdpa_peak,
            int(kv_bytes),
        ),
    ]


def run(args) -> dict[str, object]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    dtype = _dtype(args.dtype, device)
    if args.d_model % args.heads:
        raise SystemExit("--d-model must be divisible by --heads")
    linear = LinearMemoryBench(args.d_model, args.slots, args.chunk_size).to(device=device, dtype=dtype).eval()
    sdpa = SDPABench(args.d_model, args.heads).to(device=device, dtype=dtype).eval()
    results: list[BenchResult] = []
    for length in args.lengths:
        x = torch.randn(args.batch, length, args.d_model, device=device, dtype=dtype)
        linear_state_bytes = args.batch * (args.slots * args.d_model + args.slots) * 4
        kv_bytes = args.batch * length * args.d_model * 2 * torch.tensor([], dtype=dtype).element_size()
        results.append(
            benchmark_parallel(
                "linear_parallel_scan",
                linear,
                x,
                device,
                warmup=args.warmup,
                iterations=args.iterations,
                state_or_cache_bytes=linear_state_bytes,
            )
        )
        results.append(
            benchmark_parallel(
                "sdpa_full_causal",
                sdpa,
                x,
                device,
                warmup=args.warmup,
                iterations=args.iterations,
                state_or_cache_bytes=int(kv_bytes),
            )
        )
    stream_length = max(args.lengths)
    stream_x = torch.randn(args.batch, stream_length, args.d_model, device=device, dtype=dtype)
    results.extend(benchmark_streaming(linear, sdpa, stream_x, device))
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "schema_version": 1,
        "device": str(device),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dtype": args.dtype,
        "batch": args.batch,
        "d_model": args.d_model,
        "heads": args.heads,
        "slots": args.slots,
        "chunk_size": args.chunk_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": [asdict(item) for item in results],
        "notes": {
            "linear_parallel_scan": "training-style scan; recurrent accumulation is fp32",
            "linear_recurrent_stream": "fixed-size recurrent state; no history cache",
            "sdpa_kv_stream": "preallocated KV cache grows linearly with requested max length",
            "cpu": "CPU timings are smoke diagnostics only; VLab16 CUDA is the performance gate"
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Linear-memory versus SDPA benchmark")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--lengths", type=lambda value: tuple(int(v) for v in value.split(",")), default=(256, 512, 1024, 2048, 4096))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cpu" and args.dtype == "fp16":
        args.dtype = "fp32"
    result = run(args)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
