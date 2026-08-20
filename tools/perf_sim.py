from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    layers: int = 4
    hidden: int = 448
    vocab: int = 1200
    context: int = 1024

    @property
    def linear_params(self) -> int:
        # Attention QKV+O = 4d^2, FFN = 8d^2.
        return self.layers * 12 * self.hidden * self.hidden

    @property
    def params(self) -> int:
        return self.linear_params + self.vocab * self.hidden + self.layers * 4 * self.hidden + self.hidden

    @property
    def macs_per_token_with_kv(self) -> int:
        return self.linear_params + 2 * self.layers * self.context * self.hidden

    @property
    def macs_per_token_without_kv(self) -> int:
        # Full-context recompute approximation.
        return self.linear_params * self.context + 2 * self.layers * self.context * self.context * self.hidden

    @property
    def kv_cache_mb_fp16(self) -> float:
        # K and V, per layer, fp16 cache.
        return 2 * self.layers * self.context * self.hidden * 2 / 1e6


FORMATS = {
    "fp16": dict(bits=16, ops_per_mac=2.0, compute_overhead=1.00, expansion_bytes=0.0),
    "int8": dict(bits=8, ops_per_mac=2.0, compute_overhead=1.10, expansion_bytes=0.0),
}

TERNARY_MODES = {
    # Packed ternary consumed directly by a specialized kernel.
    "native": dict(ops_per_mac=1.0, compute_overhead=1.20, expansion_bytes=0.0),
    # Packed weights unpacked to int8 before/during GEMM.
    "unpack-int8": dict(ops_per_mac=1.6, compute_overhead=1.45, expansion_bytes=1.0),
    # Packed weights expanded to fp16 then evaluated with ordinary float GEMM.
    "dequant-fp16": dict(ops_per_mac=2.0, compute_overhead=1.80, expansion_bytes=2.0),
}


def predict(
    spec: ModelSpec,
    *,
    fmt: str,
    bandwidth_gbs: float,
    effective_gops: float,
    runtime_overhead_ms: float = 0.0,
    kv_cache: bool = True,
    ternary_mode: str = "native",
    ternary_kernel_efficiency: float = 1.0,
) -> dict[str, float]:
    if bandwidth_gbs <= 0 or effective_gops <= 0:
        raise ValueError("bandwidth_gbs and effective_gops must be positive")
    if not 0 < ternary_kernel_efficiency <= 1:
        raise ValueError("ternary_kernel_efficiency must be in (0, 1]")

    if fmt == "ternary":
        mode = TERNARY_MODES[ternary_mode]
        bits = 2
        ops_per_mac = mode["ops_per_mac"]
        compute_overhead = mode["compute_overhead"]
        expansion_bytes = mode["expansion_bytes"]
        effective_gops *= ternary_kernel_efficiency
    else:
        mode = FORMATS[fmt]
        bits = mode["bits"]
        ops_per_mac = mode["ops_per_mac"]
        compute_overhead = mode["compute_overhead"]
        expansion_bytes = mode["expansion_bytes"]

    packed_weight_mb = spec.params * bits / 8 / 1e6
    expanded_weight_mb = spec.params * expansion_bytes / 1e6
    kv_mb = spec.kv_cache_mb_fp16 if kv_cache else 0.0
    streamed_mb = packed_weight_mb + expanded_weight_mb + kv_mb

    bandwidth_ms = streamed_mb / (bandwidth_gbs * 1000.0) * 1000.0
    macs = spec.macs_per_token_with_kv if kv_cache else spec.macs_per_token_without_kv
    compute_ms = macs * ops_per_mac * compute_overhead / (effective_gops * 1e9) * 1000.0
    floor_ms = max(bandwidth_ms, compute_ms)
    predicted_ms = floor_ms + runtime_overhead_ms

    return {
        "params": float(spec.params),
        "packed_weight_mb": packed_weight_mb,
        "kv_cache_mb": kv_mb,
        "streamed_mb_per_token_estimate": streamed_mb,
        "bandwidth_floor_ms_per_token": bandwidth_ms,
        "compute_floor_ms_per_token": compute_ms,
        "predicted_ms_per_token": predicted_ms,
        "predicted_tokens_per_second": 1000.0 / predicted_ms,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Orbitune roofline-style inference simulator")
    p.add_argument("--format", choices=["fp16", "int8", "ternary"], default="ternary")
    p.add_argument("--ternary-mode", choices=sorted(TERNARY_MODES), default="native")
    p.add_argument("--ternary-kernel-efficiency", type=float, default=1.0)
    p.add_argument("--bandwidth-gbs", type=float, required=True)
    p.add_argument("--effective-gops", type=float, required=True)
    p.add_argument("--runtime-overhead-ms", type=float, default=0.0)
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--vocab", type=int, default=1200)
    p.add_argument("--no-kv-cache", action="store_true")
    args = p.parse_args()

    spec = ModelSpec(context=args.context, vocab=args.vocab)
    out = predict(
        spec,
        fmt=args.format,
        ternary_mode=args.ternary_mode,
        ternary_kernel_efficiency=args.ternary_kernel_efficiency,
        bandwidth_gbs=args.bandwidth_gbs,
        effective_gops=args.effective_gops,
        runtime_overhead_ms=args.runtime_overhead_ms,
        kv_cache=not args.no_kv_cache,
    )
    for k, v in out.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
