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
        # Dense linears plus attention score/value work at the configured context.
        return self.linear_params + 2 * self.layers * self.context * self.hidden


FORMATS = {
    "fp16": dict(bits=16, ops_per_mac=2.0, overhead=1.00),
    "int8": dict(bits=8, ops_per_mac=2.0, overhead=1.10),
    # Represents a packed ternary deployment kernel. It is not the current PyTorch STE implementation.
    "ternary": dict(bits=2, ops_per_mac=1.0, overhead=1.25),
}


def predict(spec: ModelSpec, *, fmt: str, bandwidth_gbs: float, effective_gops: float, runtime_overhead_ms: float = 0.0) -> dict[str, float]:
    f = FORMATS[fmt]
    weight_mb = spec.params * f["bits"] / 8 / 1e6
    bandwidth_ms = weight_mb / (bandwidth_gbs * 1000.0) * 1000.0
    compute_ms = spec.macs_per_token_with_kv * f["ops_per_mac"] * f["overhead"] / (effective_gops * 1e9) * 1000.0
    floor_ms = max(bandwidth_ms, compute_ms)
    predicted_ms = floor_ms + runtime_overhead_ms
    return {
        "params": float(spec.params),
        "weight_mb": weight_mb,
        "bandwidth_floor_ms_per_token": bandwidth_ms,
        "compute_floor_ms_per_token": compute_ms,
        "predicted_ms_per_token": predicted_ms,
        "predicted_tokens_per_second": 1000.0 / predicted_ms,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Orbitune roofline-style inference simulator")
    p.add_argument("--format", choices=sorted(FORMATS), default="ternary")
    p.add_argument("--bandwidth-gbs", type=float, required=True)
    p.add_argument("--effective-gops", type=float, required=True)
    p.add_argument("--runtime-overhead-ms", type=float, default=0.0)
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--vocab", type=int, default=1200)
    args = p.parse_args()
    spec = ModelSpec(context=args.context, vocab=args.vocab)
    out = predict(spec, fmt=args.format, bandwidth_gbs=args.bandwidth_gbs, effective_gops=args.effective_gops, runtime_overhead_ms=args.runtime_overhead_ms)
    for k, v in out.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
