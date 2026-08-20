# Inference performance simulator

Orbitune includes a simple roofline-style simulator in `tools/perf_sim.py` for comparing FP16, INT8 and packed ternary deployment assumptions before running on real phones or browsers.

The simulator estimates a lower-bound-ish latency from two bottlenecks:

- weight-memory traffic, parameterized by effective memory bandwidth
- arithmetic work, parameterized by effective GOPS

It then adds an optional runtime overhead term for JS/WASM/ORT/session overhead that must be calibrated from real devices.

The current reference shape (4 layers, hidden 448, prospective vocab 1200, context 1024) is about 10.18M parameters and about 13.3M MACs/token with KV caching.

Example:

```bash
python tools/perf_sim.py --format ternary --bandwidth-gbs 35 --effective-gops 60 --runtime-overhead-ms 1.5
```

Important limitations:

1. The numbers are not phone benchmarks. They are model-based estimates.
2. `ternary` assumes a genuine packed ternary kernel. Expanding weights to float before GEMM invalidates most compute-speed assumptions.
3. The model currently approximates KV-cached autoregressive decoding. Full-context recomputation is much more expensive.
4. Activation traffic, cache misses, browser boundary costs, threading efficiency and operator fusion are not modeled explicitly; they are absorbed into calibration/overhead.
5. The simulator should be calibrated against at least one real device per runtime/backend before making product claims.

A useful decision process is to evaluate quality and runtime as a Pareto frontier rather than selecting ternary purely for size.
