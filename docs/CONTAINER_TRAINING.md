# Container training smoke results

This document records measured engineering smoke tests for the fixed Orbitune v0 model scale. These tests do not use a real MIDI corpus and are not music-quality benchmarks.

## Environment

- Date: 2026-08-19
- Runtime: Linux x86_64 container
- CPU: AMD EPYC 9V74 virtualized, 5 vCPUs available
- Accelerator: none (CPU only)
- PyTorch: 2.10.0+cpu
- Model: `orbitune-tiny-v0`
- Architecture: 4 Transformer layers, hidden size 240, 4 attention heads, context limit 512
- Theory-REMI v0 vocabulary: 204 tokens
- Parameter count: 2,945,760

## 100-step measured training smoke run

Synthetic grammar-valid Theory-REMI patterns were used deliberately so that the test answers one narrow question: can the fixed 3M model and a LoRA adapter actually train in the target container without reducing the architecture?

### Base

- trainable parameters: 2,945,760
- batch size: 4
- sequence length: 64
- optimizer: AdamW
- learning rate: 5e-4
- steps: 100
- initial loss: 5.3546
- final loss: 0.1725

### Rank-4 LoRA

- target modules: `q_proj`, `v_proj`
- trainable parameters: 15,360
- batch size: 4
- sequence length: 64
- learning rate: 1e-3
- steps: 100
- initial loss: 7.3506
- final loss: 2.3853
- serialized adapter size: approximately 63 KB (Safetensors)

### Runtime and artifacts

- combined measured runtime: approximately 8.75 seconds in this container
- base checkpoint: approximately 11.8 MB in the unquantized PyTorch checkpoint format used by this smoke harness
- adapter: approximately 63 KB

## Full-context autoregressive inference baseline

A second measurement used the same fixed 2,945,760-parameter architecture on the same 5-vCPU container. The benchmark generated 256 tokens by rerunning the complete Transformer context for every new token; no KV cache was used.

- generated tokens: 256
- total measured time: approximately 2.08 seconds
- mean latency: approximately 8.1 ms/token
- throughput: approximately 123 tokens/second
- device: CPU
- implementation: PyTorch eager, full-context recomputation

Reproduce with:

```bash
python scripts/benchmark_inference.py \
  --tokens 256 \
  --device cpu \
  --threads 5
```

This number is **not** a smartphone or browser benchmark. WASM, JavaScript overhead, mobile CPU characteristics, thermal limits, and ONNX graph behavior can materially change latency. It is only a reference point for the current architecture.

## Interpretation

PASS:

- the fixed 2.95M-parameter architecture is small enough for CPU training smoke tests in the available container;
- rank-4 LoRA training is substantially smaller than Base training;
- full-context autoregressive inference is already inexpensive enough to justify testing a simple WASM baseline before introducing KV-cache complexity.

NOT PROVEN:

- real-corpus music quality;
- long-context musical quality;
- smartphone inference speed;
- ONNX/WASM speed;
- INT8 accuracy;
- browser memory use;
- adapter style separation on human listening tests.

The next meaningful training experiment must use a rights-cleared MIDI corpus through `orbitune prepare-split-corpus` and report held-out loss plus generated-MIDI structural/listening metrics.

## Reproduce training smoke

```bash
python scripts/smoke_train.py \
  --base-steps 100 \
  --adapter-steps 100 \
  --device cpu \
  --out smoke-training-report.json
```
