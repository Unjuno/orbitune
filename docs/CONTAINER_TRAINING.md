# Container training smoke results

This document records a measured engineering smoke test for the fixed Orbitune v0 model scale. It is not a music-quality benchmark and it does not use a real MIDI corpus.

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

## 100-step measured smoke run

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

## Interpretation

PASS: the fixed 2.95M-parameter architecture is small enough for CPU training smoke tests in the available container, and rank-4 LoRA training is substantially smaller than Base training.

NOT PROVEN: real-corpus music quality, long-context quality, smartphone inference speed, INT8 accuracy, browser inference, or style separation on human listening tests.

The next meaningful training experiment must use a rights-cleared MIDI corpus through `orbitune prepare-corpus` and report both held-out loss and generated-MIDI quality metrics.

## Reproduce

```bash
python scripts/smoke_train.py \
  --base-steps 100 \
  --adapter-steps 100 \
  --device cpu \
  --out smoke-training-report.json
```
