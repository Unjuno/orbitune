# BitNet + LoRA container experiment

This experiment compares the current ~10.2M Orbitune reference-shape Transformer with a ternary BitLinear variant trained with a straight-through estimator (STE), then tests rank-4 LoRA on the ternary Base.

## Environment

- PyTorch 2.10.0+cpu
- 5 CPU threads
- batch 4
- sequence length 64
- synthetic MIDI-like token stream
- 4 layers, hidden 448, 7 heads
- 10,215,296 parameters with the experiment vocabulary

## FP vs ternary training

At 40 steps with LR 3e-4:

| model | held-out loss | elapsed | tokens/s | training checkpoint | estimated deployment weight size |
|---|---:|---:|---:|---:|---:|
| FP | 4.948 | 6.30 s | 1626 | 40.87 MB | 20.43 MB (FP16 estimate) |
| ternary | 11.552 | 7.46 s | 1372 | 40.87 MB | 3.57 MB |

The training checkpoint remains large because STE training retains floating-point master weights. The estimated ternary deployment size packs BitLinear weights at 2 bits/weight and retains non-BitLinear parameters in FP16.

The initial gap was strongly optimizer-dependent. At 60 steps:

- ternary, LR 8e-4: held-out loss ~2.934
- ternary, LR 1.5e-3: held-out loss ~3.070
- FP, LR 8e-4: held-out loss ~2.208

So the ternary model learns successfully and much of the initial gap was optimization mismatch, but FP still wins on held-out loss in this synthetic test.

## LoRA on ternary Base

After training the ternary Base, q/v projections were frozen and rank-4 LoRA residuals were added.

- trainable LoRA parameters: 28,672
- FP16 LoRA size estimate: 56 KB
- style-distribution loss before LoRA: 39.63
- style-distribution loss after 50 LoRA steps: 29.04
- original Base-distribution loss before LoRA: 2.93
- original Base-distribution loss after LoRA: 4.69

This shows that LoRA can adapt a ternary BitLinear Base. The Base-distribution degradation means rank, target modules, regularization and mixed-corpus SFT still need tuning.

## Current interpretation

PASS for feasibility:

1. ~10M ternary Orbitune trains with STE on CPU.
2. Its deployment representation can be much smaller than FP16.
3. Floating-point LoRA can be trained on top of the ternary Base.

NOT YET proven:

1. parity with FP on real MIDI;
2. actual mobile/Web speedup, which requires ternary kernels rather than dequantizing to float GEMM;
3. final BitNet quantization recipe (activation quantization, normalization and optimizer recipe are still experimental here);
4. preservation of ControlField quality on a ternary Base.

The next experiment should combine the ternary Base with the current ControlField candidate and compare FP vs ternary using identical ControlField and MIDI-like data.