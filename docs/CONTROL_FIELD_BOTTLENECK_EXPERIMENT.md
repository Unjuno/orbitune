# ControlField compact-input bottleneck experiment

This model-free experiment measures how aggressively the information sent into the adaptive RBF controller can be compressed before long-range tuning quality degrades.

## Setup

Eight synthetic control features define a target curve containing trend, periodic behavior and a local bump. A small learned encoder compresses those features to a bottleneck of `d` dimensions. A nonlinear controller expands that bottleneck into 12 adaptive Gaussian RBFs (center, width and amplitude all adaptive).

Environment:

```text
PyTorch 2.10.0+cpu
CPU threads: 5
RBF count: 12
curve length: 128
train curves: 900
held-out curves: 300
optimizer: AdamW
```

## Results

At 500 training steps the bottleneck models were still under-converged. Extending the main comparison to 1000 steps produced:

| Bottleneck dimensions | Mean held-out MSE | Seed std | Sharp-curve MSE |
|---:|---:|---:|---:|
| 4 | 0.06912 | 0.00536 | 0.06653 |
| 6 | 0.03992 | 0.00445 | 0.03760 |
| 8 | 0.03899 | 0.00454 | 0.03623 |

A narrower follow-up around the apparent knee, using two additional seeds, gave:

| Bottleneck dimensions | Mean held-out MSE | Seed std | Sharp-curve MSE |
|---:|---:|---:|---:|
| 5 | 0.04948 | 0.01023 | 0.04690 |
| 6 | **0.02870** | 0.00827 | **0.03043** |
| 7 | 0.03103 | 0.01277 | 0.03281 |

The exact values vary by seed, but the qualitative result is consistent: four or five dimensions are clearly restrictive, while six dimensions captures most of the benefit seen at seven or eight dimensions.

## Interpretation

The current optimization candidate is therefore:

```text
~6 compact control values / token-equivalents
  -> small nonlinear controller
  -> 12 adaptive Gaussian RBFs
       center adaptive
       width adaptive
       amplitude adaptive
  -> continuous musical-time ControlField
```

This does not imply that the final tokenizer must expose six scalar tokens literally. The six-dimensional compact representation could be produced by a small number of discrete control tokens, embeddings, or a mixed discrete/continuous control encoding. The important result is that the controller does not appear to require a high-dimensional external specification to recover most of the adaptive-field performance.

## Relation to basis-ordering experiment

Adaptive RBF tuples should remain unconstrained during optimization. For inspection/export, complete `(center, width, amplitude)` tuples can be sorted by center after generation without materially changing the represented field; a float32 numerical check found maximum reconstruction difference about `9.54e-7` due only to summation order.

## Next experiment

The next target is discrete quantization. Quantize the approximately six-dimensional compact control representation to finite token vocabularies and measure the rate-distortion curve: token count / bits versus ControlField reconstruction error. This directly determines whether Orbitune can expose a very small control vocabulary without giving up the tuning power found in the continuous optimization experiments.
