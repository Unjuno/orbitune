# Model-free ControlField basis optimization

This experiment isolates ControlField representation from the Transformer. No language model is trained. The goal is to find a compact function family that can represent long-range control curves with very few control tokens.

## Setup

- 240 synthetic control curves
- 256 normalized musical-time positions
- curve classes: smooth, mixed, spiky
- basis families: Fourier, piecewise linear, normalized Gaussian RBF
- basis counts: 4, 8, 12, 16
- local override counts: 0, 1, 2, 4
- fixed random seed: 42

The objective is reconstruction MSE. A global ControlField counts as one control token; each local override adds one token-equivalent.

## Main measured result

The best global-only configuration was 16 RBF bases:

```text
all MSE     0.001423
smooth MSE  0.0000183
mixed MSE   0.0000516
spiky MSE   0.006320
```

Adding sparse local overrides mainly helps the spiky/high-frequency cases:

```text
RBF-16 + 1 local override: all MSE 0.000890
RBF-16 + 2 local overrides: all MSE 0.000624
RBF-16 + 4 local overrides: all MSE 0.000380
```

Under an equivalent control-token budget, the best observed configurations were:

```text
1 token : RBF-16 + 0 local, MSE 0.001423
2 tokens: RBF-16 + 1 local, MSE 0.000890
3 tokens: RBF-16 + 2 local, MSE 0.000624
5 tokens: RBF-16 + 4 local, MSE 0.000380
```

For comparison, global-only 8-basis errors were:

```text
Fourier-8  0.006782
Linear-8   0.015725
RBF-8      0.008777
```

At 12 bases RBF improved sharply to 0.002280, and at 16 bases to 0.001423.

## Interpretation

The model-free optimization supports a two-level control representation:

```text
smooth long-range behavior -> global RBF ControlField
abrupt/high-frequency events -> sparse local override tokens
```

A fixed 8-basis design is probably too restrictive if Orbitune wants the ControlField itself to cover more complex timing envelopes. Sixteen RBF bases remain extremely small computationally and substantially reduce approximation error. Local overrides are still useful because narrow spikes remain the dominant residual error even at 16 bases.

This does not determine the final tokenizer or ControlField ABI. Real MIDI timing distributions, tempo changes, and microtiming still need to be evaluated.

Reproduction: `python experiments/control_field_basis_optimization.py`.
