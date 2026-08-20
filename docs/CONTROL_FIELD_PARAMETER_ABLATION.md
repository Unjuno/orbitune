# ControlField adaptive-parameter ablation

This model-free experiment asks which Gaussian RBF degrees of freedom should remain adaptive when a small controller expands a few control values into a long ControlField.

## Setup

Synthetic held-out curves combine trend, periodic motion and a local Gaussian bump. Each controller receives eight compact control features and emits parameters for 12 Gaussian RBFs. Five seeds are used for the main ablation.

Environment:

```text
PyTorch 2.10.0+cpu
CPU threads: 5
RBF count: 12
curve length: 128
train curves: 900
held-out curves: 300
optimizer: AdamW
training: 500 steps/seed
```

## Main ablation

| Adaptive parameters | Mean held-out MSE | Seed std | Sharp-curve MSE | Controller params |
|---|---:|---:|---:|---:|
| center + width + amplitude | **0.03457** | 0.00308 | **0.04006** | 7,076 |
| center + amplitude | 0.04890 | **0.00163** | 0.05127 | 6,296 |
| width + amplitude | 0.06483 | 0.00105 | 0.06332 | 6,296 |
| amplitude only | 0.08706 | 0.00188 | 0.08482 | 5,516 |

The experiment therefore rejects the idea that amplitude-only adaptation is sufficient. Center placement matters more than width adaptation in isolation, but the best accuracy comes from adapting all three quantities.

## Ordering experiments

Unconstrained adaptive centers frequently change basis order (about half of neighboring pairs are inverted when inspected by raw basis index). This is not a functional error: an RBF sum is invariant to permutation of its basis tuples.

Several attempts to force ordered centers were tested:

- anchor-residual centers: stable but higher MSE (~0.0518 for all-adaptive residual form)
- soft ordering penalty: weak penalties do not resolve permutation symmetry; strong penalties hurt MSE
- structurally monotonic cumulative-gap centers: zero crossings but MSE ~0.0416

The best current decision is therefore **not to constrain basis order during optimization**. For serialization, inspection or debugging, sort complete `(center, width, amplitude)` tuples by center after the controller emits them.

A numerical check on 64 random fields found that tuple sorting changed the reconstructed field by at most about `9.54e-7` in float32 (mean absolute difference `7.43e-8`), attributable only to floating-point summation order.

## Interpretation

Current candidate:

```text
small control representation
  -> nonlinear controller
  -> 12 adaptive Gaussian RBFs
       center: adaptive
       width: adaptive
       amplitude: adaptive
  -> optional post-hoc canonical sort for inspection/export
  -> ControlField in musical-time coordinates
```

`CONTROL_NONE` remains a structural zero-field path and does not need RBF parameters.

The result does **not** imply that 12 RBFs are permanently optimal. It establishes that removing center or width freedom to simplify the controller costs substantial function-approximation accuracy on mixed smooth/local behavior.

## Next experiment

The next optimization target is the compact representation before the controller: quantify how far its dimensionality / token-equivalent count can be reduced (for example 2, 3, 4, 6, 8 control values) before the adaptive RBF reconstruction degrades materially. That directly tests the original objective: minimum tokenized control information for maximum long-range tuning ability.
