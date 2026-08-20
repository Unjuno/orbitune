# Adaptive RBF control experiment

This experiment isolates ControlField optimization from the Transformer. The goal is to test whether a small number of control quantities can drive a nonlinear controller that produces an entire timing/control function.

## Experiment A: fixed vs globally optimized basis

Dataset: 240 synthetic control curves, split across `smooth`, `mixed`, and `spiky` families. Each curve has 256 musical-time positions.

A fixed 16-RBF basis already fits smooth curves extremely well, but local spikes dominate residual error.

Held-out mean MSE:

| basis | smooth | mixed | spiky |
|---|---:|---:|---:|
| fixed RBF-16 | 1.60e-6 | 7.17e-6 | 1.352e-2 |
| globally optimized centers+widths | 4.57e-5 | 3.04e-4 | 1.317e-2 |

Globally learning one common set of centers and widths gives only a small improvement on spiky curves and harms smooth/mixed curves. It is therefore not recommended.

## Experiment B: per-control adaptive RBF upper bound

Allowing center, width and amplitude to adapt per control curve gives a much stronger upper bound:

| adaptive basis | smooth mean MSE | mixed mean MSE | spiky mean MSE |
|---|---:|---:|---:|
| RBF-8 | 2.66e-5 | 2.19e-4 | 2.34e-3 |
| RBF-12 | 1.08e-5 | 2.15e-5 | 2.17e-4 |
| RBF-16 | 9.06e-6 | 6.80e-6 | 3.22e-5 |

This shows that adaptive center/width placement is genuinely useful for abrupt local behavior.

## Experiment C: sparse local Gaussian overrides

Starting from the fixed RBF-16 global field and greedily adding optimized local Gaussian corrections gives:

| overrides | smooth | mixed | spiky |
|---:|---:|---:|---:|
| 1 | 1.31e-6 | 6.22e-6 | 9.34e-3 |
| 2 | 1.04e-6 | 5.38e-6 | 6.64e-3 |
| 4 | 7.00e-7 | 4.02e-6 | 3.63e-3 |

Local overrides help, but they do not reach the adaptive-RBF upper bound with only a few overrides.

## Experiment D: few controls -> nonlinear controller -> entire field

To test the intended sparse-token design directly, six continuous control quantities were used as the only input:

- trend
- wave amplitude
- wave frequency
- local bump amplitude
- local bump center
- local bump width

A small two-layer MLP maps these six quantities to RBF parameters. There is no Transformer in this test.

Held-out results:

| controller | held-out MSE | trainable parameters |
|---|---:|---:|
| fixed RBF-16 coefficients | 0.01663 | 11,633 |
| fixed RBF-24 coefficients | 0.01101 | 12,409 |
| **adaptive RBF-12 (center+width+amplitude)** | **0.00129** | 13,573 |
| adaptive RBF-16 | 0.00153 | 14,737 |

Adaptive RBF-12 was best in this run. Three additional random seeds for adaptive RBF-12 produced held-out MSE values 0.00374, 0.00446 and 0.00173 (mean 0.00331, std 0.00115), so optimization variance is material and must be monitored.

## Interpretation

The strongest current candidate is not a globally learned basis shared by every control. Instead:

```text
few control tokens / quantities
        -> nonlinear ControlField controller
        -> adaptive RBF centers + widths + amplitudes
        -> continuous field in musical time
        -> optional sparse local override for pathological transients
```

The key point is that outputting 37 internal RBF parameters does **not** require 37 user-visible tokens. A small control embedding or a few factorized control tokens can be expanded by the controller into those internal parameters.

## Current decision

- Keep `NONE` structurally zero.
- Use musical-time coordinates.
- Prefer adaptive RBF-12 as the next candidate.
- Keep local override events as an escape hatch for abrupt/high-frequency control.
- Do not freeze this into the Base ABI until real-MIDI experiments confirm the result.
