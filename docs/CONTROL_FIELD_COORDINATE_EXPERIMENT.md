# ControlField coordinate experiment

This experiment tests whether Orbitune should evaluate long-range control curves over raw token index or over reconstructed musical time.

## Hypothesis

If event density changes, token index and musical time diverge. A control curve attached to token position should therefore warp in time, while a curve attached to cumulative musical time should remain phase-aligned with the intended musical behavior.

## Field-only experiment

Environment:

```text
PyTorch 2.10.0+cpu
CPU threads: 5
sequence positions: 96
ControlField parameters: 3,800
basis functions: 8 cosine bases
channels: TIMING / DENSITY / VELOCITY
training: 600 steps
```

Synthetic sequences used strongly varying event density. The target controls were defined in normalized cumulative musical time.

Measured held-out results:

| Coordinate | MSE | Overall correlation | WAVE timing correlation |
|---|---:|---:|---:|
| token position | 0.03350 | 0.7944 | 0.7327 |
| musical time | 0.000211 | 0.9986 | 0.9979 |

`CONTROL_NONE` remained structurally exact zero (`max_abs = 0.0`).

Interpretation: when density changes, token-position control is not temporally stable. Musical-time coordinates reduce field error by roughly two orders of magnitude in this synthetic setting.

## 10M-class Transformer experiment

A second experiment used the reference-width Transformer shape:

```text
4 layers
hidden 448
7 heads
1024 position capacity
96-token synthetic event vocabulary
10,159,960 parameters including ControlField
sequence: 47 prediction positions
batch: 4
training: 45 steps per coordinate mode
```

Measured held-out causal LM losses:

| Coordinate | Final train loss | Held-out LM loss | Control forced NONE | Delta |
|---|---:|---:|---:|---:|
| token position | 3.7916 | 3.6362 | 3.6424 | +0.0061 |
| musical time | 3.1513 | 3.4573 | 3.4651 | +0.0078 |

The musical-time model reached a lower held-out loss after the same short training budget. The control-off deltas remain small at 45 steps, so this run is evidence for coordinate choice, not a complete control-adherence proof.

An attempted 100-step run for both models exceeded the container command time limit and was discarded; no partial numbers from that run are used.

## Decision

The architecture candidate should evaluate the global ControlField over **musical time**, not raw token position.

The practical implementation should maintain a cumulative musical-time coordinate derived from TIME_SHIFT / tempo-aware event timing and sample the 8-basis global field at that coordinate. Local override tokens remain available for abrupt changes.

Do not freeze this behavior into a published Base ABI until the production MIDI tokenizer and real-MIDI experiment pass.
