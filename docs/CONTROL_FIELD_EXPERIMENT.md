# ControlField synthetic experiment

This experiment tests whether Orbitune can use a very small learned control network to modulate long-range MIDI-like timing, density and velocity behavior without inserting dense control tokens throughout the sequence.

## Architecture under test

The experiment uses the reference Transformer shape (4 layers, hidden 448, 7 heads, context capacity 1024) with a synthetic MIDI-like vocabulary and a short 48-token training sequence. The resulting model has about 10.13M parameters; the ControlField branch adds 7,384 parameters.

The control branch is:

```text
control id
  -> 32-d embedding
  -> 2-layer MLP
  -> 8 smooth basis coefficients x 3 channels
  -> continuous field over sequence positions
  -> FiLM gamma/beta
  -> every Transformer block
```

Control id 0 is structurally masked to exactly zero, so `NONE` cannot alter the hidden state through the ControlField branch.

## Synthetic controls

- `NONE`: neutral distribution
- `SLOW`: longer time shifts, lower event density, lower velocity
- `BUILDUP`: progressively shorter time shifts, higher density, higher velocity
- `WAVE`: periodic timing/density/velocity behavior

Each synthetic event slot emits a TIME, PITCH-or-REST and VELOCITY token. This is deliberately simpler than the production tokenizer; the experiment isolates whether the control mechanism can be learned end-to-end.

## Container measurements

Environment:

```text
PyTorch 2.10.0+cpu
CPU threads: 5
batch: 4
sequence: 48
training: 150 steps
optimizer: AdamW
```

With an auxiliary control-field regression term, a 150-step run produced clear control separation and a timing-wave correlation of about 0.956 for `WAVE`.

More importantly, the auxiliary control loss was then removed. With **causal language-model loss only**, forcing the learned control to `NONE` at evaluation increased held-out loss by approximately:

```text
SLOW     +0.94
BUILDUP  +0.70
WAVE     +0.77
```

The `WAVE` timing curve still had approximately 0.947 correlation with the intended periodic timing pattern. `NONE` maintained an exact maximum control-field magnitude of 0.0.

This demonstrates that the control branch receives a useful learning signal directly through next-token prediction; a dedicated control regression objective is not required for the mechanism to become behaviorally relevant in this synthetic setting.

## Previous basis-capacity experiment

A separate function-approximation experiment compared 4, 8 and 16 smooth basis functions. Eight bases represented constant, ramp, hump and one-period wave controls with MSE below 1e-3, while a four-period wave was substantially harder. Sixteen bases reduced the four-period error strongly.

Interpretation: basis count acts as a control-bandwidth limit. Orbitune should therefore use a small global field for smooth long-range behavior and retain local override events for abrupt or high-frequency control.

## What this does not prove

This is not evidence that the mechanism already works on real MIDI pretraining. The dataset is synthetic, the MIDI event grammar is simplified, and only a short sequence was trained. The next required experiment is to integrate the ControlField branch into the production Orbitune model/tokenizer and measure real generated TIME_SHIFT, event-density and velocity statistics.

## Current decision

The mechanism is strong enough to keep as an architecture candidate:

```text
Global ControlField: 8 bases
Initial channels: TIMING / DENSITY / VELOCITY
NONE: structural zero field
Local override tokens: retained for abrupt control
```

Do not freeze this into a published Base ABI until the real-MIDI experiment passes.
