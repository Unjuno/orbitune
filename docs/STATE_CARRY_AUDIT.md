# Training / Generation State Semantics — Compound Base

## TL;DR

`CompoundHierarchicalGPT.encode()` (training) and
`CompoundHierarchicalGPT.advance_stream()` (generation) are **not**
mathematically equivalent paths through the model. They share the same
submodules but differ in one important way:

| Sub-path | Training `encode()` | Generation `advance_stream()` |
|----------|----------------------|--------------------------------|
| Local attention (window=64) | identical | identical |
| Medium pooling + attention | identical | identical |
| Global pooling + attention | identical | identical |
| **Recurrent memory** | **reset per 256-event window** | **carried event-by-event** |
| Fusion + decoder | identical (teacher-forced) | identical (autoregressive) |

Because the recurrent memory is **reset** at the start of every training
window, the recurrent context the model sees during training is an
**approximation** of what it will see during generation. The local /
medium / global attention paths are causal and properly bounded, so they
are unaffected. The recurrent path is the only divergence.

## What this means in practice

- For windows 1 and 2 (positions 0..511 of any song), training memory
  runs once over the 256 events of each window independently. The model
  never learns to integrate memory across windows.
- At generation time, memory has been integrated since the start of the
  primer. After 256+ events, the model's recurrent context is more
  informative than anything it saw during training.
- This is **not** necessarily a quality bug — most of the model's signal
  comes from local / medium / global attention, which is properly
  causal in both paths. But it does mean that very long generation
  traces enter a region of the recurrent state space the model was never
  trained on.

## Why we did not "fix" this in the long-run audit

Implementing full state-carry training (TBPTT across song boundaries)
would require:

1. Refactoring `TensorSampler` to emit variable-length sequences with
   boundary tags instead of fixed-length windows.
2. Refactoring `encode()` to accept an initial state and `detach()`
   state gradients at chunk boundaries.
3. Refactoring `advance_stream` to be testable in both training and
   inference modes.
4. Re-deriving the equivalence of the model's loss formulation when the
   recurrent memory is detached at chunk boundaries.
5. Re-running the full CFE bench to verify the new training loop does
   not regress on the same head geometry / batch envelope.

Per the user's instructions ("do not start a new architecture study"),
this is **out of scope** for the production-readiness audit. It is
explicitly listed below as the **LONG-RUN BLOCKER FOR STATE-CARRY
TRAINING** that must be resolved before claiming "training equivalent
to generation" semantics.

## Equivalence test (window 1 only)

The training-vs-generation equivalence holds **exactly** for window 1
(positions 0..255) of any song, because both paths start with
`memory.state = None`. See `tests/test_compound_state_semantics.py` for
the test that pins down the matching context vector and per-position
logit within numerical tolerance for a 256-event prefix.

This test guards against accidentally breaking window-1 equivalence
during future refactors.

## Decision

For the **first** multi-hour production training run we will keep the
existing window-based training. This is a deliberate choice, not an
oversight:

- The CFE bench and the n_head=7 quality gate both showed the model
  trains and generalises sensibly with window-based training.
- Implementing state-carry TBPTT in the same change window risks
  confounding the CFE measurement, the validation comparison and the
  long-run telemetry. The user explicitly requested this be deferred.
- The equivalence test on window 1 ensures the next refactor that does
  tackle state-carry training can detect regressions.

**LONG-RUN BLOCKER**: implement and verify state-carry TBPTT training
**before** claiming "production model semantically matches generation
for arbitrary song prefixes". Filed as a follow-up; not blocking the
current fixed-window production training run.