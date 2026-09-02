# Compound state-carry TBPTT implementation

Status: **implemented, correctness-gated, awaiting RTX 3080 throughput/quality A/B**.

This path does not introduce a new model architecture. It reuses the existing
`CompoundHierarchicalGPT` modules and makes the generation-time state semantics
trainable across song-internal chunk boundaries.

## What carries between chunks

Each batch lane owns an independent `StreamState` containing:

- local raw-record history;
- partial medium buffer and bounded medium-summary history;
- partial global buffer and bounded global-summary history;
- recurrent fast / medium / slow memory tensors;
- stream step count.

At a song boundary only that lane is reset. At a TBPTT boundary the values are
preserved but every differentiable tensor in the carried state is detached from
the previous autograd graph.

`orbitune/compound_tbptt.py::_advance_stream_grad` is intentionally a
differentiable counterpart of `CompoundHierarchicalGPT.advance_stream`.
Evaluation tests require the two paths to produce the same context values.

## Sequential sampler

`SequentialSongChunkSampler` never samples a random mid-song window. A lane
starts at song offset zero and advances monotonically by `seq_len` records.
When fewer than `seq_len + 1` records remain, the tail is dropped, a new song is
selected, and that lane is marked in `reset_mask`.

The sampler lane positions and the sampler RNG are checkpointed. The carried
stream states are checkpointed separately as CPU tensors. A resumed TBPTT run
therefore has enough information to select the same next chunks and reconstruct
the same pre-chunk context.

## Training entry point

```powershell
python scripts/compound_tbptt_train.py `
  --train-jsonl data/real_midi/train.jsonl `
  --validation-jsonl data/real_midi/val.jsonl `
  --resume runs/compound/base-maestro2004.best.pt `
  --checkpoint runs/compound/tbptt-smoke.pt `
  --steps 1920 `
  --batch-size 2 `
  --seq-len 32 `
  --override-resume-lr 1e-4
```

The first transition from a fixed-window checkpoint requires an explicit
`--override-resume-lr`. This prevents the old optimizer state from silently
choosing the learning rate for a semantically different training regime.
Fixed-window validation history / best-loss state is reset at the transition,
because fixed-window and streaming-state validation are not the same metric.

The initial defaults (`batch_size=2`, `seq_len=32`) are conservative correctness
settings, **not an RTX 3080 CFE optimum**. The differentiable path mirrors the
existing streaming implementation and therefore performs event-wise recurrent
and hierarchy updates. It must be profiled before scaling.

## Acceptance tests

`tests/test_compound_tbptt.py` gates:

1. differentiable TBPTT contexts equal `advance_stream()` contexts in eval;
2. one-shot vs arbitrary chunk partition gives the same values;
3. per-lane reset masks reset only the selected song lane;
4. backward produces finite gradients;
5. TBPTT boundary detach removes the previous autograd graph;
6. the sequential sampler never crosses song boundaries;
7. sampler lane state + RNG round-trip selects the same next chunk;
8. carried stream state survives CPU checkpoint serialization;
9. the TBPTT trainer CLI imports on CPU-only CI.

## What this changes in project status

The fixed-window step-1900 checkpoint remains the shipped baseline. The new
TBPTT path is an experimental continuation path until it passes real-GPU smoke,
exact resume, throughput/VRAM profiling, and real-MIDI validation A/B.

Project status after merge should therefore be read as:

- `FIXED_WINDOW_BASE = COMPLETE`
- `SHIP_CHECKPOINT = step 1900 / base-maestro2004.best.pt`
- `FURTHER_FIXED_WINDOW_TRAINING = STOP`
- `STATE_CARRY_TBPTT = IMPLEMENTED_AWAITING_GPU_AB`
- `NEXT_EXPERIMENT = TBPTT_SMOKE_PROFILE_AND_AB`

## Required next experiment

Do not launch a multi-hour TBPTT run first. On the RTX 3080 Laptop 16 GB:

1. run 5-20 training steps from the step-1900 checkpoint;
2. verify finite loss/gradients and checkpoint/resume continuity;
3. measure events/sec and peak VRAM for a small grid such as
   `(batch, seq) = (1,32), (2,32), (2,64), (4,32)`;
4. choose the fastest stable shape with headroom;
5. run a short continuation A/B against frozen step 1900 using a
   streaming-state validation metric;
6. only then decide whether longer TBPTT training is justified.

The existing fixed-window CFE result (`batch=144`, `seq_len=256`) must not be
assumed to transfer to this execution path.
