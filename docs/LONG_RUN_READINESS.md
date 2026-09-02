# Compound Base long-run readiness

This note supersedes any earlier CFE report statement that the fixed-window
trainer is unconditionally ready for a multi-hour production run.

## What is ready

The RTX 3080 Context Fit Envelope remains the measured performance baseline:

- `n_head=7`, `head_dim=32`;
- `seq_len=256`;
- microbatch 144;
- BF16/`precision=auto` on the measured RTX 3080 environment;
- causal SDPA fast path enabled.

The production long-run path is now `scripts/compound_longrun_train.py`, invoked
by `run_cfe_train.ps1`. It provides:

- distinct global Python RNG and sampler RNG checkpoint state;
- CPU-normalized CUDA RNG restore;
- atomic latest/healthy/best checkpoints;
- fixed validation window plans across evaluations and resumes;
- runtime-drift checks;
- non-finite loss/gradient checks **before** optimizer mutation;
- source commit recording via `ORBITUNE_SOURCE_COMMIT`;
- synthetic-data guardrails.

`tools/cuda_smoke.py` is the CUDA acceptance test. It checks next-window resume,
continued optimizer/model state, and a generated MIDI write/read round-trip.

## What is not claimed

Random fixed-window training does not preserve history from before the sampled
window. Local, medium, global and recurrent context are all affected at a
mid-song boundary. See `docs/STATE_CARRY_AUDIT.md`.

For that reason the production launcher requires `-AllowFixedWindowTraining`.
This is an explicit acknowledgement of the current approximation, not a claim
that train and generation state semantics are identical.

## Required launch shape

```powershell
pwsh -File run_cfe_train.ps1 `
  -TrainJsonl data\real_midi\train.jsonl `
  -ValJsonl data\real_midi\val.jsonl `
  -Checkpoint runs\compound\run1.pt `
  -Steps 50000 `
  -AllowFixedWindowTraining
```

Do not start a final Base training run from synthetic/fixture data. A real-MIDI
validation gate is required before the resulting checkpoint is designated as
the final Base.

## Remaining structural follow-up

State-carry TBPTT remains a separate model-execution change. It should be
implemented and benchmarked independently, then compared against the measured
fixed-window baseline before replacing this mode.
