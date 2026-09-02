# Training / Generation State Semantics — Compound Base

## TL;DR

The current production-compatible training mode samples random fixed-length
windows from inside each song. Generation, by contrast, starts from a primer
and carries history forward continuously.

Therefore **all pre-window history is absent during fixed-window training**,
not only the recurrent-memory state.

| Context path | Random fixed-window training | Streaming generation |
|---|---|---|
| Local attention | history before sampled window is absent | prior local records carried |
| Medium summaries | summaries before sampled window are absent | prior summaries carried |
| Global summaries | summaries before sampled window are absent | prior summaries carried |
| Recurrent memory | reset at sampled window start | fast/medium/slow state carried |
| Decoder | teacher-forced inside window | autoregressive |

The model remains causal inside each sampled window, but a window beginning at
song event 1000 does **not** receive the context produced by events 0..999.

## Recovery inside a fixed window

The four paths have different recovery horizons.

- Local attention can recover its full bounded local context after roughly
  `local_window` events have elapsed inside the sampled window.
- Medium summaries are constructed only from local states observed inside the
  sampled window. No completed medium groups from before the boundary exist.
- Global summaries likewise start from an empty hierarchy at the boundary.
- Recurrent memory starts from zero/None and cannot reproduce the carried
  fast/medium/slow state that streaming generation would have at that offset.

Consequently the previous statement that "only recurrent memory diverges" was
incorrect and has been withdrawn.

## What the equivalence test actually proves

`encode()` and `advance_stream()` can be compared when both start from an empty
state at the **beginning of a song/prefix**. The existing first-window test is
useful for guarding that common-start behavior.

It does **not** prove that a random training window cut from the middle of a
song is equivalent to streaming generation at the same absolute song offset.
That stronger claim would require carrying the complete hierarchical state
across chunk boundaries.

## State required for strict chunk carry

A future state-carry TBPTT path must preserve, per batch lane:

1. local record/history needed by the local attention window;
2. the partial medium buffer plus completed medium summaries;
3. the partial global buffer plus completed global summaries;
4. recurrent fast / medium / slow memory tensors;
5. song identity and chunk position so only lanes that cross a song boundary
   are reset.

At a TBPTT boundary the carried tensors must be detached from the previous
chunk's graph while retaining their values.

## Why state-carry TBPTT is not silently enabled here

Adding correct batched state carry changes the training execution semantics and
may change the measured RTX 3080 Context Fit Envelope. It should not be mixed
into the operational resume/health fixes without its own equivalence and
throughput validation.

The measured CFE result therefore remains valid for the mode it actually
benchmarked:

- `n_head=7`, `head_dim=32`;
- BF16 on the measured RTX 3080 environment;
- `seq_len=256`;
- microbatch 144;
- random fixed-window training.

## Production safety decision

`scripts/compound_longrun_train.py` and `run_cfe_train.ps1` require an explicit
fixed-window acknowledgement before they will train. This prevents a multi-hour
run from being mistaken for strict train/generation state equivalence.

For the current mode:

- checkpoint/resume, RNG state, validation windows, health checks and MIDI
  output can be made production-safe;
- long-range state equivalence is **not claimed**;
- real-MIDI quality validation is still required before treating a trained
  checkpoint as the final Base.

## Follow-up acceptance gate for state-carry TBPTT

Before claiming strict streaming-equivalent training, require all of:

1. song boundary reset and same-song chunk carry;
2. per-lane local/medium/global/recurrent state preservation;
3. detach at TBPTT boundaries;
4. one-shot vs arbitrary-chunk context/logit equivalence in eval mode;
5. checkpoint/resume of sampler position and carried state;
6. RTX 3080 throughput/VRAM remeasurement;
7. real-MIDI validation A/B against the fixed-window baseline.

Until then, fixed-window training is an explicit, documented approximation —
not an equivalent streaming training path.
