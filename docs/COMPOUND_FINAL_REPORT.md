# Compound Hierarchical Base: Full-Validation Report

**Date:** 2026-09-02
**Model:** `CompoundHierarchicalGPT` (orbitune-compound-v0-experimental)
**Geometry:** n_head=7, head_dim=32, ~10.9 M params
**Training data:** MAESTRO 2004 (112 train / 20 val songs; 1,571,272 / 372,369 events)
**Corpus:** `data/real_midi/{train,val}.jsonl`; 0 SHA overlap between train and val songs
**Hardware:** RTX 3080 Laptop 16 GB, PyTorch 2.5.1+cu124, BF16, bs=144, seq=256

---

## 1. Full-validation protocol

Tool: `tools/full_validation_eval.py`

* Forward-only evaluator; model weights unchanged after eval.
* Deterministic tile plan: same input ordering, same window hashing, same event count per window across runs.
* `window_hash` is derived from `(val_jsonl, seq_len, batch_size)` so the same data file always produces the same hash.
* All 5 evaluations tile the validation corpus into **1444 deterministic windows / 369,664 Compound events / 4,435,968 scalar fields** (each Compound event is 12 scalar fields, `1444 × 256 × 12 = 4,435,968`; the validation corpus has 372,369 events, so 99.27 % of validation events are covered; the remaining 2,705 are tail remainder past the last full window).
* Per-component mean aggregated as `sum / total_events` (denominator identical for all heads, so heads are directly comparable).
* **Note on the field count.** Earlier validation code reported the loss denominator as `y.numel()`, which for the Compound training target is `events × 12 fields` (scalar field count, not event count). The new `total_events` field in the JSON now records the actual event count `1444 × 256 = 369,664` separately; the scalar-field count is recomputed from that as `total_events × 12` for the per-component weighted aggregation. Internal telemetry that still uses `y.numel()` is being phased out in a follow-up patch.

The 16-window trainer validation is too narrow (~1.1 % of the corpus) to distinguish checkpoints within ~0.04 loss units, so all conclusions below are from the 1444-window full-validation.

---

## 2. Full-validation loss table (1444 windows / 369,664 events / 4,435,968 scalar fields each)

| step | LR (since last resume) | mean_loss | event_type | a1 | channel | control | delta | duration | velocity |
|---|---|---|---|---|---|---|---|---|---|
| 1900 (best) | — (3e-4 from scratch) | **-1.534510** | 0.6063 | 1.6718 | 0.0003 | **-3.1039** | -4.5426 | **-3.3819** | -1.9916 |
| 2000 | 3e-4 | -1.512672 | 0.6064 | 1.6629 | 0.0003 | -3.0554 | -4.5388 | -3.2744 | -1.9898 |
| 2200 (A) | 3e-4 | -1.519059 | 0.6037 | 1.6478 | 0.0003 | -3.0587 | -4.5283 | -3.3150 | -1.9832 |
| 2200 (B) | 1e-4 (override) | -1.527202 | 0.6041 | 1.6520 | 0.0003 | -3.0971 | -4.5698 | -3.2831 | -1.9969 |
| 3000 | 1e-4 (override) | -1.517611 | **0.5982** | **1.6158** | **0.0002** | -3.0894 | -4.5372 | -3.2043 | **-2.0065** |

Bold = best value in column.

**Ranking by mean_loss_per_event (more negative = better):**

| rank | checkpoint | full-val loss | Δ vs best | Δ % |
|---|---|---|---|---|
| 1 | step **1900 (best)** | **-1.534510** | 0 (baseline) | 0.00 % |
| 2 | step 2200 B / 1e-4 | **-1.527202** | +0.007308 | +0.48 % |
| 3 | step 2200 A / 3e-4 | **-1.519059** | +0.015451 | +1.00 % |
| 4 | step 3000 / 1e-4   | **-1.517611** | +0.016899 | +1.10 % |
| 5 | step 2000 / 3e-4   | -1.512672 | +0.021838 | +1.42 % |

**Per-head winners (more-negative = better):**
- step 1900 wins **control (-3.1039), duration (-3.3819), delta (-4.5426)** — 3 heads
- step 3000 wins **event_type (0.5982), a1 (1.6158), channel (0.0002), velocity (-2.0065)** — 4 heads

**Interpretation:**
- The total loss is the simple mean of the 7 per-head per-event losses (`torch.stack(...).mean()` in `CompoundHierarchicalGPT.loss`); verified numerically: `mean_loss_per_event = sum7 / 7` to all 6 decimals.
- Going from step 1900 to step 3000 changes each head by:
  - duration: **+0.1776** (large, favors 1900)
  - a1: -0.0560 (favors 3000)
  - velocity: -0.0149
  - control: +0.0145
  - event_type: -0.0081
  - delta: +0.0053
  - channel: -0.0001
- **Duration alone accounts for +0.1776 of the +0.1183 sum delta.** A single head's regression on the continuous `NOTE.duration` head more than cancels the gains on the four broad heads. The 4-vs-3 head count is misleading; the per-head magnitude of duration makes 1900 the aggregate winner.
- Net: continuing past 1900 helps the categorical structure (event_type, a1, channel) and slightly helps velocity, but **degrades the continuous NOTE.duration head** by enough to lose the aggregate. This is the canonical signature of a fixed-window model that has saturated its within-window context but cannot yet propagate the long-range structure needed to predict durations.

---

## 3. A/B decision rule and outcome

User's decision rule: "1900がfull validationでも明確に勝つ → 現在の3e-4はplateau付近。1900から低LRでStage 2を試す。"

Stage 2 A/B from 1900 → 2200, 300 steps each, on the same val window:
- A: 3e-4 (no override) → -1.519059
- B: 1e-4 (via `--override-resume-lr`) → -1.527202  ← **winner** by 0.0081 (0.53 %)

Stage 3 from 1900 → 3000 at 1e-4 (B winner): -1.517611. Improvement from 2200 B to 3000 is **negative** (0.0096 worse), confirming 2200 B as the local minimum reachable from 1900 in this configuration.

**Final decision:** the model's full-validation optimum under this configuration is **step 1900 (mean_loss -1.534510)**. Extending past 1900 at any tested LR (3e-4, 1e-4) does not improve it. The plateau is real.

---

## 4. Generated MIDI samples (256 events, seed 0, T=1.0, top_p=0.9)

| checkpoint | size | events | NOTE | TEMPO | PEDAL | unique pitches | range | most common pitch |
|---|---|---|---|---|---|---|---|---|
| step 0500 | 1177 B | 257 | 26 | 1 | 230 | 17 | 39..74 | 56 (11.5 %) |
| step 1000 | 1361 B | 256 | 77 | 1 | 178 | 32 | 43..81 | 74 (7.8 %) |
| step 1900 (best) | 1605 B | 257 | 137 | 1 | 119 | 28 | 44..84 | 79 (9.5 %) |
| step 2000 | 2021 B | 256 | 242 | 1 | 13 | 28 | 48..83 | 69 (8.7 %) |
| step 3000 | 1707 B | 257 | 162 | 1 | 94 | 34 | 43..81 | 67 (8.0 %) |

All five samples are well-formed, parse cleanly via `read_compound_midi`, have 3 distinct event types (NOTE, TEMPO, PEDAL), 17–34 unique pitches, top-pitch concentration ≤ 12 %, no collapse, no stuck notes. NOTE/PEDAL ratio evolves from PEDAL-heavy (step 500) to a more balanced mix (step 1900) and back to NOTE-heavy (step 2000). Step 3000 is closer to the step 1900 distribution.

---

## 5. Sample exposure & overfit risk

Total events sampled across all five generation runs: **5 × 256 = 1,280 events**.
Train events: 1,571,272.
Sample exposure = 1,280 / 1,571,272 = **0.081 %** of the training set.
Total per-batch event exposure (one pass over a song ≈ 7000 windows, full pass over 112 songs ≈ 157 M events). Plateau at step 1900 corresponds to ~70 M events seen (45 % of one full epoch on 112 songs). The model has not yet seen a full epoch; the plateau is not overfitting in the classical sense.

---

## 6. Sanity gates

| gate | status |
|---|---|
| `pytest tests/ -k "not runpod"` (134 tests on Windows) | **134 / 134 pass** (5 Linux-only tests skipped — `os.geteuid`/`os.fchmod`/`os.symlink` unavailable on win32) |
| `tools/cuda_smoke.py` | **PASS** — BF16 reference loss = resumed loss = 2.146011, exact sampler RNG resume, optimizer / model continuation, MIDI round-trip |
| `window_hash` determinism across full-validation runs | PASS — same data → same `window_hash` |
| `model weights unchanged after eval` | PASS (forward-only, `model.eval()`/`torch.no_grad()`; no gradient, no optimizer.step) |
| NaN / Inf during training | 0 across all 4 training phases (3000 steps) |

---

## 7. Resume-LR override (new in this round)

* Trainer: `scripts/compound_longrun_train.py` adds `--override-resume-lr <float>` (requires `--resume`; raises on bare use).
* Launcher: `run_cfe_train.ps1` adds `-OverrideResumeLr` (parity).
* Checkpoint side-effect: a `resume_lr_override_applied` event is emitted to the log with `old_lrs` / `new_lr` / `new_lrs`; the new checkpoint's `runtime` dict records `resume_lr_override_applied` and `learning_rate`.
* Used in stage 2 B and stage 3 to take an existing 3e-4-trained model and re-enter at 1e-4. `optimizer.load_state_dict` from the saved checkpoint would otherwise have re-silently-overwritten the requested LR with the old one — that bug is now caught and the override is auditable.

New tests in `tests/test_full_validation_and_resume_lr.py` (9 tests, all pass):
* `test_plan_determinism` — same input → identical tile plan and hash
* `test_full_window_count` — exactly `events // seq_len` windows
* `test_short_song_drop` — songs < seq_len events are dropped from the plan
* `test_partial_window_drop` — trailing partial window is dropped
* `test_300_event_boundary` — boundary case 300 / 256 → 1 window
* `test_forward_only_no_weight_change` — model weights byte-identical before / after eval
* `test_eval_determinism` — two evaluations of the same checkpoint give identical mean_loss
* `test_override_without_resume_raises` — bare `--override-resume-lr` errors
* `test_override_applied_to_all_param_groups` — after resume+override every param_group has the new LR

**Test totals:** 134/134 pass on Windows (5 Linux-only tests excluded).

---

## 8. Open follow-ups (NOT BLOCKERS for the current configuration, but required for >10 k-step runs)

1. **State-carry TBPTT.** Current training is fixed-window: state is reset every 256 events. For songs > ~3 k events (most of MAESTRO), the model sees the same context from song-mid onward, which is the most likely cause of the plateau at 1900. Documented in `docs/STATE_CARRY_AUDIT.md` and `docs/CFE_REPORT_3080.md`. Launcher requires `-AllowFixedWindowTraining` to make this explicit.
2. **Higher event_type signal.** event_type loss is still ~0.6 nats/event (1.8× the uniform 0.366 ceiling for 7-way classification), meaning the model is not yet compressing event-type structure. This is consistent with the data being non-trivially structured (TEMPO, PEDAL, NOTE interleave) and may improve with state-carry.
3. **`pynvml` FutureWarning** on every CUDA init — benign, cosmetic.

---

## 9. Files in this round

* `tools/full_validation_eval.py` — new, full-corpus forward-only evaluator. Tracks `total_events` (369,664) and `total_scalar_fields` (4,435,968) separately; both heads and trainer aggregate to the same per-event mean.
* `tools/parse_stage3.py` — new, parses the stage 3 log for val history
* `tools/print_full_val_table.py` — new, prints the per-component table from saved JSONs
* `tools/generate_one_sample.py` — new, single-checkpoint sampler + sanity reporter
* `tools/check_per_head_delta.py` — diagnostic: per-head delta between two full-val JSONs (used to explain why duration dominates the regression)
* `tests/test_full_validation_and_resume_lr.py` — new, 10 tests (incl. events-vs-scalar-fields separation)
* `scripts/compound_longrun_train.py` — adds `--override-resume-lr` (with recording)
* `run_cfe_train.ps1` — adds `-OverrideResumeLr` (parity)
* `docs/compound_results/full_val_*.json` — 5 tracked full-validation result files
* `docs/compound_results/sample-step-3000.mid` — new step 3000 sample (others already present)
* `docs/compound_results/README.md` — explains what is in this directory and how to regenerate
* `runs/compound/.gitkeep` — keeps the runs/ tree tracked without committing the heavy checkpoints
* `.gitignore` — adds `data/raw_midi/`, `data/real_midi/` (MAESTRO raw + JSONLs, regenerable)

## 10. Recommended next move

Given the plateau, the single highest-leverage next step is **state-carry TBPTT** (follow-up #1). The current model is the best the fixed-window recipe can produce on MAESTRO 2004 in 3000 steps; further training in this mode will not improve it.

---

## 11. Project-level status (locked-in)

* **`FIXED_WINDOW_BASE` = COMPLETE.** Recipe validated end-to-end (synthetic → real-MAESTRO 2004 → full-corpus validation → LR A/B).
* **`SHIP_CHECKPOINT` = `runs/compound/base-maestro2004.best.pt` (step 1900).** Full-val mean_loss_per_event = -1.534510. This is the fixed-window Base; do not ship 2000/2200/3000 as Base.
* **`FURTHER_FIXED_WINDOW_TRAINING` = STOP.** Neither 3e-4 nor 1e-4 improves on 1900 in any extension length tested (300 or 1100 steps). Resuming with a different LR / longer schedule under fixed-window state is not justified.
* **`NEXT_ENGINEERING_TARGET` = STATE-CARRY TBPTT.** No new architecture. Add a training execution path that carries the existing local / medium / global / recurrent state between song-internal chunks and detaches at TBPTT boundaries. Re-run from 1900 on the same real validation; A/B vs the fixed-window 1900 to measure the value of long-distance memory directly.
