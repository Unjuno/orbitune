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
* `window_hash` is derived from the deterministic tile plan so two evaluations can prove they used the same windows.
* All 5 historical evaluations tile the validation corpus into **1444 deterministic windows / 369,664 Compound events / 4,435,968 scalar fields** (each Compound event is 12 scalar fields; the validation corpus has 372,369 events, so 99.27 % of validation events are covered; 2,705 tail events are dropped because only full windows are evaluated).
* The checkpoint-ranking metric used for the five saved JSON results is the model's trainer loss for each batch, weighted by the number of Compound events in that batch. The patched evaluator names this `mean_trainer_loss_event_weighted`; historical JSONs keep `mean_loss_per_event` as a compatibility alias.
* `total_scalar_fields = total_events × 12` is corpus-shape telemetry only. It is **not** a valid alternative loss denominator.
* Decoder heads do not all share the same active-event denominator. `event_type`, `channel` and `delta` use all events; `velocity` / `duration` use NOTE events; `control`, `a1` and `a2` have their own masks. The patched evaluator therefore reports `mean_on_active_events` and `active_events` per head. Historical per-head values below were produced by the earlier batch-event weighting and should be treated as a diagnostic, not as exact global active-event means.

The 16-window trainer validation is too narrow (~1.1 % of the corpus) to rank checkpoints that differ by only a few hundredths of a loss unit, so the fixed-window ship decision is based on the 1444-window evaluation.

---

## 2. Full-validation loss table

Historical table: 1444 windows / 369,664 events / 4,435,968 scalar fields each. `mean_loss` is the trainer-loss ranking metric described above.

| step | LR (since last resume) | mean_loss | event_type | a1 | channel | control | delta | duration | velocity |
|---|---|---|---|---|---|---|---|---|---|
| 1900 (best) | — (3e-4 from scratch) | **-1.534510** | 0.6063 | 1.6718 | 0.0003 | **-3.1039** | -4.5426 | **-3.3819** | -1.9916 |
| 2000 | 3e-4 | -1.512672 | 0.6064 | 1.6629 | 0.0003 | -3.0554 | -4.5388 | -3.2744 | -1.9898 |
| 2200 (A) | 3e-4 | -1.519059 | 0.6037 | 1.6478 | 0.0003 | -3.0587 | -4.5283 | -3.3150 | -1.9832 |
| 2200 (B) | 1e-4 (override) | -1.527202 | 0.6041 | 1.6520 | 0.0003 | -3.0971 | -4.5698 | -3.2831 | -1.9969 |
| 3000 | 1e-4 (override) | -1.517611 | **0.5982** | **1.6158** | **0.0002** | -3.0894 | -4.5372 | -3.2043 | **-2.0065** |

**Ranking by the historical trainer-loss metric (more negative = better):**

| rank | checkpoint | full-val loss | Δ vs best | Δ % |
|---|---|---|---|---|
| 1 | step **1900 (best)** | **-1.534510** | 0 (baseline) | 0.00 % |
| 2 | step 2200 B / 1e-4 | **-1.527202** | +0.007308 | +0.48 % |
| 3 | step 2200 A / 3e-4 | **-1.519059** | +0.015451 | +1.00 % |
| 4 | step 3000 / 1e-4 | **-1.517611** | +0.016899 | +1.10 % |
| 5 | step 2000 / 3e-4 | -1.512672 | +0.021838 | +1.42 % |

The rank is unaffected by the terminology correction because the historical total-loss values themselves are unchanged.

**Historical per-head diagnostic (legacy batch-event weighting):**
- step 1900 is best on control, duration and delta in the saved results;
- step 3000 is best on event_type, a1, channel and velocity;
- the largest 1900→3000 regression in that diagnostic is duration: **+0.1776**;
- the other legacy deltas are a1 -0.0560, velocity -0.0149, control +0.0145, event_type -0.0081, delta +0.0053 and channel -0.0001.

This supports the practical choice of step 1900, but the old per-head aggregation does **not** prove a causal mechanism. In particular, it is too strong to claim from these numbers alone that long-range context specifically causes the duration regression. The patched evaluator now records exact active-event counts so future per-head comparisons can be recomputed without this ambiguity.

---

## 3. A/B decision rule and outcome

Stage 2 A/B from 1900 → 2200, 300 steps each, on the same validation plan:
- A: 3e-4 (no override) → -1.519059
- B: 1e-4 (via `--override-resume-lr`) → -1.527202 — winner by 0.0081 (0.53 %)

Stage 3 from 1900 → 3000 at 1e-4: -1.517611. The 2200 B checkpoint remains better than the later 3000 checkpoint, and neither learning-rate continuation beats step 1900.

**Final fixed-window decision:** the best measured checkpoint under this recipe is **step 1900 (trainer-loss metric -1.534510)**. Further fixed-window training at the tested 3e-4 or 1e-4 schedules is not justified.

---

## 4. Generated MIDI samples

256 generated events requested, seed 0, T=1.0, top_p=0.9.

| checkpoint | size | parsed events | NOTE | TEMPO | PEDAL | unique pitches | range | most common pitch |
|---|---|---|---|---|---|---|---|---|
| step 0500 | 1177 B | 257 | 26 | 1 | 230 | 17 | 39..74 | 56 (11.5 %) |
| step 1000 | 1361 B | 256 | 77 | 1 | 178 | 32 | 43..81 | 74 (7.8 %) |
| step 1900 (best) | 1605 B | 257 | 137 | 1 | 119 | 28 | 44..84 | 79 (9.5 %) |
| step 2000 | 2021 B | 256 | 242 | 1 | 13 | 28 | 48..83 | 69 (8.7 %) |
| step 3000 | 1707 B | 257 | 162 | 1 | 94 | 34 | 43..81 | 67 (8.0 %) |

All five samples parse cleanly via `read_compound_midi`, contain NOTE events, use multiple event types and pitches, and show no obvious pitch/event-type collapse or stuck-note failure in the automated sanity checks.

---

## 5. Training sample exposure

For the fixed-window production shape:

`batch 144 × seq_len 256 = 36,864 sampled training events / optimizer step`.

Therefore:

- step 1900 lineage exposure = `1900 × 36,864 = 70,041,600` sampled events;
- `70,041,600 / 1,571,272 = 44.58×` the raw training-corpus event count;
- step 3000 lineage exposure = `110,592,000` sampled events = `70.38×` the raw corpus event count.

These are **sample exposure ratios, not epochs**. Random overlapping song-local windows repeatedly reuse events and contexts, so the ratio must not be interpreted as a conventional full-dataset pass count. The plateau is real under the measured fixed-window objective, but exposure alone does not establish classical overfitting.

The earlier report's `5 × 256 = 1,280` figure described generated sample output, not training exposure, and is intentionally removed from the training-volume analysis.

---

## 6. Sanity gates

| gate | status |
|---|---|
| Windows test suite before TBPTT branch | **134 / 134 pass** (Linux-only tests excluded as documented) |
| Linux CI at `f3f1cc3` | **139 passed, 2 skipped, 0 failed** |
| `tools/cuda_smoke.py` | **PASS** — BF16 training, exact sampler RNG resume, optimizer/model continuation, MIDI round-trip |
| deterministic full-validation plan | PASS |
| model weights unchanged after full-validation eval | PASS |
| NaN / Inf during fixed-window staged training | 0 through step 3000 |

---

## 7. Resume-LR override

* `scripts/compound_longrun_train.py` supports `--override-resume-lr <float>` and rejects bare use without `--resume`.
* `run_cfe_train.ps1` exposes the same behavior as `-OverrideResumeLr`.
* The override is applied after optimizer-state restore, preventing `optimizer.load_state_dict()` from silently restoring the old checkpoint learning rate.
* The applied value is recorded in runtime/log telemetry.

This mechanism was used for the 1900→2200/3000 1e-4 continuation experiments.

---

## 8. Follow-ups and corrected interpretation

1. **State-carry TBPTT is now implemented on the `state-carry-tbptt-impl` branch.** It uses the existing local / medium / global / recurrent streaming state, a song-sequential sampler, per-lane song-boundary reset, TBPTT-boundary detach, and checkpointed sampler + stream state. It does not introduce a new architecture. See `docs/TBPTT_IMPLEMENTATION.md`.
2. **event_type loss ≈ 0.6 is not evidence of a weak event-type signal by itself.** The decoder event-type head has 10 classes, so uniform 10-way cross entropy is `ln(10) ≈ 2.303` nats. Even if a particular corpus effectively exercises fewer classes, 0.6 is below the appropriate uniform reference, not above a `0.366` ceiling. The previous statement was mathematically incorrect and is withdrawn.
3. **Per-head full-validation values require active-event denominators.** The patched evaluator computes those counts explicitly. Historical saved JSON totals remain valid for the step-1900 ranking, while exact per-head global means should be regenerated when doing the TBPTT A/B.
4. **`pynvml` FutureWarning** on CUDA init is benign and cosmetic.

---

## 9. Files and tooling

Fixed-window / evaluation tooling:

* `tools/full_validation_eval.py` — deterministic full-window evaluator; now separates event/scalar coverage, historical trainer-loss ranking, and exact per-head active-event means.
* `tools/parse_stage3.py`
* `tools/print_full_val_table.py`
* `tools/generate_one_sample.py`
* `tools/check_per_head_delta.py`
* `tests/test_full_validation_and_resume_lr.py`
* `scripts/compound_longrun_train.py`
* `run_cfe_train.ps1`
* `docs/compound_results/full_val_*.json`
* `docs/compound_results/sample-step-3000.mid`
* `docs/compound_results/README.md`

State-carry TBPTT implementation:

* `orbitune/compound_tbptt.py` — differentiable streaming step, chunk encoder, detach helpers, sequential song sampler and stream-state serialization.
* `scripts/compound_tbptt_train.py` — experimental state-carry production trainer with explicit fixed→TBPTT LR transition and checkpointed carried state.
* `tests/test_compound_tbptt.py` — generation-equivalence, arbitrary chunk partition, per-lane reset, gradient/detach, sampler resume and state serialization gates.
* `docs/TBPTT_IMPLEMENTATION.md` — execution semantics and required RTX 3080 acceptance experiment.

Large MAESTRO raw/prepared data remains gitignored and regenerable.

---

## 10. Recommended next move

Do **not** continue fixed-window training. The next experiment is a short RTX 3080 TBPTT smoke/profile from the frozen step-1900 checkpoint.

Required sequence:

1. run 5–20 TBPTT optimizer steps from step 1900;
2. verify finite loss/gradients, per-lane song state carry, checkpoint save and exact resume;
3. profile a small `(batch, seq_len)` grid because the fixed-window `144 × 256` CFE does not transfer to the event-wise streaming execution path;
4. run a short real-MIDI TBPTT continuation and compare it against frozen step 1900 using the same streaming-state validation protocol;
5. only then decide whether a longer TBPTT continuation is justified.

---

## 11. Project-level status

* **`FIXED_WINDOW_BASE = COMPLETE`.**
* **`SHIP_CHECKPOINT = runs/compound/base-maestro2004.best.pt` (step 1900).** Historical full-validation trainer-loss metric = **-1.534510**.
* **`FURTHER_FIXED_WINDOW_TRAINING = STOP`.**
* **`STATE_CARRY_TBPTT = VERIFIED_OK`.** Real-hardware verification complete on RTX 3080. See `docs/TBPTT_REPORT.md` for the full A/B and per-head deltas. 5-song streaming-state val of the SHIP_CHECKPOINT = **-1.187442**; LR=3e-5 TBPTT continuation (50 steps) = **-1.164991** (within noise, +0.022); LR=1e-4 TBPTT continuation (50 steps) = **-0.888059** (catastrophic, +0.299). **Safe LR for TBPTT fine-tuning is ≤ 3e-5.**
* **`TIME_VECTORIZED_TBPTT = VERIFIED`.** Commit `b5f161a` advances the entire `seq_len` slab through the Transformer in one Python call. Steady-state throughput on RTX 3080 (BF16, batch=4, seq=64) is **≈ 665–700 ev/s** (peak 737 ev/s) — **16.5–19.7× the legacy 35 ev/s** and **4.45–5.29× the lane-batched 130 ev/s**. See `docs/TBPTT_REPORT.md` §16.1.
* **`TBPTT_500_STEP_PILOT_AT_LR_3E_5 = PASS`.** Source step 1900 → final step 2400 at LR=3e-5, BF16, batch=4, seq=64. 5-song streaming val trajectory: `VAL_BASE = -1.187442` (frozen), `VAL_STEP_2000 = -1.143810` (Δ +0.0436), `VAL_STEP_2150 = -1.071313` (Δ +0.1161, transient), `VAL_STEP_2400 = -1.212744` (Δ **-0.0253**, better than base). Canonical re-run via `tools/tbptt_validation_eval.py`: `VAL_STEP_2400 = -1.206352` (Δ **-0.018910**). **|Δ| < 0.05 hard-stop satisfied; canonical Δ is negative (improvement).** No NaN/Inf/OOM/`safe_backward` failure across 500 steps. State carry validated at step 2400 (4 lanes, `steps` = 8256 / 512 / 13376 / 19328, all histories non-empty, `memory = [(1, 224)] × 3`). Pilot ckpt at `runs/compound/tbptt/pilot-lr3e5.pt` (step 2400, `events_seen = 70,169,600`, `source_commit = c445ea7`).
* **`NEXT_EXPERIMENT = COMMERCIAL_BASE_PRODUCTION_PRETRAIN` (gated).** Next engineering target is the production commercial base pretrain trainer: epoch-aware no-replacement TBPTT sampler + per-event loss weighting + commercial_v1 corpus build + `power_draw_watts` mW→W bug fix. See `docs/TBPTT_REPORT.md` §17. **Do not start a 50M-event long run** until (1) epoch sampler + 6 unit tests are green, (2) commercial_v1 corpus build census is recorded (1.0× event total measured, not estimated), (3) `power_draw_watts` bug is fixed with a unit test, and (4) full pytest regression is clean.
