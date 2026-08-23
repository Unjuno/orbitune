# Orbitune Development Handoff

Last updated: 2026-08-23 after full repository audit.

For a fresh session, read in this order:

1. `docs/AUDIT_2026-08-23.md`
2. this file
3. `docs/DESIGN_STATUS.md`
4. `docs/POST_TRAINING_RESEARCH.md`

The audit is the authoritative list of known risks. Do not jump directly into model code without checking its P0 blockers.

## 1. Product goal

Orbitune is a lightweight local-first symbolic MIDI generation framework with:

- a compact shared pretrained Base;
- distributable LoRA adapters tied to immutable Base checkpoint hashes;
- local/browser-oriented inference;
- MIDI/event modeling rather than raw audio generation.

The production direction is **Hybrid Compound Events**, not the legacy flat Theory-REMI representation.

## 2. Architecture decisions already made

Do not reopen these without contradictory real-data evidence:

- causal decoder-only Transformer;
- approximately 10M parameters as the current reference point, pending real-MIDI 5M/10M/20M sweep;
- one musical event = one Transformer step;
- Compound event types currently cover NOTE, CC, PROGRAM, BANK, TEMPO, PEDAL, PITCH_BEND, CHANNEL_PRESSURE, POLY_PRESSURE, TIME_SIGNATURE;
- deterministic MIDI canonicalization: merge same-onset/channel/pitch duplicates and truncate an active same-channel/pitch NOTE on retrigger;
- same-step state ordering: TIME_SIGNATURE → TEMPO → BANK → PROGRAM → controls → NOTE;
- candidate timing grid: 96 steps per quarter note;
- candidate in-range DELTA/DURATION encoding: 7 coarse ranges + 16 residual levels;
- continuous attributes use factorized coarse + residual representation rather than large flat vocabularies;
- NOTE intra-event decoding direction: lightweight autoregressive MLP cascade;
- long generation: recent dense sliding window plus deterministic historical anchors, with optional randomized historical samples;
- LoRA remains a runtime adapter over an immutable Base;
- deployment research: packed ternary candidate, INT8 fallback, FP16 fallback;
- post-training/policy learning is not mandatory by default.

Rejected/downranked directions include mandatory 3M Base, pure flat production representation, independent Compound heads, GRU/mini-attention NOTE decoders, recent-only infinite context, and mandatory DPO/RL.

## 3. Legacy versus production-candidate code

### Legacy/reference path

The existing `theory-remi-v0` / ~10.2M `OrbituneGPT` path remains operational for reference, CI, Web export and current Adapter infrastructure. It is **not the frozen production ABI**.

Current reference shape: 4 layers / hidden 448 / 7 heads / context 1024.

The historical 4x240 ~3M config is retained only as a legacy experiment configuration.

Do not silently mutate the legacy ABI while developing Compound support.

### Experimental Compound path

Implemented:

- `orbitune/compound.py`
  - `orbitune-compound-v0-experimental` event primitives;
  - deterministic note canonicalization;
  - semantic same-step event ordering;
  - 96/qn timing constants;
  - current 7+16 timing quantization.
- `orbitune/compound_midi.py`
  - parallel MIDI type-0/1 parser preserving the current Compound event scope;
  - intentionally separate from legacy `orbitune.midi.read_midi`.
- `orbitune/quantization.py`
  - factorized continuous-value quantization helpers.
- `orbitune/tokenizer/compound_event.py`
  - one-event-per-record representation;
  - factorized DELTA/DURATION and continuous attributes;
  - single source of truth for `COMPOUND_RECORD_WIDTH`.
- `orbitune/compound_dataset.py`
  - song-preserving Compound JSONL preparation;
  - tokenizer ABI + record width embedded in every row;
  - SHA-256 grouped train/validation split to prevent exact duplicate leakage.
- `orbitune/compound_training.py`
  - ABI/record-layout validation;
  - fixed-length training windows without crossing song boundaries.
- `scripts/prepare_compound_corpus.py`
  - experimental real-corpus preparation entry point.
- Compound tests exist under `tests/` and are included by the normal `test.yml` pytest job.

The Compound ABI is still experimental and must not be used as an immutable contributed Base target yet.

## 4. P0 blockers discovered by the audit

Close these before freezing field/output ABI. Full details are in the audit document.

### P0-A — time values above 1536 steps

Current `quantize_time()` clips silently above 1536 steps (four 4/4 bars at 96/qn). This can shorten long rests and long notes and shift all following decoded event times.

Run the long-time representation experiment before official training. Candidate families:

- extended geometric coarse boundaries;
- explicit long-time/skip representation.

No official tokenizer may silently truncate long timing.

### P0-B — BOS/EOS/start semantics

Define composition start, unconditional generation, optional stopping and continuation semantics. Compound currently has no explicit BOS/EOS event.

### P0-C — explicit field schema/cardinalities/masks

Create one source of truth for every event type's active fields, categorical ranges, loss masks, inference masks and intra-event conditioning order.

Only after this should the Compound model output heads be frozen.

### P0-D — production deduplication

Current split blocks exact-byte duplicates only. Production evaluation needs near-duplicate/composition-aware grouping.

## 5. Immediate implementation sequence

The next critical task remains the **Compound Base model**, but the P0 field/start/time decisions must be resolved or isolated before output ABI freeze.

Recommended sequence:

1. run long-time representation experiment and remove silent timing truncation;
2. define BOS/EOS/start semantics;
3. define explicit field cardinalities/masks from one module;
4. implement Compound input embeddings and a small configurable causal Transformer;
5. implement event-type-conditioned lightweight autoregressive attribute heads;
6. implement masked losses so unused attributes do not contribute;
7. run synthetic one-batch forward/backward + deterministic overfit test;
8. run tiny real-MIDI overfit test;
9. parameterize 5M/10M/20M from the same model path;
10. only then connect full checkpoint/continuous/export infrastructure.

Invariant: the Transformer advances once per musical event. Attribute autoregression is intra-event and must not inflate Transformer sequence length.

## 6. Other semantic experiments still required

- external real-MIDI 96/qn + 7+16 validation;
- CC64 half-pedal distribution: binary versus factorized continuous PEDAL;
- canonical BANK state versus exact CC0/CC32 traffic;
- integer-BPM tempo precision;
- dangling NOTE-ON repair/rejection policy;
- prevalence of MIDI Port / unsupported meta/SysEx semantics;
- production corpus provenance/license/quality gates;
- near-duplicate/composition-aware split grouping.

## 7. Base scale / long memory / control / runtime

After the model and corpus path work:

- train comparable 5M/10M/20M Compound models on real MIDI;
- compare quality / MB / latency / parameter;
- validate recent-only sliding versus anchored/dilated memory on trained rollouts;
- validate ControlField quantization/adherence on real corpus;
- define a new Compound Web/export ABI;
- benchmark INT8 versus a real packed-ternary kernel on target devices.

PyTorch STE ternary timings are not production runtime evidence.

## 8. Post-training / policy-learning policy

Status: **OPEN / NOT REQUIRED BY DEFAULT**.

Required order:

```text
Base pretraining
→ held-out rollout evaluation
→ compare Base with high-quality SFT
→ if a meaningful selection gap remains, test DPO
→ reward-based RL only after reward validity and anti-collapse checks
```

The branch may terminate with `NOT REQUIRED`.

Keep these evaluation dimensions separate:

- continuation fit;
- standalone musical quality;
- diversity/non-collapse;
- control adherence.

Do not collapse them into one opaque reward before validity is established.

## 9. Continuous training policy

The existing scheduled continuous workflow is explicitly **legacy/reference-only** after the audit. It requires:

```text
data/continuous/ENABLE_LEGACY_REFERENCE_TRAINING
data/continuous/train.tokens
data/continuous/validation.tokens
```

Do not use it for Compound data.

Important state invariants to carry into future Compound training:

- resumable model + optimizer + RNG state;
- last healthy rollback point;
- best held-out-validation checkpoint;
- spike/health reporting;
- full immutable milestone snapshots;
- mutable training state must never become an Adapter compatibility target.

## 10. Base/Adapter compatibility invariant

A published Base is identified by stable Base id + exact checkpoint SHA-256. An Adapter targets exactly one Base checkpoint. If checkpoint bytes change, create a new Base id/lineage.

Do not publish Compound community Bases/Adapters until a structured Compound architecture/tokenizer/runtime/Adapter compatibility contract is frozen.

## 11. Current empirical evidence

Proxy evidence in `docs/DESIGN_STATUS.md` includes:

- Compound randomized canonicalized roundtrip in the tested in-range stress distribution;
- proxy scale sweep where ~10M is the mobile-oriented middle point, not a proven optimum;
- long-memory proxy strongly favoring deterministic historical anchors;
- ControlField proxy favoring roughly 16 scalar levels/dimension as an efficiency/accuracy candidate.

The timing roundtrip proxy does **not** prove correctness for values above the current 1536-step cap.

## 12. Session continuation checklist

At a new session:

1. read `docs/AUDIT_2026-08-23.md`;
2. inspect the latest GitHub Actions `test` result for the audit commits;
3. read this file and `docs/DESIGN_STATUS.md`;
4. preserve the legacy Theory-REMI path unless explicitly migrating it;
5. start with long-time/BOS-field-schema blockers, then Compound Base implementation;
6. update the audit/status/handoff when a blocker is closed or a candidate is rejected.

## 13. Critical-path summary

```text
DONE: concept / compatibility model / Compound representation direction
DONE: experimental Compound event primitives
DONE: Compound MIDI parser
DONE: factorized Compound tokenizer records
DONE: ABI-tagged song-preserving Compound dataset + validated training-window loader
DONE: audit fixes for same-step ordering, legacy docs and accidental legacy continuous training
NEXT: long-time representation experiment
NEXT: BOS/EOS/start semantics + explicit field schema/masks
THEN: Compound Base model + masked factorized losses
THEN: synthetic/tiny-real overfit tests
THEN: external real-MIDI tokenizer + corpus validation
THEN: 5M/10M/20M real-MIDI sweep
THEN: long-memory / ControlField / target-runtime validation
THEN: Base rollout evaluation
OPTIONAL: SFT → DPO → reward RL, only if empirical gates require them
```
