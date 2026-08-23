# Orbitune Development Handoff

Last updated: 2026-08-23

This document is the compact source of truth for continuing Orbitune development in a fresh session. Read this together with `README.md`, `docs/DESIGN_STATUS.md`, and `docs/POST_TRAINING_RESEARCH.md` before changing architecture.

## 1. Product goal

Orbitune is a lightweight local-first symbolic MIDI generation framework with:

- a compact shared pretrained Base;
- distributable LoRA adapters tied to immutable Base checkpoint hashes;
- local/browser-oriented inference;
- MIDI/event modeling rather than raw audio generation.

The current production direction is **Hybrid Compound Events**, not the legacy flat Theory-REMI representation.

## 2. Architecture decisions already made

Do not reopen these without contradictory real-data evidence:

- causal decoder-only Transformer;
- approximately 10M parameters as the current reference point, pending real-MIDI 5M/10M/20M sweep;
- one musical event = one Transformer step;
- Compound event types: NOTE, CC, PROGRAM, BANK, TEMPO, PEDAL, PITCH_BEND, CHANNEL_PRESSURE, POLY_PRESSURE, TIME_SIGNATURE;
- deterministic MIDI canonicalization: merge same-onset/channel/pitch duplicates and truncate an active same-channel/pitch NOTE on retrigger;
- candidate timing grid: 96 steps per quarter note;
- candidate DELTA/DURATION encoding: 7 coarse ranges + 16 residual levels;
- continuous attributes use factorized coarse + residual representation rather than large flat vocabularies;
- NOTE intra-event decoding direction: lightweight autoregressive MLP cascade;
- long generation: recent dense sliding window plus deterministic historical anchors, with optional randomized historical samples;
- LoRA remains a runtime adapter over an immutable Base;
- deployment research: packed ternary candidate, INT8 fallback, FP16 fallback.

Rejected/downranked directions include mandatory 3M Base, pure flat event production representation, independent Compound heads, GRU/mini-attention NOTE decoders, recent-only infinite context, and mandatory DPO/RL.

## 3. Legacy versus production-candidate code

### Legacy/reference path

The existing `theory-remi-v0` / ~10.2M `OrbituneGPT` path remains operational for reference, CI, existing Web export and adapter infrastructure. It is **not the frozen production ABI**.

Do not silently mutate this ABI while developing Compound support.

### Experimental Compound path

Implemented so far:

- `orbitune/compound.py`
  - `orbitune-compound-v0-experimental` event primitives;
  - canonicalization;
  - 96/qn timing constants;
  - 7+16 timing quantization.
- `orbitune/compound_midi.py`
  - parallel MIDI type-0/1 parser preserving the Compound event scope;
  - intentionally separate from legacy `orbitune.midi.read_midi`.
- `orbitune/quantization.py`
  - factorized continuous-value quantization helpers.
- `orbitune/tokenizer/compound_event.py`
  - one-event-per-record representation;
  - factorized DELTA/DURATION and continuous attributes.
- `orbitune/compound_dataset.py`
  - song-preserving Compound JSONL preparation;
  - SHA-256 grouped train/validation split to prevent exact duplicate leakage.
- `orbitune/compound_training.py`
  - fixed-length training windows from Compound JSONL;
  - windows do not cross composition boundaries.
- `scripts/prepare_compound_corpus.py`
  - entry point for preparing a real Compound corpus.
- Compound tests have been added for tokenizer/pipeline/training primitives.

The Compound ABI is still experimental and must not be used as an immutable contributed Base target yet.

## 4. Immediate next implementation

The next critical task is **the Compound Base model**.

Expected path:

```text
CompoundRecord [B, S, fields]
→ factorized field embeddings
→ causal decoder-only Transformer
→ event-type prediction
→ lightweight intra-event autoregressive attribute cascade
→ factorized attribute heads
```

Do not prematurely build production DPO/RL infrastructure.

Recommended implementation sequence:

1. define explicit field cardinalities and masking/validity rules for every Compound event type;
2. implement Compound input embeddings and a small configurable Transformer;
3. implement event-type-conditioned lightweight autoregressive attribute heads;
4. implement loss masking so unused attributes do not contribute loss;
5. run synthetic one-batch forward/backward and overfit tests;
6. run a tiny real-MIDI overfit test;
7. parameterize model size so 5M/10M/20M sweeps use the same code path;
8. only then connect full pretraining/checkpoint/export infrastructure.

A key invariant is that the Transformer advances once per musical event. Attribute autoregression is intra-event and must not inflate Transformer sequence length.

## 5. Experiments still blocking production decisions

Architecture ideation is mostly complete. Remaining decisions should be made by experiments, not additional speculation.

### E1 — timing/tokenizer validation

Use external real MIDI to compare reconstruction/error and sequence efficiency for the 96/qn and 7+16 candidate against nearby alternatives.

### E2 — production corpus

Finalize corpus composition, provenance/license policy, quality filtering, deduplication and split strategy. Do not start an official Base on unreviewed data.

### E3 — Base scale

Train comparable 5M/10M/20M Compound models on real MIDI. The current ~10M choice is only a reference knee point from proxy experiments.

### E4 — long memory

Compare recent-only sliding context against anchored/dilated memory on trained long rollouts. Proxy experiments strongly favored anchors, but production acceptance requires real-model validation.

### E5 — ControlField

Validate real-corpus control adherence and quantization. Current reference is roughly six control dimensions with 16 levels/dimension and musical-time Adaptive Gaussian RBF behavior.

### E6 — deployment quantization

Benchmark INT8 versus packed ternary on actual target browser/device runtimes. PyTorch STE ternary results are not a production runtime benchmark.

### E7 — post-training necessity

Only after Base pretraining and rollout evaluation. See `docs/POST_TRAINING_RESEARCH.md`.

## 6. Post-training / policy-learning policy

Status: **OPEN / NOT REQUIRED BY DEFAULT**.

Required order:

```text
Base pretraining
→ held-out rollout evaluation
→ compare Base with high-quality SFT
→ if a meaningful selection gap remains, test DPO
→ reward-based RL only after reward validity and anti-collapse checks
```

The project must allow the branch to terminate with `NOT REQUIRED`. A pretrained Base is not assumed to require human-preference alignment merely because LLM pipelines often do.

Primary evaluation dimensions must remain separated:

- continuation fit;
- standalone musical quality;
- diversity/non-collapse;
- control adherence.

Do not collapse these into a single opaque reward before validity is established.

## 7. Continuous training policy

The repository already contains scheduled continuous-training infrastructure for the legacy/reference path. Important concepts to preserve when Compound training replaces/extends it:

- resumable model + optimizer + RNG state;
- last healthy rollback point;
- best held-out-validation checkpoint;
- spike/health reporting;
- full immutable snapshots by cumulative token/event milestone;
- continuous mutable training state must never become a Base compatibility target.

Earlier project direction chose approximately 10M-token snapshot milestones for the existing continuous loop.

## 8. Base/Adapter compatibility invariant

A published Base is identified by a stable Base id plus exact checkpoint SHA-256. An Adapter targets exactly one Base checkpoint. Never treat a Base id as a mutable rolling slot. If checkpoint bytes change, create a new Base id/lineage.

This is important because contributors may add adapters independently.

## 9. Current empirical evidence

Proxy experiments recorded in `docs/DESIGN_STATUS.md` include:

- Compound randomized property tests with effectively complete canonicalized encode/decode roundtrip in the tested stress distribution;
- proxy model-scale sweep where ~10M was the mobile-oriented middle point between ~5M and ~20M;
- long-memory proxy where deterministic anchors materially improved access to dependencies extending to 256 bars;
- ControlField proxy where 16 scalar levels/dimension was a strong efficiency/accuracy candidate.

Treat all proxy results as evidence, not production proof. External real-MIDI and target-runtime experiments remain the acceptance gates.

## 10. Session continuation checklist

At the start of a new development session:

1. read this file;
2. read `docs/DESIGN_STATUS.md`;
3. read `docs/POST_TRAINING_RESEARCH.md`;
4. inspect the current Compound files before editing;
5. preserve the legacy Theory-REMI path unless explicitly migrating it;
6. continue with Compound Base model implementation and smoke tests;
7. update this handoff whenever a candidate becomes accepted/rejected or the critical path changes.

## 11. Critical-path summary

```text
DONE: concept / compatibility model / Compound representation direction
DONE: experimental Compound event primitives
DONE: Compound MIDI parser
DONE: factorized Compound tokenizer records
DONE: song-preserving Compound dataset + training-window loader
NEXT: Compound Base model + masked factorized losses
THEN: synthetic/tiny-real overfit tests
THEN: external real-MIDI tokenizer + corpus validation
THEN: 5M/10M/20M real-MIDI sweep
THEN: long-memory / ControlField / target-runtime validation
THEN: Base rollout evaluation
OPTIONAL: SFT → DPO → reward RL, only if empirical gates require them
```
