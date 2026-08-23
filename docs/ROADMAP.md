# Orbitune Roadmap

This roadmap tracks the current production direction. Historical 3M/Theory-REMI milestones are retained only as legacy implementation history; they are not the current critical path.

## Milestone A — repository and compatibility foundation — DONE

- Apache-2.0 source license
- Base/Adapter immutable checkpoint-hash dependency model
- Base and Adapter manifests/registries
- GitHub Actions test/export/validation infrastructure
- browser reference UI
- resumable continuous-training/snapshot infrastructure for the legacy path

## Milestone B — legacy/reference implementation — DONE

- Theory-REMI tokenizer and MIDI note roundtrip
- decoder-only GPT reference implementation
- ~10.2M current reference configuration
- LoRA training/runtime path
- ONNX/Web export path

This stack remains useful for reference and CI, but it is not the future production tokenizer ABI.

## Milestone C — Compound representation and data path — IN PROGRESS

Completed:

- Hybrid Compound Event direction: one musical event = one Transformer step
- deterministic MIDI-1 note canonicalization
- parallel Compound MIDI type-0/1 parser
- factorized timing and continuous-value records
- song-preserving Compound JSONL corpus preparation
- exact-byte duplicate split protection
- fixed-length song-local training-window loader

Still required before ABI freeze:

- external real-MIDI timing validation
- resolve time values beyond the current 1536-step experimental range
- decide BOS/EOS / unconditional-generation start semantics
- define explicit field cardinalities and validity masks
- validate expressive pedal representation and other rare MIDI events
- production corpus provenance, quality filtering and near-duplicate/composition-aware splitting

## Milestone D — Compound Base model — NEXT

- factorized CompoundRecord embeddings
- configurable causal decoder-only Transformer
- event-type-conditioned lightweight intra-event autoregressive heads
- unused-field loss masking
- synthetic forward/backward and overfit tests
- tiny real-MIDI overfit test
- checkpoint/resume support using the same training-state invariants as the existing continuous loop

Reference size starts near 10M, but size is not frozen.

## Milestone E — empirical model selection

- external real-MIDI 5M / 10M / 20M sweep
- compare quality / MB / latency / parameter
- freeze the first production Base size only after this experiment
- validate 96/qn and 7+16 DELTA/DURATION against nearby alternatives on the same corpus

## Milestone F — long generation and control

- recent-only sliding baseline
- deterministic 16/32/64/128/256-bar historical anchors
- optional randomized historical samples
- trained-model long-rollout repetition/motif/coherence evaluation
- ControlField quantization/adherence evaluation on real MIDI

## Milestone G — mobile/browser runtime

- define Compound runtime/export ABI
- benchmark INT8 baseline
- implement/test packed ternary only with a real native/WASM/WebGPU kernel
- benchmark latency, memory and thermal behavior on target phones
- retain FP16 fallback where appropriate

## Milestone H — Base/Adapter ecosystem

- publish immutable Base id + exact checkpoint hash
- define Compound-compatible Adapter ABI
- verify LoRA target modules/ranks against the selected Compound Base
- community adapters may then target that exact lineage

## Milestone I — post-training necessity gate

Policy learning is not mandatory by default.

```text
Base pretraining
→ held-out rollout evaluation
→ high-quality SFT comparison
→ DPO only if a reliable selection gap remains
→ reward-based RL only after reward validity + anti-collapse tests
```

The branch is allowed to terminate at `NOT REQUIRED`.

See `docs/POST_TRAINING_RESEARCH.md`.

## Ongoing requirement — continuous pretraining

After the production Compound corpus and Base path are accepted:

- continuous quality-gated corpus accumulation;
- resumable optimizer/model/RNG state;
- held-out health checks and automatic rollback;
- immutable milestone snapshots;
- mutable continuous-training state must never be used as an Adapter compatibility target.
