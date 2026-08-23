# Orbitune Architecture

Orbitune is a MIDI-only generation system with repository-managed compact Base models and small LoRA Adapters.

## Compatibility graph

```text
Base A (id=base-a, checkpoint SHA=A)
  ├── Adapter A1 (base_model=base-a, base_sha256=A)
  └── Adapter A2 (base_model=base-a, base_sha256=A)

Base B (id=base-b, checkpoint SHA=B)
  └── Adapter B1 (base_model=base-b, base_sha256=B)
```

A Base id is immutable after Adapters depend on it. Changing checkpoint bytes means introducing a new Base id. Existing Base/Adapter lineages remain valid.

Protocol identifiers are separate from Base identity. In particular, the current legacy/reference stack and the experimental production-candidate Compound stack must not be conflated.

## Legacy/reference ABI

The existing operational reference path is:

- architecture ABI: `orbitune-midi-gpt-v0`
- tokenizer ABI: `theory-remi-v0`
- reference configuration: 4 layers, hidden size 448, 7 heads, context 1024
- current 204-token vocabulary parameter count: about 10.2M
- LoRA targets: `q_proj`, `v_proj`
- LoRA rank: 4
- Adapter format ABI: `orbitune-lora-v0`

This path remains available for tests, existing training/export infrastructure and browser reference work. It is **not** the frozen production architecture for future Compound Bases.

The historical 4-layer/240-wide ~3M configuration remains under `configs/base_3m.json` as an experiment/legacy configuration only. It is not the current reference Base size.

## Production-candidate Compound architecture

Current direction:

```text
MIDI type 0/1
→ Compound MIDI parser + deterministic canonicalization
→ one Compound Event per Transformer step
→ factorized field embeddings
→ causal decoder-only Transformer (~10M reference target)
→ event-type prediction
→ lightweight intra-event autoregressive attribute cascade
→ factorized timing / continuous-value heads
→ Compound Event
→ MIDI
```

Current experimental tokenizer ABI: `orbitune-compound-v0-experimental`.

Reference candidates under validation:

- 96 steps per quarter note;
- DELTA and DURATION: 7 coarse ranges + 16 residual levels;
- continuous controls: coarse + residual factorization;
- recent dense context + deterministic historical anchors for long generation;
- ControlField as a separate extension boundary;
- packed ternary as a deployment candidate, with INT8 and FP16 fallbacks.

See `docs/DESIGN_STATUS.md` for acceptance status and `docs/HANDOFF.md` for the current implementation critical path.

## Stable extension boundaries

The architecture should preserve separate interfaces for:

- `LinearBackend` — dense/reference today; ternary is a candidate implementation;
- `ControlField` — null/default versus experimental musical-time control;
- `MemoryPolicy` — sliding/reference versus anchored/dilated long-history selection.

Concrete experimental implementations must not silently redefine the Base/Tokenizer ABI.

## Repository pipeline

```text
licensed/provenance-reviewed MIDI corpus
→ parse / canonicalize / quality filter / deduplicate
→ composition-aware train-validation split
→ Compound encoding
→ Base candidate training
→ best held-out checkpoint
→ runtime/export validation
→ stage bases/<base-id>/ with exact hashes
→ Base registry generation
→ Adapter training against an immutable selected Base
→ Adapter manifest + artifact Base id/hash binding
→ dependency validation
→ browser/local Base/Adapter selection
```

The current Compound dataset implementation already prevents **exact-byte duplicate** leakage by SHA-256. Near-duplicate/composition-family deduplication is still a production-corpus validation gate and must not be confused with exact-hash grouping.

## Browser runtime

GitHub Pages receives generated `bases.json` and `adapters.json`. For the legacy Web path, only Web ONNX artifacts are copied into the static site; training checkpoints remain repository artifacts.

A future Compound Web ABI must be introduced explicitly. The existing Theory-REMI ONNX graph is not automatically compatible with Compound records.

Before inference the runtime must verify artifact hashes and Adapter/Base compatibility metadata.

## Repository policy

- Base artifacts are committed under `bases/<base-id>/`.
- Each Base checkpoint and Web artifact is limited to 95 MiB by current CI policy.
- Base manifests allow at most 100M parameters.
- Adapters are committed under `adapters/official` or `adapters/community`.
- Adapter manifests reference Base id + exact checkpoint SHA-256.
- CI rejects unknown Base ids, hash mismatches, incompatible declared ABIs and oversized artifacts.

## Production blockers

Before freezing the Compound ABI or publishing an official Compound Base, close at least these gates:

1. external real-MIDI timing/tokenizer validation;
2. long-delta/duration representation beyond the current 1536-step experimental range;
3. explicit field cardinalities, masks, BOS/EOS/start-of-generation behavior and masked losses;
4. production corpus provenance plus near-duplicate/composition-aware splitting;
5. real-MIDI 5M/10M/20M scale sweep;
6. real-device runtime benchmark;
7. trained-model long-memory and ControlField evaluation.
