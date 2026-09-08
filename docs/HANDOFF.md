# Orbitune development handoff

> **Historical pointer:** the previous contents of this file were the 2026-08-24 development handoff, written before the current hierarchical Compound Base, corpus-scale research training, native V2 Web ABI and Pages runtime were completed. The original snapshot is preserved at [history/HANDOFF_2026-08-24.md](history/HANDOFF_2026-08-24.md).

For current work, read in this order:

1. [Repository overview](../README.md)
2. [Current roadmap](ROADMAP.md)
3. [Compound Base](COMPOUND_BASE.md)
4. [Compound browser runtime](COMPOUND_WEB_RUNTIME.md)
5. [Compound LoRA policy](COMPOUND_LORA_POLICY.md)
6. [Publication status](PUBLICATION.md)
7. [Research model card](../models/research_nc_aria_gigamidi_v1/README.md)
8. model-specific data/reproducibility documents when changing claims about the frozen checkpoint

## Current handoff summary

```text
Public source repository             ready
Compound hierarchical Base          implemented
Frozen research model V1            documented at step 100,000
Indexed research corpus             ~4.069B active next-event pairs
Frozen V1 cumulative events_seen    409.6M
Native Compound Base Web ABI V2     validated in local export work
Compound browser runtime source     merged and deployed
Model / ONNX distribution           still frozen pending redistribution review
Compound dynamic LoRA ABI           not frozen
Planned LoRA Web strategy            pre-merged variant after Adapter + merged-artifact parity
Long-run Base continuation          active local run toward step 1M; mutable checkpoint remains unpublished
```

Latest reported continuation evidence is recorded in [ROADMAP.md](ROADMAP.md). It reports the immutable 100k parent unchanged and the local continuation active beyond step 100k. Treat that evidence as a local-run status snapshot, not as a public checkpoint-byte verification or release.

## Non-negotiable boundaries

- Do not overwrite historical frozen model identities with later continuation checkpoints.
- Do not describe corpus capacity as unique training exposure.
- Do not describe the active continuation as completed until an actual final checkpoint is saved, loaded, hashed, evaluated and documented.
- Do not treat reported local continuation state as a downloadable or publicly verified checkpoint.
- Do not reuse the legacy Theory-REMI `orbitune-lora-v0` ABI for Compound.
- Do not use mutable training state as a public Adapter compatibility target.
- Do not treat Base V2 Web parity as proof that a future LoRA-merged export passes parity; validate the exact merged bytes separately.
- Do not publish research-NC model/derived artifacts as commercially eligible.
- Do not publish checkpoint/ONNX URLs before redistribution review is actually completed.

The old handoff remains useful for reconstructing the state and rationale of the August architecture work, but it is no longer the implementation critical path.
