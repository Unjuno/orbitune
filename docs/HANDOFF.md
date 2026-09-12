# Orbitune development handoff

> **Historical pointer:** the previous contents of this file were the 2026-08-24 development handoff, written before the current hierarchical Compound Base, corpus-scale research training, native V2 Web ABI and Pages runtime were completed. The original snapshot is preserved at [history/HANDOFF_2026-08-24.md](history/HANDOFF_2026-08-24.md).

For current work, read in this order:

1. [Repository overview](../README.md)
2. [Current roadmap](ROADMAP.md)
3. [Compound Base](COMPOUND_BASE.md)
4. [Compound browser runtime](COMPOUND_WEB_RUNTIME.md)
5. [Compound LoRA policy](COMPOUND_LORA_POLICY.md)
6. [Publication status](PUBLICATION.md)
7. [Current A2-512 model card](../models/research_nc_aria_gigamidi_a2_512_v1/README.md)
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
PyTorch A2-512 distribution         published on Hugging Face
Compound ONNX distribution          not published; exact-artifact parity gate remains
Compound dynamic LoRA ABI           not frozen
Planned LoRA Web strategy            pre-merged variant after Adapter + merged-artifact parity
A2-512 Base                         complete; step 220,813; events_seen 4,096,016,384
A2-512 checkpoint                   frozen and published externally
A2-512 source provenance            resolved to reachable commit 8489870f...
```

The canonical state is the [A2-512 release manifest](../models/research_nc_aria_gigamidi_a2_512_v1/manifest.json). The historical 100k model remains a separate unchanged identity. PyTorch Base publication is complete; Compound ONNX and Adapter publication are not.

## Non-negotiable boundaries

- Do not overwrite historical frozen model identities with later continuation checkpoints.
- Do not describe corpus capacity as unique training exposure.
- Do not mutate or repoint the completed A2-512 model identity or checkpoint SHA.
- Do not reuse the legacy Theory-REMI `orbitune-lora-v0` ABI for Compound.
- Do not use mutable training state as a public Adapter compatibility target.
- Do not treat Base V2 Web parity as proof that a future LoRA-merged export passes parity; validate the exact merged bytes separately.
- Do not publish research-NC model/derived artifacts as commercially eligible.
- Do not treat the published PyTorch checkpoint as evidence that an ONNX artifact passed parity.

The old handoff remains useful for reconstructing the state and rationale of the August architecture work, but it is no longer the implementation critical path.
