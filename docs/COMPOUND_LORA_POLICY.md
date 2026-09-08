# Compound LoRA policy

## Status

This document defines the public policy for applying LoRA-style adaptation to Orbitune's Compound model family. It is a **policy and compatibility boundary**, not a claim that a frozen Compound Adapter ABI already exists.

The currently implemented public Adapter ABI, `orbitune-lora-v0`, belongs to the separate legacy Theory-REMI stack. It must not be applied to Compound checkpoints merely because both models contain attention layers.

## Pretraining and adaptation are separate stages

Orbitune treats Base pretraining and LoRA adaptation as different optimization stages:

```text
full-parameter Base pretraining
→ evaluate / select checkpoint
→ freeze immutable Base identity + SHA-256
→ define and validate a Compound Adapter ABI
→ freeze Base weights during Adapter training
→ train LoRA parameters only
→ evaluate Adapter against Base-only behavior
→ publish only after rights + compatibility review
```

LoRA is **not** used to finish or replace full-parameter Base pretraining. A mutable continuation checkpoint is training state, not a public Adapter compatibility target. The repository now records a reported active local long-run Base continuation toward step 1,000,000; that run remains mutable training state until a completed checkpoint is evaluated, selected, frozen, hashed and documented under a new Base identity.

Intermediate checkpoints may be used for local experiments, but an Adapter trained against one must remain explicitly experimental and bound to that exact checkpoint SHA-256. If the Base changes, the Adapter must be retrained or separately revalidated; compatibility is never inferred from architecture name or parameter count.

## Current Compound Base state

The repository currently documents the frozen research checkpoint `research-nc-aria-gigamidi-v1` at global step 100,000. Its exact reported SHA-256 remains the model identity recorded in the model card and publication record.

A separate local continuation is reported active from that immutable parent. It does not overwrite the historical 100k model, and its mutable continuation checkpoint is not distributed by this repository. A later continuation checkpoint may become a new Base candidate only after it is completed, evaluated, frozen, hashed and documented under a new model identity.

No public Compound community Adapter target exists today because the Compound Adapter ABI itself has not been frozen.

## What must be frozen before a public Compound Adapter ABI

A Compound Adapter ABI must be introduced under a new identifier; it must not reuse `orbitune-lora-v0`.

Before accepting public Compound Adapters, freeze and test at least:

1. exact Base model id and checkpoint SHA-256;
2. architecture ABI and tokenizer ABI;
3. exact target-module names and tensor shapes;
4. rank / alpha conventions and scaling semantics;
5. Adapter tensor names, dtype and serialization format;
6. Base-binding metadata embedded in the Adapter artifact;
7. training/inference merge semantics and numerical tolerances;
8. checkpoint loading behavior for missing, duplicate or incompatible tensors;
9. rights/provenance metadata required from Adapter contributors;
10. local Python and browser deployment behavior.

Target modules and rank are deliberately **not frozen by this document**. They must be selected from measured Compound experiments against the exact final Base rather than copied from the legacy `q_proj` / `v_proj`, rank-4 configuration.

## Adapter training contract

Once a Compound Adapter ABI is frozen, Adapter training should follow these invariants:

- Base parameters remain frozen.
- Only declared Adapter parameters are trainable.
- The Adapter records the exact Base model id and checkpoint SHA-256.
- The Adapter records its Adapter ABI, target modules, rank, alpha/scaling, source commit and training configuration.
- Resume state must not silently change the Base checkpoint.
- Adapter validation data remains separate from Adapter training data.
- Base-only generation remains reproducible after Adapter injection/removal.
- A failed or missing Adapter must not mutate or partially load into the Base.

The initial public implementation should prefer a small, inspectable Safetensors artifact rather than arbitrary executable checkpoint formats.

## Rights and lineage

An Adapter cannot broaden the permissions of its Base.

For the current research-NC Compound lineage:

```text
Base: research-NC / noncommercial
        ↓
LoRA Adapter targeting that Base
        ↓
merged derivative / Web variant

maximum allowed scope remains research-NC / noncommercial
```

Adapter contributors must also have rights to the data they use for adaptation and must declare those rights separately. A permissively licensed Adapter training set does not make a restricted Base commercially eligible.

Commercial and research-NC Base lineages must remain separate. No Adapter, merge or export operation may create a research-NC → commercial backflow.

## Web deployment policy

The validated Compound **Base** browser runtime uses a two-graph native-generation ABI (`stream` + `decoder_prefix`). Dynamic Compound LoRA injection is not currently a validated Web ABI.

The planned initial LoRA Web deployment strategy is:

```text
frozen Compound Base
+ validated LoRA Adapter
→ merge locally against the exact Base
→ export a matched stream.onnx + decoder_prefix.onnx pair
→ validate native/Web parity for those exact merged bytes
→ only then publish as kind = "lora-premerged"
```

This is a deployment plan, not evidence that a Compound LoRA merged artifact has already passed parity. Every future pre-merged variant must remain bound to the exact Base checkpoint SHA and an explicit Adapter identity. The browser runtime must reject mismatched Base, architecture, tokenizer or runtime ABI metadata.

Dynamic browser-side LoRA may be added later only as a separate ABI after tensor packing, numerical parity, memory cost and load-time behavior are validated.

## Public contribution status

Accepted now:

- Compound LoRA research code that is clearly experimental;
- bounded target-module/rank experiments;
- tests and documentation for a future Compound Adapter ABI;
- evaluation tooling that does not publish restricted model artifacts.

Not accepted yet as public Compound Adapter releases:

- community Adapter binaries claiming production Compound compatibility;
- use of the legacy Theory-REMI Adapter manifest/ABI for Compound;
- Adapters targeting mutable long-run training state;
- public merged Compound weights or ONNX variants before redistribution review;
- claims of commercial eligibility for research-NC descendants.

When the final public target Base and Compound Adapter ABI are frozen, this policy should be supplemented by a concrete contribution schema and CLI workflow.

## Release gate

A future Compound Adapter release is ready only when all of the following are true:

- Base checkpoint identity is immutable and publicly documented;
- Compound Adapter ABI is frozen and versioned;
- target modules/rank/scaling are tested on the exact Base;
- Adapter artifact is strictly Base-SHA bound;
- generated MIDI and held-out evaluation are recorded;
- training-data rights are declared;
- Base and Adapter distribution terms are compatible;
- Web deployment, if offered, passes native-generation parity with the exact merged artifact;
- publication metadata exposes real versioned URLs and SHA-256 values rather than placeholders.

Until then, the repository should describe Compound LoRA as a supported **planned adaptation path with explicit gates**, not as a currently downloadable community Adapter ecosystem.
