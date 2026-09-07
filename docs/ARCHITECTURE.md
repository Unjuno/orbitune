# Orbitune architecture

This document describes the **current repository architecture**. Older audit/design files are historical snapshots and may describe pre-implementation states.

Orbitune has two intentionally separate model/runtime families:

1. **Compound Transformer** — the current research model and new training/generation path.
2. **Theory-REMI reference** — the older operational Base/LoRA/ONNX path retained for compatibility and tests.

Their tokenizer, checkpoint, Adapter and browser ABIs are not interchangeable.

## Compound model

One serialized Compound MIDI event is one temporal model step. The current architecture is `orbitune-compound-hierarchical-gpt-v1` with tokenizer ABI `orbitune-compound-v0-experimental`.

```text
CompoundRecord (12 fields)
        ↓
factorized event embedding
        ↓
local causal Transformer ────────────────┐
        ↓                                │
medium summary Transformer ──────────────┤
        ↓                                ├─ context fusion
 global summary Transformer ─────────────┤
        ↓                                │
fast / medium / slow recurrent memory ───┘
        ↓
intra-event Transformer
        ↓
discrete + bounded continuous heads
        ↓
next CompoundRecord
```

The persistent generation state is bounded. It carries recent local records, bounded medium/global summary histories and fixed-size fast/medium/slow recurrent state instead of growing a full-history KV cache indefinitely.

The documented research checkpoint `research-nc-aria-gigamidi-v1` has 8,857,250 parameters and is frozen at global step 100,000. The checkpoint itself is not distributed by this repository; see the [model card](../models/research_nc_aria_gigamidi_v1/README.md) and [publication status](PUBLICATION.md).

## Compound event representation

The current event types are:

```text
NOTE
CC
PROGRAM
BANK
TEMPO
PEDAL
PITCH_BEND
CHANNEL_PRESSURE
POLY_PRESSURE
TIME_SIGNATURE
```

A serialized record contains 12 integer fields. Timing uses the current 96-steps-per-quarter representation with factorized coarse/residual values; continuous MIDI controls use factorized unsigned coarse/residual values. Event-specific masks and unused-field zeroing are part of native generation semantics.

The tokenizer ABI name still contains `experimental` for compatibility with the trained checkpoint. Renaming it would be an ABI change; documentation status must not be inferred from the string alone.

## Native generation state

Native Compound generation is streaming rather than stateless fixed-window recomputation. The model advances a carried state containing:

- local records;
- medium buffer/history;
- global buffer/history;
- fast/medium/slow recurrent memory;
- generation step count.

The intra-event decoder is also autoregressive: event type, channel, delta, attributes, velocity, duration and control are sampled in sequence, with each sampled prefix value conditioning later slots.

These semantics matter for export and Adapter work. A wrapper that teacher-forces a full record or reconstructs only the last 64 records is not equivalent to native generation.

## Compound browser ABI

The validated browser architecture is a two-graph V2 interface:

```text
stream.onnx
previous tensorized stream state + accepted record
→ context + updated stream state

        +

decoder_prefix.onnx
context + sampled intra-event prefix
→ decoder heads for the eight slot stages
```

The browser performs categorical/top-p and Gaussian sampling, masks, quantization, record construction and MIDI serialization outside the graphs. It reproduces Python half-to-even rounding and exact quantization boundary rules.

The source runtime is implemented under `web/`; the ONNX binaries are deliberately not published while redistribution review remains pending. See [COMPOUND_WEB_RUNTIME.md](COMPOUND_WEB_RUNTIME.md).

## Base identity and immutable lineages

A compatibility target is not just an architecture name. A frozen Base identity includes at least:

```text
model id
architecture ABI
tokenizer ABI
exact checkpoint SHA-256
rights / distribution scope
```

If checkpoint bytes change, that is a different compatibility target. Long-running mutable training state must never be used as a public Adapter dependency.

Historical model lineages remain immutable records rather than being overwritten by later continuations.

## Compound pretraining and LoRA

Compound Base pretraining is full-parameter optimization. LoRA is a separate post-Base adaptation stage:

```text
full-parameter pretraining
→ checkpoint selection
→ immutable Base freeze
→ Compound Adapter ABI freeze
→ frozen-Base LoRA training
```

The existing legacy `orbitune-lora-v0` ABI is **not** the Compound Adapter ABI. Compound target modules, ranks, scaling and serialization must be measured and frozen against an exact final Base under a new versioned compatibility identifier.

Until that is done, public Compound Adapter binaries are not accepted as production-compatible artifacts. Experimental LoRA code and bounded target/rank studies are allowed when clearly labeled. See [COMPOUND_LORA_POLICY.md](COMPOUND_LORA_POLICY.md).

For browser deployment, the initial supported Compound LoRA strategy is a pre-merged variant: merge a validated Adapter against the exact Base locally, export a matched V2 stream/decoder pair, repeat parity validation, then publish that pair only after the release/rights gates are closed.

## Theory-REMI legacy/reference ABI

The separate legacy/reference stack remains:

```text
architecture ABI   orbitune-midi-gpt-v0
tokenizer ABI      theory-remi-v0
Adapter ABI        orbitune-lora-v0
LoRA targets       q_proj + v_proj
LoRA rank          4
reference shape    4 layers / hidden 448 / 7 heads / context 1024
```

This path continues to support the existing `orbitune` CLI, legacy Base/Adapter registry, LoRA tooling and original browser runtime. It must remain isolated from Compound compatibility metadata.

## Repository pipeline

Current Compound project flow:

```text
rights/provenance-reviewed MIDI
→ parse / canonicalize / filter / deduplicate
→ indexed Compound corpus
→ full-parameter Base training / resume
→ immutable milestone checkpoint
→ held-out + generated-MIDI evaluation
→ native-generation export validation
→ publication/redistribution review
→ optional model artifact release
→ Compound Adapter ABI experiments/freeze
→ LoRA training against one immutable Base
→ optional pre-merged Web variants
```

Source publication, model publication and Adapter publication are separate gates.

## Rights boundary

The Apache-2.0 source license does not grant rights to datasets, Base checkpoints, Adapters or merged derivatives. A downstream Adapter cannot broaden the permissions of its Base.

For the documented Aria+GigaMIDI research lineage, `commercial_eligible=false` and the project distribution scope remains noncommercial. Any LoRA or pre-merged descendant must preserve that restriction unless it is built from an independently eligible Base lineage.

## Current public status

Available publicly:

- Python/runtime source;
- Compound architecture/training code;
- model documentation and hashes;
- V2 browser-runtime source and tests;
- publication and Compound LoRA policies.

Not currently distributed by this repository:

- the documented research `model.pt`;
- Compound `stream.onnx` / `decoder_prefix.onnx` binaries;
- public Compound model variants;
- production Compound Adapter binaries.

See [PUBLICATION.md](PUBLICATION.md) for the release checklist and current availability.
