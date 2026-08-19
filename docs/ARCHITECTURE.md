# Orbitune Architecture

Orbitune v0 is a MIDI-only generation system with one fixed small base model and small LoRA adapters.

## Core idea

```text
Base Model = shared MIDI grammar
LoRA Adapter = style / mood / genre / texture-like generation tendency
```

The system distributes generation capability rather than finished audio tracks.

## Fixed v0 architecture

`orbitune-tiny-v0` is intentionally a single compatibility target:

- architecture id: `orbitune-midi-gpt-v0`
- parameters: **2,945,760** with the fixed Theory-REMI v0 vocabulary
- Transformer layers: 4
- hidden size: 240
- attention heads: 4
- head dimension: 60
- maximum context: 512 tokens
- vocabulary: 204 tokens
- tied token embedding / language-model head
- LoRA targets: `q_proj`, `v_proj`
- default LoRA rank: 4

Adapters for a differently shaped base model are not Orbitune v0-compatible. If the base architecture changes incompatibly, it must receive a new base-model/version identifier.

## v0 pipeline

```text
MIDI corpus
  -> Type-0 / Type-1 MIDI reader
  -> Theory-REMI v0 tokenizer
  -> corpus preparation with song boundaries
  -> orbitune-tiny-v0 base training
  -> LoRA adapter training
  -> Base + Adapter MIDI generation
  -> structural MIDI evaluation
  -> MIDI playback / export
```

## Runtime target

The default v0 runtime target is smartphone-friendly symbolic generation:

- fixed 2.95M-parameter base model
- context length: 512 tokens
- output: MIDI event tokens
- one adapter at a time
- generate-then-play UX for 4 to 8 bars
- WebAssembly-compatible browser inference as the baseline target
- WebGPU acceleration may be added where supported, but is not required for the product contract

## Event format

Theory-REMI v0 uses only music events:

- `BAR`
- `POSITION_0..15`
- `NOTE_PITCH_21..108`
- `NOTE_DURATION_1..64`
- `VELOCITY_1..32`

Future versions may add control and texture-control events, but v0 does not generate audio waveforms.

## Deployment graph

The PyTorch model can be captured with a dynamic sequence axis using `torch.export`. ONNX translation is optional and requires the repository's `export` dependencies. The deployment wrapper emits logits only; sampling and MIDI grammar constraints remain runtime logic.

The first browser implementation may recompute the context for each generated token. KV-cache export is a later optimization and should only be added after smartphone measurements justify the extra graph/interface complexity.

## Repository policy

- Source code and small adapters may be committed.
- Base model weights are not committed under `models/`.
- Community adapters must validate and include metadata, demo MIDI, and training-data rights declarations.
