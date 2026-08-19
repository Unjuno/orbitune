# Orbitune Architecture

Orbitune is a MIDI-only generation system with one immutable small Base checkpoint and small LoRA adapters.

## Compatibility model

```text
orbitune-base.pt bytes
        |
        +-- SHA-256 H = permanent compatibility key
        |
        +-- LoRA adapters must declare base_sha256 = H
```

The public Base is not a rolling version. Once released, `orbitune-base.pt` must never be replaced. A future different Base is a separate compatibility lineage; it does not supersede or mutate the original Base/Adapter ecosystem.

Protocol identifiers such as `orbitune-midi-gpt-v0`, `theory-remi-v0`, and `orbitune-lora-v0` are ABI/file-format versions. They are deliberately separate from Base weight identity.

## Fixed Base architecture

- public identity: `orbitune-base`
- exact checkpoint compatibility: SHA-256
- architecture ABI: `orbitune-midi-gpt-v0`
- parameters: 2,945,760
- Transformer layers: 4
- hidden size: 240
- attention heads: 4
- head dimension: 60
- maximum context: 512 tokens
- vocabulary: 204 tokens
- tokenizer ABI: `theory-remi-v0`
- LoRA targets: `q_proj`, `v_proj`
- LoRA rank: 4
- adapter ABI: `orbitune-lora-v0`

Matching architecture dimensions are necessary but not sufficient for Adapter compatibility. The exact Base checkpoint hash must also match.

## Pipeline

```text
MIDI corpus
  -> Type-0 / Type-1 MIDI reader
  -> Theory-REMI tokenizer
  -> file-content dedup + train/validation split
  -> Base candidate training
  -> best held-out-validation checkpoint selection
  -> freeze and publish orbitune-base exactly once
  -> Adapter training pinned to exact Base SHA-256
  -> Base + Adapter generation
  -> MIDI playback / export
```

## Runtime target

- one immutable 2.95M-parameter Base
- context length 512
- MIDI event-token output
- one rank-4 Adapter at a time
- browser baseline: WebAssembly
- optional acceleration may be evaluated separately

The browser ONNX graph uses external LoRA inputs, so every Adapter remains small. Browser loading verifies the ONNX asset hash and independently verifies each Adapter's `base_sha256` against the immutable Base checkpoint identity.

## Event ABI

Theory-REMI currently contains:

- `BAR`
- `POSITION_0..15`
- `NOTE_PITCH_21..108`
- `NOTE_DURATION_1..64`
- `VELOCITY_1..32`

These event/file ABIs may receive explicit compatibility versions. Such ABI versioning must never be interpreted as permission to silently replace the Base checkpoint.

## Repository policy

- Source code and small compatible Adapters may be committed.
- Base weights are distributed outside Git history.
- Adapter manifests and Safetensors metadata both carry the exact Base SHA-256.
- CI rejects mixed Base hashes in the bundled Adapter registry.
