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

A Base id is immutable after Adapters depend on it. Changing the Base checkpoint means introducing a new Base id. Existing Base/Adapter lineages remain valid.

Protocol identifiers such as `orbitune-midi-gpt-v0`, `theory-remi-v0`, and `orbitune-lora-v0` are ABI/file-format versions and are separate from Base identity.

## Current reference ABI

- architecture ABI: `orbitune-midi-gpt-v0`
- parameters: 2,945,760
- Transformer layers: 4
- hidden size: 240
- attention heads: 4
- context: 512 tokens
- vocabulary: 204 tokens
- tokenizer ABI: `theory-remi-v0`
- LoRA targets: `q_proj`, `v_proj`
- LoRA rank: 4
- Adapter ABI: `orbitune-lora-v0`

Current rank-4 LoRA packing is shape-specific to this ABI. A contributed Base with another architecture can coexist in the Base registry, but needs a matching Adapter ABI before LoRA Adapters may target it.

## Repository pipeline

```text
MIDI corpus
  -> tokenizer / dedup / train-validation split
  -> Base candidate training
  -> best validation checkpoint
  -> ONNX export
  -> stage bases/<base-id>/ with exact hashes
  -> Base registry generation
  -> Adapter training against a selected Base
  -> Adapter manifest + Safetensors base id/hash binding
  -> dependency validation
  -> browser Base/Adapter selection
  -> MIDI generation
```

## Browser runtime

GitHub Pages receives generated `bases.json` and `adapters.json`. For each Base only its Web ONNX artifact is copied into the static site; PyTorch checkpoints remain in the repository for training and Adapter creation.

Selecting an Adapter determines its required Base. Before inference the browser verifies:

1. the selected Base ONNX SHA-256;
2. the Adapter's embedded `base_sha256` against the selected Base checkpoint identity.

## Repository policy

- Base artifacts are committed under `bases/<base-id>/`.
- Each Base checkpoint and Web ONNX artifact is limited to 95 MiB by current CI policy.
- Base manifests allow at most 100M parameters.
- Adapters are committed under `adapters/official` or `adapters/community`.
- Adapter manifests reference Base id + exact checkpoint SHA-256.
- CI rejects unknown Base ids, hash mismatches, ABI mismatches, and oversized artifacts.
