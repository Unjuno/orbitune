# Orbitune

Orbitune is a lightweight MIDI generation framework for local BGM creation. The repository manages compact Base models and small LoRA Adapters together so contributors can add either new generation foundations or new style tendencies without breaking existing compatibility lineages.

## Core dependency model

A Base is identified by a stable `base-id` plus the exact SHA-256 of its checkpoint bytes. An Adapter always targets exactly one Base checkpoint.

```text
bases/base-a/model.pt  SHA=A
  ├── adapters/chill-a      base_model=base-a, base_sha256=A
  └── adapters/fantasy-a    base_model=base-a, base_sha256=A

bases/base-b/model.pt  SHA=B
  └── adapters/jazz-b       base_model=base-b, base_sha256=B
```

A Base id is not a rolling version slot. If its checkpoint changes, contribute a new Base id. Existing Bases and their Adapters remain valid.

`orbitune-midi-gpt-v0`, `theory-remi-v0`, and `orbitune-lora-v0` are ABI/version identifiers for architecture, tokenization, and Adapter format. They are separate from Base identity.

## Current reference architecture

```text
Parameters       2,945,760
Layers           4
Hidden size      240
Attention heads  4
Context          512 tokens
Architecture ABI orbitune-midi-gpt-v0
Tokenizer ABI    theory-remi-v0 (204 tokens)
LoRA rank        4
LoRA targets     q_proj + v_proj
Adapter format   Safetensors / orbitune-lora-v0
```

Orbitune is MIDI-only. Raw audio, vocals, audio-codec tokens, and DAW-quality mixing are outside the current scope.

## Repository layout

```text
bases/
  <base-id>/
    manifest.json
    model.pt
    web.onnx
    README.md

adapters/
  official/<adapter-id>/
  community/<adapter-id>/

registry/
  bases.json
  adapters.json
```

Current repository policy allows Base models up to 100M parameters and limits each committed Base binary artifact to 95 MiB. Adapters remain much smaller; the hard Adapter directory threshold is 5 MiB.

## Quick start

```bash
git clone https://github.com/Unjuno/orbitune.git
cd orbitune
python -m pip install -e .[dev]
python -m pytest -q
orbitune model-info
```

## Prepare MIDI data

Orbitune accepts Standard MIDI File type 0 and type 1. The current Theory-REMI tokenizer is 4/4-oriented and rejects unsupported time signatures rather than silently corrupting bar positions.

```bash
orbitune prepare-split-corpus data/raw \
  --train-out data/tokens/train.tokens \
  --validation-out data/tokens/validation.tokens \
  --report data/tokens/split-report.json \
  --validation-fraction 0.1 \
  --min-events 8
```

Identical MIDI bytes are grouped by SHA-256 so renamed duplicates cannot leak across train and validation.

## Train a Base candidate

```bash
orbitune train-base \
  --tokens data/tokens/train.tokens \
  --validation-tokens data/tokens/validation.tokens \
  --validation-interval 100 \
  --out models/my-base.pt \
  --steps 1000 \
  --batch-size 8 \
  --seq-len 128
```

With periodic validation enabled, Orbitune restores the checkpoint with minimum held-out validation loss instead of blindly publishing the final step.

Export the browser graph:

```bash
python -m pip install -e '.[export]'
orbitune export-web-onnx --base models/my-base.pt --out my-base-web.onnx
```

Stage the Base directly into the repository:

```bash
python scripts/add_base.py \
  --id my-base \
  --display-name "My Base" \
  --checkpoint models/my-base.pt \
  --web-onnx my-base-web.onnx \
  --license Apache-2.0 \
  --training-license original
```

This creates `bases/my-base/`, copies both binaries, computes SHA-256/byte counts, and writes the Base manifest.

See [`CONTRIBUTING_BASES.md`](CONTRIBUTING_BASES.md).

## Train an Adapter

```bash
orbitune init-adapter adapters/community/my-style-v0 \
  --name my-style-v0 \
  --display-name "My Style"

orbitune train-adapter \
  --base bases/my-base/model.pt \
  --tokens data/tokens/style-train.tokens \
  --validation-tokens data/tokens/style-validation.tokens \
  --validation-interval 50 \
  --out adapters/community/my-style-v0/adapter.safetensors
```

The Safetensors metadata stores the exact Base checkpoint SHA-256. Set the Adapter manifest `base_model` to the selected Base id and `base_sha256` to the same hash. CI rejects an unknown Base id, a Base hash mismatch, or an ABI mismatch.

See [`CONTRIBUTING_ADAPTERS.md`](CONTRIBUTING_ADAPTERS.md).

## Generate MIDI

```bash
orbitune generate \
  --base bases/my-base/model.pt \
  --adapter adapters/community/my-style-v0/adapter.safetensors \
  --bars 8 \
  --temperature 0.85 \
  --bpm 84 \
  --out generated.mid
```

## Registries

The registries are generated from repository contents:

```bash
PYTHONPATH=. python scripts/build_registry.py \
  --bases bases \
  --adapters adapters \
  --base-out registry/bases.json \
  --adapter-out registry/adapters.json
```

`registry/bases.json` describes available Base models. `registry/adapters.json` records each Adapter's Base dependency.

## Browser deployment

GitHub Pages builds both registries. Only `web.onnx` is copied into the static Pages artifact; training checkpoints remain in the Git repository but are not served to the browser.

The UI exposes a Base selector and an Adapter selector. Choosing an Adapter filters to and validates its required Base automatically. The browser verifies the Base ONNX SHA-256 before creating an ONNX Runtime session and verifies the Adapter's embedded `base_sha256` against the selected Base checkpoint hash before applying LoRA matrices.

Current browser graph inputs for the reference ABI:

```text
input_ids    int64    [1, sequence]
lora_a       float32  [4, 2, 4, 240]
lora_b       float32  [4, 2, 240, 4]
lora_scale   float32  [1]
```

## CI

- `test.yml`: Python unit/integration tests
- `web-test.yml`: browser runtime tests
- `ml-smoke.yml`: full 2.95M Base + LoRA CPU training smoke
- `export-smoke.yml`: ONNX export and runtime execution
- `validate-adapters.yml`: Base/Adapter manifest, hash, dependency, and size validation
- `pages.yml`: generated Base/Adapter registries and static browser assets

## License

Orbitune source code is licensed under Apache-2.0. Each contributed Base and Adapter declares its own compatible license and training-data rights status.
