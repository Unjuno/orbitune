# Orbitune

Orbitune is a lightweight MIDI generation framework for local BGM creation. It uses one shared small Base model and small LoRA adapters so contributors can publish generation tendencies without redistributing a full model for every style.

## The compatibility rule that must never be broken

**`orbitune-base` is an immutable Base checkpoint, not a rolling model version.**

Once the official Base is published, its checkpoint bytes are frozen permanently. Community adapters are bound to the exact Base checkpoint by SHA-256. An adapter trained against any other checkpoint is rejected even when the architecture, parameter count, and tokenizer are identical.

```text
orbitune-base.pt
  SHA-256 = H
       │
       ├── chill-piano-v0      base_sha256 = H
       ├── fantasy-town-v0     base_sha256 = H
       └── battle-loop-v0      base_sha256 = H
```

If Orbitune ever develops a genuinely different Base, it must be introduced as a separate compatibility lineage. It must **not** replace `orbitune-base`, and existing adapters remain paired with the original checkpoint forever.

Version strings such as `orbitune-midi-gpt-v0`, `theory-remi-v0`, and `orbitune-lora-v0` refer to protocol/file/runtime ABIs. They do not mean that the Base weights are expected to roll forward.

## Fixed Base architecture

```text
Base model       orbitune-base
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
Compatibility    exact Base checkpoint SHA-256
```

Orbitune is MIDI-only. Raw audio, vocals, audio-codec tokens, and DAW-quality mixing are outside the current scope.

## Quick start

```bash
git clone https://github.com/Unjuno/orbitune.git
cd orbitune
python -m pip install -e .[dev]
python -m pytest -q
orbitune model-info
```

## Prepare a MIDI corpus

Orbitune accepts Standard MIDI File type 0 and type 1. Type-1 note tracks are merged. The current tokenizer is 4/4-oriented and rejects unsupported time signatures rather than silently corrupting bar positions.

Use a file-content split for real experiments:

```bash
orbitune prepare-split-corpus data/raw \
  --train-out data/tokens/train.tokens \
  --validation-out data/tokens/validation.tokens \
  --report data/tokens/split-report.json \
  --validation-fraction 0.1 \
  --min-events 8
```

Identical MIDI bytes are grouped by SHA-256 so renamed duplicates cannot leak across train and validation. Song boundaries are represented with `EOS / BOS`.

## Train the Base candidate

Before the first public Base exists, training may be repeated while selecting the final checkpoint. **No public adapter compatibility is promised during this pre-publication phase.**

```bash
orbitune train-base \
  --tokens data/tokens/train.tokens \
  --validation-tokens data/tokens/validation.tokens \
  --validation-interval 100 \
  --out models/orbitune-base.pt \
  --steps 1000 \
  --batch-size 8 \
  --seq-len 128
```

With periodic validation enabled, Orbitune restores the checkpoint with minimum held-out validation loss instead of blindly publishing the final training step.

The moment the official `orbitune-base.pt` is published, its SHA-256 becomes the permanent compatibility anchor. Its bytes must never be replaced under the same Base identity.

## Train a community Adapter

Create a scaffold:

```bash
orbitune init-adapter adapters/community/my-style-v0 \
  --name my-style-v0 \
  --display-name "My Style"
```

Train against the official Base:

```bash
orbitune train-adapter \
  --base models/orbitune-base.pt \
  --tokens data/tokens/my-style-train.tokens \
  --validation-tokens data/tokens/my-style-validation.tokens \
  --validation-interval 50 \
  --out adapters/community/my-style-v0/adapter.safetensors \
  --steps 500
```

`adapter.safetensors` stores the exact Base SHA-256 automatically. Copy that same hash into the adapter `manifest.json`, add `demo.mid`, complete the license/training-data declaration, then validate.

```bash
orbitune validate-adapter adapters/community/my-style-v0/manifest.json
```

Both Python and browser runtimes reject an adapter when its `base_sha256` differs from the loaded Base checkpoint.

## Generate MIDI

```bash
orbitune generate \
  --base models/orbitune-base.pt \
  --adapter adapters/community/my-style-v0/adapter.safetensors \
  --bars 8 \
  --temperature 0.85 \
  --bpm 84 \
  --out generated.mid
```

The grammar keeps complete note events, supports chords with a bounded number of simultaneous notes, and requires the requested bar structure before `EOS`.

## Browser deployment

One ONNX Base graph is shared by every compatible adapter. LoRA matrices are external graph inputs:

```text
input_ids    int64    [1, sequence]
lora_a       float32  [4, 2, 4, 240]
lora_b       float32  [4, 2, 240, 4]
lora_scale   float32  [1]
```

Export:

```bash
python -m pip install -e '.[export]'
orbitune export-web-onnx \
  --base models/orbitune-base.pt \
  --out orbitune-base-web.onnx
```

The browser verifies two independent identities:

1. `model_sha256`: SHA-256 of the downloaded ONNX bytes.
2. `base_sha256`: SHA-256 of the immutable PyTorch Base checkpoint used to train adapters.

Adapter loading is rejected unless its embedded `base_sha256` equals the runtime Base hash.

## Stage the first immutable Base release

```bash
python scripts/package_base_release.py \
  --base models/orbitune-base.pt \
  --web-onnx orbitune-base-web.onnx \
  --out-dir dist/orbitune-base-release \
  --repository Unjuno/orbitune \
  --release-tag orbitune-base
```

The staging directory contains:

```text
orbitune-base.pt
orbitune-base-web.onnx
orbitune-base-manifest.json
runtime-config.json
```

The manifest records the exact Base checkpoint SHA-256. Once these assets are published, replacing `orbitune-base.pt` under the same identity is prohibited.

## Download the Base safely

```bash
python scripts/download_base_model.py --out models
```

The downloader validates model identity, architecture ABI, tokenizer ABI, parameter count, byte size, and SHA-256 before replacing the destination file.

## Adapter repository policy

Community adapters are committed directly under:

```text
adapters/community/<adapter-id>/
```

Each accepted adapter directory contains:

```text
manifest.json
adapter.safetensors
demo.mid
README.md
```

The manifest and Safetensors metadata must contain the same `base_sha256`. The registry additionally rejects a repository state that mixes adapters for multiple Base hashes.

Base weights and large training data remain outside Git history.

## CI

- `test.yml`: Python unit/integration tests
- `web-test.yml`: browser runtime, MIDI, Safetensors, SHA verification
- `ml-smoke.yml`: full 2.95M Base + LoRA CPU training smoke
- `export-smoke.yml`: ONNX export, ORT execution, immutable Base release staging
- `validate-adapters.yml`: manifest, weight, demo, registry and size checks
- `pages.yml`: static browser deployment

## License

Orbitune source code is Apache-2.0. Every community adapter must separately declare its adapter license and training-data rights status.
