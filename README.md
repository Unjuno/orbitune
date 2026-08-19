# Orbitune

Orbitune is a lightweight MIDI generation framework for local BGM creation.

It uses one shared small MIDI Base model and small LoRA adapters to generate style-specific background music. The project focuses on symbolic MIDI generation, contributor-trainable adapters, local inference, and a smartphone-oriented browser UI.

Orbitune v0 is MIDI-only. It does not generate raw audio waveforms, audio-codec tokens, vocals, mixes, or environmental sound directly.

## Project goals

- Provide one fixed MIDI Base: `orbitune-tiny-v0`.
- Keep the Base small enough for container training experiments and smartphone-oriented inference.
- Keep community style packs as small rank-4 LoRA Safetensors files.
- Let contributors train, validate, and commit compatible adapters directly to the repository.
- Keep Base weights out of Git history.
- Generate MIDI locally with Adapter, BPM, Length, and Temperature as the primary UI controls.

## v0 compatibility contract

```text
Base model       orbitune-tiny-v0
Architecture     orbitune-midi-gpt-v0
Parameters       2,945,760
Layers           4
Hidden size      240
Attention heads  4
Context          512 tokens
Tokenizer        theory-remi-v0 (204 tokens)
LoRA rank        4
LoRA targets     q_proj + v_proj
Adapter format   Safetensors
```

The v0 contract is intentionally narrow. An adapter with a different Base, tokenizer, hidden size, rank, or LoRA target set is not v0-compatible.

## Non-goals

- Raw audio generation
- Audio codec token generation
- Vocal generation
- Full DAW-quality mixing
- Multi-instrument orchestration as the v0 target
- Multiple LoRA composition in v0
- WebGPU-only runtime assumptions

## Quick start

```bash
git clone https://github.com/Unjuno/orbitune.git
cd orbitune
python -m pip install -e .[dev]
python -m pytest -q
orbitune model-info
```

Generate a deterministic demo MIDI and roundtrip it through Theory-REMI:

```bash
orbitune generate-demo --out examples/generated/demo.mid --bars 4 --bpm 84
orbitune tokenize examples/generated/demo.mid --out examples/generated/demo.tokens
orbitune detokenize examples/generated/demo.tokens --out examples/generated/demo_roundtrip.mid --bpm 84
orbitune inspect examples/generated --out examples/generated/inspect.json
orbitune eval-midi examples/generated/demo.mid --out examples/generated/demo-eval.json
```

## Prepare a real MIDI corpus

Orbitune v0 accepts Standard MIDI File type 0 and type 1. Type 1 note tracks are merged. By default corpus preparation rejects unsupported time signatures so the current 16-position-per-4/4-bar representation is not silently corrupted.

For real experiments, split at the **MIDI-file content level**, not the token level. Duplicate MIDI bytes are grouped by SHA-256 so renamed copies cannot leak across train and validation:

```bash
orbitune prepare-split-corpus data/raw \
  --train-out data/tokens/train.tokens \
  --validation-out data/tokens/validation.tokens \
  --report data/tokens/split-report.json \
  --validation-fraction 0.1 \
  --min-events 8
```

Song boundaries are preserved with `EOS / BOS` tokens. Bad or unsupported files are recorded in the report rather than aborting the whole conversion.

For a single corpus without a validation split:

```bash
orbitune prepare-corpus data/raw \
  --out data/tokens/base.tokens \
  --report data/tokens/base-report.json \
  --min-events 8
```

## Train the Base

```bash
orbitune train-base \
  --tokens data/tokens/train.tokens \
  --validation-tokens data/tokens/validation.tokens \
  --out models/orbitune-tiny-v0.pt \
  --steps 1000 \
  --batch-size 8 \
  --seq-len 128
```

Training reports include training loss, held-out validation loss when supplied, elapsed time, processed-token count, and token throughput. The Base checkpoint is ignored by Git and is distributed separately as a release asset.

## Train a community Adapter

Create the directory scaffold first:

```bash
orbitune init-adapter adapters/community/my-style-v0 \
  --name my-style-v0 \
  --display-name "My Style"
```

Then train the fixed rank-4 adapter:

```bash
orbitune train-adapter \
  --base models/orbitune-tiny-v0.pt \
  --tokens data/tokens/my-style-train.tokens \
  --validation-tokens data/tokens/my-style-validation.tokens \
  --out adapters/community/my-style-v0/adapter.safetensors \
  --steps 500
```

Complete `manifest.json`, add `demo.mid`, confirm the training-data rights declaration, and validate before committing.

## Generate MIDI

```bash
orbitune generate \
  --base models/orbitune-tiny-v0.pt \
  --adapter adapters/community/my-style-v0/adapter.safetensors \
  --bars 8 \
  --temperature 0.85 \
  --bpm 84 \
  --out generated.mid
```

The generation grammar enforces complete note events and the requested number of bars before `EOS`. Positions are nondecreasing inside a bar: the same position may repeat for a chord, up to eight distinct pitches, while lower positions cannot reappear. Each completed bar must reach the final quarter of the bar, preventing an 8-bar request from collapsing into eight one-note fragments.

## Reproducible CPU smoke training

The full fixed architecture can be exercised without reducing the model size:

```bash
python scripts/smoke_train.py \
  --base-steps 100 \
  --adapter-steps 100 \
  --device cpu \
  --out smoke-training-report.json
```

Measured container results and limitations are documented in [`docs/CONTAINER_TRAINING.md`](docs/CONTAINER_TRAINING.md). The synthetic smoke corpus verifies the training pipeline and compute scale; it is **not** evidence of music quality.

## Browser deployment contract

Orbitune keeps one Base model and passes LoRA matrices as external runtime inputs. Adapters therefore remain small instead of requiring a separate Base export for every style.

Install optional export dependencies and export the browser graph:

```bash
python -m pip install -e '.[export]'
orbitune export-web-onnx \
  --base models/orbitune-tiny-v0.pt \
  --out orbitune-tiny-v0-web.onnx
```

Browser graph inputs:

```text
input_ids    int64    [1, sequence]
lora_a       float32  [4, 2, 4, 240]
lora_b       float32  [4, 2, 240, 4]
lora_scale   float32  [1]
```

`web/orbitune-runtime.mjs` packs a v0 Adapter Safetensors file into those inputs and produces MIDI bytes after grammar-constrained sampling. WASM is the default browser execution provider; acceleration can be evaluated separately after the baseline works.

## Stage a verified Base release

After training the real Base and exporting its browser ONNX graph, create a release staging directory:

```bash
python scripts/package_base_release.py \
  --base models/orbitune-tiny-v0.pt \
  --web-onnx orbitune-tiny-v0-web.onnx \
  --out-dir dist/orbitune-tiny-v0-release \
  --repository Unjuno/orbitune \
  --release-tag orbitune-tiny-v0
```

The staging directory contains:

```text
orbitune-tiny-v0.pt
orbitune-tiny-v0-web.onnx
orbitune-tiny-v0-manifest.json
runtime-config.json
```

The packager refuses a checkpoint that is not the fixed 2,945,760-parameter v0 architecture. The manifest records the exact checkpoint and ONNX byte sizes, SHA-256 hashes, and tag-specific GitHub Release URLs. `runtime-config.json` carries the ONNX URL and hash expected by the browser.

After uploading the first three files to the matching GitHub Release tag, copy the generated `runtime-config.json` into `web/runtime-config.json`. The browser downloads the ONNX bytes, verifies SHA-256 with Web Crypto, and only then creates the ONNX Runtime session.

Until the official asset exists, the committed `web/runtime-config.json` intentionally keeps `model_url` and `model_sha256` empty, so generation remains disabled rather than silently loading an unversioned model.

## Download a published Base safely

Once the release exists, the default downloader reads the latest release manifest and verifies the artifact hash before replacing the destination file:

```bash
python scripts/download_base_model.py --out models
```

A specific manifest can also be used for reproducibility:

```bash
python scripts/download_base_model.py \
  --manifest path/to/orbitune-tiny-v0-manifest.json \
  --artifact checkpoint \
  --out models
```

Use `--artifact web_onnx` to fetch the browser graph. Remote manifests and artifact URLs must use HTTPS.

## Adapter registry and Pages

Contributor adapters are committed directly under:

```text
adapters/official/<adapter-id>/
adapters/community/<adapter-id>/
```

Each accepted adapter directory contains:

```text
manifest.json
adapter.safetensors
demo.mid
README.md
```

The registry is generated from these directories rather than edited manually:

```bash
PYTHONPATH=. python scripts/build_registry.py --adapters adapters --out registry/adapters.json
```

The GitHub Pages workflow uses the same builder and copies bundled adapter assets into the static site artifact automatically.

## Adapter validation

```bash
orbitune validate-adapter adapters/community/my-style-v0/manifest.json
orbitune package-adapter adapters/community/my-style-v0 \
  --manifest adapters/community/my-style-v0/manifest.json \
  --out my-style-v0.zip
```

## CI layers

- `test.yml`: Python unit/integration tests
- `web-test.yml`: Node tests for browser grammar, sampling, MIDI generation, Safetensors packing, and model SHA-256 verification
- `ml-smoke.yml`: full-size Base + LoRA CPU training smoke
- `export-smoke.yml`: actual ONNX export, ONNX Runtime dynamic-sequence inference, and release-package staging
- `validate-adapters.yml`: bundled community/official adapter validation
- `pages.yml`: static site and adapter asset build/deploy

Python CI installs the CPU-only PyTorch wheel explicitly before the Orbitune package to avoid downloading unused CUDA runtime packages on CPU GitHub runners.

## Current command surface

```text
orbitune generate-demo
orbitune inspect
orbitune tokenize
orbitune prepare-corpus
orbitune prepare-split-corpus
orbitune detokenize
orbitune eval-midi
orbitune init-adapter
orbitune model-info
orbitune train-base
orbitune train-adapter
orbitune generate
orbitune export-onnx
orbitune export-web-onnx
orbitune validate-adapter
orbitune package-adapter
```

Release staging and verified downloading currently live under `scripts/` because they operate on repository/release artifacts rather than the core generation API.

## Repository policy

Base model weights are not committed to this repository. Community adapters are intentionally small and may be committed directly when they satisfy the compatibility, metadata, rights, and CI checks.

## License

The Orbitune source code is licensed under Apache-2.0. Community adapters may use their own compatible licenses, but every adapter must declare its license and training-data status.
