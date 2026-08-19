# Orbitune

Orbitune is a lightweight MIDI generation framework for local BGM creation.

It uses a shared small MIDI base model and small LoRA adapters to generate style-specific background music. The project focuses on symbolic MIDI generation, adapter training, adapter validation, and a simple web UI for selecting adapters and generating MIDI.

Orbitune v0 is MIDI-only. It does not generate raw audio waveforms, audio-codec tokens, vocals, mixes, or environmental sound directly.

## Project goals

- Provide a fixed ~3M-parameter MIDI base model target: `orbitune-tiny-v0`.
- Support small LoRA adapters as style, genre, mood, or texture-like BGM packs.
- Let contributors train, validate, and commit compatible adapters.
- Keep the base model weights out of the repository.
- Provide a smartphone-oriented web UI with adapter selection, BPM, length, and temperature.

## v0 scope

- Symbolic MIDI generation only
- Theory-REMI v0 token format
- Piano/BGM-oriented generation
- Fixed 2,945,760-parameter decoder-only Transformer (`4 layers × 240 hidden`, 4 heads, context 512)
- LoRA on `q_proj` and `v_proj`
- Safetensors adapter files
- Base + one LoRA adapter at a time
- 4 to 8 bar generation as the default UX
- GitHub-first distribution

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

Generate a deterministic demo MIDI file and roundtrip it through Theory-REMI tokens:

```bash
orbitune generate-demo --out examples/generated/demo.mid --bars 4 --bpm 84
orbitune tokenize examples/generated/demo.mid --out examples/generated/demo.tokens
orbitune detokenize examples/generated/demo.tokens --out examples/generated/demo_roundtrip.mid --bpm 84
orbitune inspect examples/generated --out examples/generated/inspect.json
orbitune eval-midi examples/generated/demo.mid --out examples/generated/demo-eval.json
```

## Prepare a MIDI corpus

For a directory of MIDI files, build one training token corpus and a data-quality report:

```bash
orbitune prepare-corpus data/raw \
  --out data/tokens/base.tokens \
  --report data/tokens/base-report.json \
  --min-events 8
```

Bad or unsupported files are recorded in the report instead of aborting the entire corpus conversion.

## Train a base model

```bash
orbitune train-base \
  --tokens data/tokens/base.tokens \
  --out models/orbitune-tiny-v0.pt \
  --steps 1000 \
  --batch-size 8 \
  --seq-len 128
```

Training reports include loss, elapsed time, processed-token count, and token throughput. The base checkpoint is intentionally ignored by Git and should be distributed as a release asset.

## Train a LoRA adapter

```bash
orbitune train-adapter \
  --base models/orbitune-tiny-v0.pt \
  --tokens data/tokens/my-style.tokens \
  --out adapters/community/my-style-v0/adapter.safetensors \
  --rank 4 \
  --steps 500
```

Adapters target `q_proj` and `v_proj`. Training seeds exist for reproducible experiments, but seed selection is not part of the v0 web UI.

## Generate MIDI

```bash
orbitune generate \
  --base models/orbitune-tiny-v0.pt \
  --adapter adapters/community/my-style-v0/adapter.safetensors \
  --temperature 0.85 \
  --bpm 84 \
  --out generated.mid
```

Generation uses a grammar constraint so the model emits `BAR / POSITION / PITCH / DURATION / VELOCITY` sequences that can be converted back into MIDI.

## Reproducible CPU smoke training

The full fixed architecture can be exercised without reducing the model size:

```bash
python scripts/smoke_train.py \
  --base-steps 100 \
  --adapter-steps 100 \
  --device cpu \
  --out smoke-training-report.json
```

Measured container results and limitations are documented in [`docs/CONTAINER_TRAINING.md`](docs/CONTAINER_TRAINING.md). This smoke test uses synthetic grammar-valid patterns and therefore verifies the pipeline and compute scale, not music quality.

## Adapter validation

```bash
orbitune validate-adapter path/to/manifest.json
orbitune package-adapter path/to/adapter_dir --manifest path/to/manifest.json --out adapter.zip
```

Small compatible adapters may be committed directly under `adapters/official/` or `adapters/community/`, provided that they include a manifest, README, demo MIDI, license declaration, and pass validation.

## Current command surface

```text
orbitune generate-demo
orbitune inspect
orbitune tokenize
orbitune prepare-corpus
orbitune detokenize
orbitune eval-midi
orbitune model-info
orbitune train-base
orbitune train-adapter
orbitune generate
orbitune validate-adapter
orbitune package-adapter
```

## Repository policy

Base model weights are not committed to this repository. Use `scripts/download_base_model.py` once an official release asset is available.

Community adapters are intentionally small and may be committed directly when they satisfy the repository policy and compatibility checks.

## License

The Orbitune source code is licensed under Apache-2.0. Community adapters may use their own compatible licenses, but every adapter must declare its license and training-data status.
