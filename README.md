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

## Current reference architecture

```text
Parameters       10,200,960 with the current 204-token vocabulary
Layers           4
Hidden size      448
Attention heads  7
Head dimension   64
Context          1024 tokens
Architecture ABI orbitune-midi-gpt-v0
Tokenizer ABI    theory-remi-v0
LoRA rank        4
LoRA targets     q_proj + v_proj
Adapter format   Safetensors / orbitune-lora-v0
```

The 448-wide architecture was chosen so a substantially larger MIDI tokenizer can be introduced without moving far beyond the 10M parameter class. The tokenizer is still under active design; no public Base should be frozen before that work and dataset research are complete.

Orbitune is MIDI-only. Raw audio, vocals, audio-codec tokens, and DAW-quality mixing are outside the current scope.

## Repository layout

```text
bases/<base-id>/      accepted immutable Base checkpoints and Web ONNX files
adapters/             official and community LoRA Adapters
models/               local/training candidate outputs
registry/             generated Base and Adapter dependency registries
data/continuous/      dataset gate for scheduled continuous training
web/                  local browser runtime / GitHub Pages app
```

The repository policy permits compact contributed Bases up to 100M parameters, subject to the repository binary-size policy. Existing Base ids are immutable once accepted.

## Quick start

```bash
git clone https://github.com/Unjuno/orbitune.git
cd orbitune
python -m pip install -e .[dev]
python -m pytest -q
orbitune model-info
```

## Prepare MIDI data

```bash
orbitune prepare-split-corpus data/raw \
  --train-out data/tokens/train.tokens \
  --validation-out data/tokens/validation.tokens \
  --report data/tokens/split-report.json \
  --validation-fraction 0.1 \
  --min-events 8
```

Identical MIDI bytes are grouped by SHA-256 so renamed duplicates cannot leak across train and validation. Dataset provenance and rights must be reviewed before the corpus is used for the official Base.

## Train the 10M reference Base

```bash
orbitune train-base \
  --tokens data/tokens/train.tokens \
  --validation-tokens data/tokens/validation.tokens \
  --validation-interval 100 \
  --out models/my-base.pt \
  --steps 1000 \
  --batch-size 8 \
  --seq-len 256
```

With periodic validation enabled, Orbitune restores the checkpoint with minimum held-out validation loss instead of blindly publishing the final step.

## Continuous GitHub Actions training

`.github/workflows/continuous-train.yml` is scheduled every six hours at minute 17. It is intentionally idle until both files exist:

```text
data/continuous/train.tokens
data/continuous/validation.tokens
```

Once the dataset gate is armed, each run restores the previous model **and AdamW optimizer state**, trains the 10M Base for up to five hours, validates periodically, and persists state in two places:

1. Actions cache for fast restoration.
2. The mutable `continuous-training` prerelease attached to this repository for durable recovery.

Continuous state is separated by purpose:

```text
state.pt       latest resumable model + optimizer + RNG state
healthy.pt     last health-confirmed rollback point
best.pt        model checkpoint with best held-out validation loss
report.json    health metrics, tokens seen, spikes and snapshot records
snapshots/     full immutable training states created by token milestones
```

A full training snapshot is created every **10,000,000 tokens seen**. Release asset names additionally include the training step and GitHub Actions run id so a later rollback/retrain cannot overwrite an older forensic snapshot.

The health monitor records training loss, rolling loss z-score, pre-clipping gradient norm, validation loss, spike events and cumulative tokens. A single difficult batch does not trigger rollback. Orbitune rolls back to `healthy.pt` when it encounters a non-finite loss/gradient/validation value, persistent loss spikes, or a loss spike accompanied by a severe gradient spike. The state and forensic report are persisted first; the workflow then fails visibly so the anomaly cannot pass silently.

The prerelease is training state only. It is not a published Base compatibility target. Final accepted Bases still enter `bases/<base-id>/` with immutable checkpoint bytes and SHA-256 identities.

`continuous-smoke.yml` verifies actual continuation by training one step, serializing model/optimizer/RNG state, restoring it, advancing another step, and checking cumulative `tokens_seen`, token snapshots and the healthy checkpoint.

## Export and contribute a Base

```bash
python -m pip install -e '.[export]'
orbitune export-web-onnx --base models/my-base.pt --out my-base-web.onnx

python scripts/add_base.py \
  --id my-base \
  --display-name "My Base" \
  --checkpoint models/my-base.pt \
  --web-onnx my-base-web.onnx \
  --license Apache-2.0 \
  --training-license original
```

This creates `bases/my-base/`, computes artifact hashes and writes the Base manifest. See [`CONTRIBUTING_BASES.md`](CONTRIBUTING_BASES.md).

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

The current Adapter ABI targets the 10M reference shape: four 448-wide layers, rank-4 LoRA on `q_proj` and `v_proj`. Adapter metadata stores the exact Base checkpoint SHA-256. See [`CONTRIBUTING_ADAPTERS.md`](CONTRIBUTING_ADAPTERS.md).

## Browser deployment

The current Web graph inputs are:

```text
input_ids    int64    [1, sequence]
lora_a       float32  [4, 2, 4, 448]
lora_b       float32  [4, 2, 448, 4]
lora_scale   float32  [1]
```

The browser context limit is 1024 tokens. Base ONNX bytes are verified by SHA-256 before loading; Adapter Base hashes are checked before LoRA application.

## CI

- `test.yml`: Python unit/integration tests
- `web-test.yml`: browser runtime tests
- `ml-smoke.yml`: full 10.2M Base + LoRA CPU training smoke
- `continuous-smoke.yml`: resumable model/optimizer state, tokens-seen, health-state and snapshot smoke
- `continuous-train.yml`: six-hour scheduled continuation loop with spike monitoring, automatic rollback and 10M-token snapshots once real data is configured
- `export-smoke.yml`: ONNX export and runtime execution
- `validate-adapters.yml`: Base/Adapter manifest, hash, dependency and size validation
- `pages.yml`: generated Base/Adapter registries and static browser assets

## License

Orbitune source code is licensed under Apache-2.0. Each contributed Base and Adapter declares its own compatible license and training-data rights status.
