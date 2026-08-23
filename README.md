# Orbitune

Orbitune is a lightweight, local-first symbolic MIDI generation framework built around a compact shared Base model and distributable LoRA Adapters. Contributors can publish new adapters against immutable Base checkpoints without breaking existing compatibility lineages.

> **Development continuation:** read [`docs/AUDIT_2026-08-23.md`](docs/AUDIT_2026-08-23.md) and [`docs/HANDOFF.md`](docs/HANDOFF.md) first when resuming implementation. They record current risks, implemented Compound pipeline, unresolved experiment gates, and decisions that should not be reopened without contradictory evidence.

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

## Current reference design

Orbitune's production representation is moving from the earlier flat `theory-remi-v0` prototype toward **Hybrid Compound Events**: one musical event consumes one causal Transformer step, while event attributes are predicted by small factorized heads.

Current reference decisions:

```text
Model family          causal decoder-only Transformer
Base size             ~10M parameters (reference; 5M/10M/20M real-MIDI sweep still required)
Representation        Hybrid Compound Events
Temporal grid         96 steps / quarter note (candidate pending external real-MIDI validation)
NOTE decoder          lightweight intra-event autoregressive MLP cascade
DELTA/DURATION        7 coarse ranges + 16 residual levels (candidate)
Continuous values     factorized coarse + residual heads
Long context          recent dense window + deterministic historical anchors
LoRA                  runtime adapter over an immutable Base
Deployment            packed ternary candidate; INT8 and FP16 fallbacks
```

The currently implemented 10.2M / 204-token `theory-remi-v0` model remains a **legacy/reference implementation**, not the frozen production ABI. Do not publish a final Base against the experimental tokenizer until the external real-MIDI validation gates in [`docs/DESIGN_STATUS.md`](docs/DESIGN_STATUS.md) and audit blockers are closed.

The experimental Compound data path is now implemented through MIDI parsing, factorized event records, ABI-tagged song-preserving JSONL preparation, and fixed-length training-window loading. The next critical implementation is the Compound Base model and masked factorized losses.

Orbitune is MIDI-only. Raw audio, vocals, audio-codec tokens, and DAW-quality mixing are outside the current scope.

## Compound MIDI scope

The current production schema candidate covers:

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

MIDI ingestion is canonicalized deterministically before tokenization: same-onset/channel/pitch duplicate notes are merged, and an overlapping same-channel/same-pitch note is truncated when that pitch is retriggered. Same-step metadata/control events are ordered before NOTE events so the causal model sees applicable state first.

Continuous MIDI values and timing are factorized rather than represented by huge flat vocabularies. This keeps `1 event = 1 Transformer step` while retaining fine-grained reconstruction.

The current Compound schema is **semantic rather than lossless SMF serialization**. Known open items include long timing values beyond 1536 steps, BOS/EOS/start semantics, half-pedal representation, and some rare/meta MIDI semantics; see the audit document.

## Repository layout

```text
bases/<base-id>/      accepted immutable Base checkpoints and Web ONNX files
adapters/             official and community LoRA Adapters
models/               local/training candidate outputs
registry/             generated Base and Adapter dependency registries
data/continuous/      explicitly gated legacy scheduled-training data
experiments/          reproducible architecture/tokenizer evaluation scripts
docs/                 audit, accepted/candidate/open decisions and handoff
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

The legacy/reference tokenizer path remains available:

```bash
orbitune prepare-split-corpus data/raw \
  --train-out data/tokens/train.tokens \
  --validation-out data/tokens/validation.tokens \
  --report data/tokens/split-report.json \
  --validation-fraction 0.1 \
  --min-events 8
```

For the experimental Compound path, use `scripts/prepare_compound_corpus.py` and keep its ABI-tagged JSONL output separate from legacy `.tokens` files. Identical MIDI bytes are grouped by SHA-256 so renamed duplicates cannot leak across train and validation. This is **exact duplicate protection only**; production near-duplicate/composition-aware grouping remains a validation gate.

Dataset provenance and rights must be reviewed before the corpus is used for an official Base.

## Train the current 10M reference implementation

The command below trains the currently implemented Theory-REMI reference model. It does **not** imply that the legacy tokenizer ABI is frozen.

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

`.github/workflows/continuous-train.yml` is scheduled every six hours at minute 17 but remains **legacy Theory-REMI reference training only**. To prevent accidental obsolete training, it stays idle unless all three files exist and are non-empty:

```text
data/continuous/ENABLE_LEGACY_REFERENCE_TRAINING
data/continuous/train.tokens
data/continuous/validation.tokens
```

Do not add the marker merely to make CI run. The corpora must already pass provenance/license, quality, deduplication, and split review.

Once explicitly armed, each run restores the previous model **and AdamW optimizer state**, validates periodically, and persists state in Actions cache and the mutable `continuous-training` prerelease.

Continuous state is separated by purpose:

```text
state.pt       latest resumable model + optimizer + RNG state
healthy.pt     last health-confirmed rollback point
best.pt        model checkpoint with best held-out validation loss
report.json    health metrics, tokens seen, spikes and snapshot records
snapshots/     full immutable training states created by token milestones
```

A full legacy-reference training snapshot is created every **10,000,000 tokens seen**. The prerelease is training state only; it is not a published Base compatibility target. Compound continuous training will use a separate gate after the Compound model/checkpoint ABI exists.

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

This current contribution path targets the operational legacy/reference ABI. Do not publish immutable community Bases against `orbitune-compound-v0-experimental`. See [`CONTRIBUTING_BASES.md`](CONTRIBUTING_BASES.md).

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

The currently implemented Adapter ABI targets the existing reference shape: four 448-wide layers, rank-4 LoRA on `q_proj` and `v_proj`. Future accepted Compound Base checkpoints must declare their exact architecture/tokenizer/Adapter ABI and hash so adapters remain deterministic. See [`CONTRIBUTING_ADAPTERS.md`](CONTRIBUTING_ADAPTERS.md).

## Browser deployment

The current legacy Web graph inputs are:

```text
input_ids    int64    [1, sequence]
lora_a       float32  [4, 2, 4, 448]
lora_b       float32  [4, 2, 448, 4]
lora_scale   float32  [1]
```

These describe the current reference implementation, not the final Compound Event Web ABI. Base ONNX bytes are verified by SHA-256 before loading; Adapter Base hashes are checked before LoRA application.

## Design validation status

Architecture ideation is largely complete. The remaining blocking validations are empirical and ABI-specific:

1. external real-MIDI validation of 96/qn and the 7+16 timing factorization;
2. long DELTA/DURATION representation beyond the current 1536-step experimental range;
3. BOS/EOS/start semantics plus explicit field cardinalities and masks;
4. production corpus composition, provenance filtering and near-duplicate/composition-aware deduplication;
5. real-MIDI 5M/10M/20M scale sweep;
6. real-device/Web INT8 vs packed-ternary benchmark;
7. trained-model long-rollout validation of anchored memory;
8. real-corpus ControlField adherence/quantization validation;
9. Base-rollout post-training necessity gate; policy learning remains optional and experiment-gated.

See [`docs/AUDIT_2026-08-23.md`](docs/AUDIT_2026-08-23.md), [`docs/DESIGN_STATUS.md`](docs/DESIGN_STATUS.md), [`docs/POST_TRAINING_RESEARCH.md`](docs/POST_TRAINING_RESEARCH.md), and [`docs/HANDOFF.md`](docs/HANDOFF.md).

## CI

- `test.yml`: Python unit/integration tests, including Compound tests
- `web-test.yml`: legacy browser runtime tests
- `ml-smoke.yml`: full legacy 10.2M Base + LoRA CPU training smoke
- `continuous-smoke.yml`: legacy resumable model/optimizer state, tokens-seen, health-state and snapshot smoke
- `continuous-train.yml`: explicitly gated legacy scheduled continuation loop
- `export-smoke.yml`: legacy ONNX export and runtime execution
- `validate-adapters.yml`: Base/Adapter manifest, hash, dependency and size validation
- `pages.yml`: generated Base/Adapter registries and static browser assets

## License

Orbitune source code is licensed under Apache-2.0. Each contributed Base and Adapter declares its own compatible license and training-data rights status.
