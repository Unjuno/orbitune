# Orbitune

Orbitune is a lightweight, local-first symbolic MIDI generation framework built around a compact shared Base model and distributable LoRA Adapters. Contributors can publish adapters against immutable Base checkpoint hashes without breaking existing compatibility lineages.

> **Development continuation:** read [`docs/AUDIT_2026-08-23.md`](docs/AUDIT_2026-08-23.md) and [`docs/HANDOFF.md`](docs/HANDOFF.md) before resuming architecture work. For remote GPU execution, also read [`workloads/runpod-training-canary/README.md`](workloads/runpod-training-canary/README.md).

## Compatibility model

A Base is identified by stable `base-id` plus exact checkpoint SHA-256. An Adapter targets exactly one Base checkpoint. A Base id is not a mutable rolling slot: changed checkpoint bytes require a new Base id.

## Current reference design

Orbitune is moving from the operational flat `theory-remi-v0` reference path toward **Hybrid Compound Events**: one musical event consumes one causal Transformer step and attributes are predicted by small factorized heads.

```text
Model family          causal decoder-only Transformer
Base size             ~10M reference; real-MIDI 5M/10M/20M sweep still required
Representation        Hybrid Compound Events
Temporal grid         96 steps / quarter note candidate
NOTE decoder          lightweight intra-event autoregressive MLP cascade
DELTA/DURATION        7 coarse ranges + 16 residual levels candidate
Continuous values     factorized coarse + residual heads
Long context          recent dense window + deterministic historical anchors
LoRA                  runtime adapter over immutable Base
Deployment            packed ternary candidate; INT8 / FP16 fallbacks
```

The currently implemented 10.2M / 204-token `orbitune-midi-gpt-v0` + `theory-remi-v0` model remains a **legacy/reference implementation**, not the frozen production ABI. Do not publish immutable community Bases against `orbitune-compound-v0-experimental` until the audit blockers and real-data gates close.

## Compound MIDI scope

Current production-schema candidate:

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

MIDI ingestion is canonicalized deterministically. Same-onset/channel/pitch duplicate notes are merged, overlapping same-channel/pitch notes are truncated on retrigger, unused event fields are forced to canonical zero values, and same-step metadata/control events precede NOTE so the causal model sees applicable state first.

The Compound schema is **semantic rather than lossless SMF serialization**. Known open items include timing beyond 1536 steps, BOS/EOS/start semantics, half-pedal representation, production deduplication, and some rare/meta MIDI semantics.

## Implemented Compound data path

```text
MIDI
→ orbitune.compound_midi
→ CompoundEvent
→ factorized CompoundRecord
→ ABI-tagged song-preserving JSONL
→ validated fixed-length training windows
→ NEXT: Compound Base + event-conditioned masked losses
```

The Transformer invariant remains `1 musical event = 1 Transformer step`.

## Repository layout

```text
bases/<base-id>/      immutable accepted Base artifacts + manifests
adapters/             official/community LoRA Adapters
models/               local candidate checkpoints
registry/             generated Base/Adapter dependency registries
data/continuous/      explicitly gated legacy scheduled-training data
experiments/          reproducible design/tokenizer experiments
workloads/            bounded external-compute workloads
docs/                 audit, handoff, design status and research plans
web/                  local browser runtime / GitHub Pages app
```

## Quick start

```bash
git clone https://github.com/Unjuno/orbitune.git
cd orbitune
python -m pip install -e .[dev]
python -m pytest -q
orbitune model-info
```

## Prepare MIDI data

Legacy/reference tokenizer path:

```bash
orbitune prepare-split-corpus data/raw \
  --train-out data/tokens/train.tokens \
  --validation-out data/tokens/validation.tokens \
  --report data/tokens/split-report.json \
  --validation-fraction 0.1 \
  --min-events 8
```

Experimental Compound preparation uses `scripts/prepare_compound_corpus.py`. Keep its ABI-tagged JSONL separate from legacy `.tokens`. Exact byte duplicates are grouped by SHA-256, but production near-duplicate/composition-aware grouping is still required.

## Train the current 10M reference implementation

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

This trains the operational Theory-REMI reference only. It does not freeze the future Compound ABI.

## First remote GPU workload

Orbitune now contains a bounded infrastructure canary aligned with `Unjuno/gpu-control`:

```text
workloads/runpod-training-canary/
```

Default paid-canary work:

```text
10.2M reference Base
250 optimizer steps
batch 8
sequence 256
512,000 training tokens
validation every 50 steps
synthetic deterministic data
/outputs/result.json
/outputs/canary-base.pt
```

The canary is deliberately small and is **not** a musical-quality benchmark. It exists to prove CUDA visibility, forward/backward, optimizer state, validation, checkpointing, bounded result collection and provider cleanup before Compound or corpus-scale training is attempted.

A CPU one-step contract runs in `.github/workflows/runpod-canary-smoke.yml`. A paid GPU canary is accepted only if the result reports CUDA execution, exactly 512k processed training tokens, finite/improving validation, matching checkpoint SHA-256, finalized provider lifecycle and confirmed cleanup.

See [`workloads/runpod-training-canary/README.md`](workloads/runpod-training-canary/README.md) for the exact `gpu-control verify-source` handoff and acceptance gates.

## Continuous GitHub Actions training

`.github/workflows/continuous-train.yml` remains **legacy Theory-REMI reference training only** and is intentionally idle unless all three files are present and non-empty:

```text
data/continuous/ENABLE_LEGACY_REFERENCE_TRAINING
data/continuous/train.tokens
data/continuous/validation.tokens
```

The state preserves model, optimizer, RNG, health/spike history and milestone snapshots. Mutable continuous-training state is not a Base compatibility target. Compound continuous training will use a separate gate after the Compound checkpoint ABI exists.

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
  --training-license original \
  --rights-confirmed
```

`--rights-confirmed` is an explicit acknowledgement, not proof by itself. Reviewers must still verify provenance. The current public staging tool targets the operational legacy/reference ABI. See [`CONTRIBUTING_BASES.md`](CONTRIBUTING_BASES.md).

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

The current Adapter ABI is four 448-wide layers with rank-4 LoRA on `q_proj` and `v_proj`. Future Compound Bases need a separate frozen architecture/tokenizer/runtime/Adapter contract.

## Browser deployment

Current legacy Web graph inputs:

```text
input_ids    int64    [1, sequence]
lora_a       float32  [4, 2, 4, 448]
lora_b       float32  [4, 2, 448, 4]
lora_scale   float32  [1]
```

These are not the future Compound Web ABI. Base ONNX bytes and Adapter/Base hashes are verified before use.

## Current blocking validations

1. long DELTA/DURATION representation beyond 1536 steps;
2. BOS/EOS/start semantics and one explicit field-cardinality/mask schema;
3. external real-MIDI validation of 96/qn and 7+16 timing;
4. production provenance + near-duplicate/composition-aware deduplication;
5. Compound Base synthetic/tiny-real overfit;
6. real-MIDI 5M/10M/20M scale sweep;
7. trained long-memory and ControlField validation;
8. actual-device INT8 vs packed-ternary benchmark;
9. Base-rollout post-training necessity gate; SFT/DPO/RL remain optional and evidence-gated.

## CI

- `test.yml`: Python unit/integration tests including Compound tests
- `runpod-canary-smoke.yml`: one-step CPU contract for the remote-GPU workload
- `web-test.yml`: legacy browser runtime tests
- `ml-smoke.yml`: legacy 10.2M Base + LoRA CPU training smoke
- `continuous-smoke.yml`: resumable legacy training state/health/snapshot smoke
- `continuous-train.yml`: explicitly gated legacy scheduled training
- `export-smoke.yml`: legacy ONNX export/runtime/Base staging smoke
- `validate-adapters.yml`: Base/Adapter manifests, checkpoint consistency, hashes and size policy
- `pages.yml`: generated registries and browser assets

## License

Orbitune source code is licensed under Apache-2.0. Each contributed Base and Adapter declares its own compatible license and training-data rights status.
