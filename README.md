# Orbitune

Orbitune is a local-first symbolic MIDI generation framework. The repository currently contains two runnable model paths:

- **Compound Hierarchical Base** — the current Transformer-first path for new training and local MIDI generation.
- **Theory-REMI reference Base** — the older ~10M compatibility/reference path used by the existing Base/Adapter ecosystem.

The two paths intentionally coexist. Existing Theory-REMI checkpoints and adapter tooling are not replaced by the Compound implementation.

## Local Compound Base quick start

```bash
git clone https://github.com/Unjuno/orbitune.git
cd orbitune
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
orbitune-compound inspect --config configs/compound_hierarchical_9m.json
```

The default Compound model is an old-Base-scale model, not the small proxy models used during architecture experiments.

### 1. Prepare MIDI

Place MIDI files under `data/raw`, then create song-preserving Compound JSONL splits:

```bash
python scripts/prepare_compound_split.py \
  --source data/raw \
  --train-out data/compound/train.jsonl \
  --validation-out data/compound/validation.jsonl \
  --report-out data/compound/split-report.json \
  --validation-fraction 0.1 \
  --min-events 32
```

Exact MIDI byte duplicates are grouped by SHA-256 so they cannot cross train/validation.

### 2. Train

```bash
orbitune-compound train \
  --train-jsonl data/compound/train.jsonl \
  --validation-jsonl data/compound/validation.jsonl \
  --config configs/compound_hierarchical_9m.json \
  --checkpoint checkpoints/compound-base.pt \
  --device cpu \
  --steps 10000
```

`--device cuda` can be selected deliberately on a local CUDA machine. Remote GPU infrastructure is not required for the Compound path.

### 3. Resume

```bash
orbitune-compound train \
  --train-jsonl data/compound/train.jsonl \
  --validation-jsonl data/compound/validation.jsonl \
  --checkpoint checkpoints/compound-base.pt \
  --resume checkpoints/compound-base.pt \
  --device cpu \
  --steps 20000
```

`--steps` is the final global step. Checkpoints contain model and optimizer state, RNG state, sampler state, global step, config, tokenizer ABI, and source metadata.

### 4. Generate MIDI

```bash
orbitune-compound generate \
  --checkpoint checkpoints/compound-base.pt \
  --out generated.mid \
  --events 512 \
  --device cpu
```

Continue an existing MIDI file with `--primer-midi prompt.mid`.

For architecture details and the full local workflow, see [`docs/COMPOUND_BASE.md`](docs/COMPOUND_BASE.md).

## Compound architecture

One Compound MIDI event is one temporal model step.

```text
Compound Event
      |
      +--> Local Transformer
      +--> Medium summary Transformer
      +--> Global summary Transformer
      +--> routed fast/medium/slow recurrent memory
      |
      v
context fusion
      |
      v
Intra-event Transformer
      |
      +--> categorical: event type / channel / pitch-state attributes
      +--> continuous: delta / duration / velocity / MIDI controls
      |
      v
next Compound Event
```

The local/summary histories are bounded and the recurrent state is fixed-size, so persistent generation state does not grow with total song length.

The Compound MIDI schema currently covers NOTE, CC, PROGRAM, BANK, TEMPO, PEDAL, PITCH_BEND, CHANNEL_PRESSURE, POLY_PRESSURE, and TIME_SIGNATURE. MIDI ingestion canonicalizes duplicates/retriggers and emits state/control events before same-step NOTE events.

## Verify before a long local run

```bash
python -m pytest -q tests/test_compound_base.py
python -m pytest -q
```

The Compound tests cover forward/backward, exact checkpoint resume, bounded streaming state, MIDI roundtrip, and default model-size constraints.

## Legacy Theory-REMI reference path

The existing `orbitune` CLI remains available for compatibility:

```bash
orbitune model-info
orbitune train-base \
  --tokens data/tokens/train.tokens \
  --validation-tokens data/tokens/validation.tokens \
  --out models/my-base.pt \
  --steps 1000 \
  --batch-size 8 \
  --seq-len 256
```

Legacy LoRA/Base registry, web export, and adapter commands continue to target the Theory-REMI compatibility ABI unless explicitly documented otherwise.

## Repository layout

```text
orbitune/             runtime, model, tokenizer and training code
configs/              model configurations
scripts/              corpus preparation and repository utilities
tests/                CPU unit/integration tests
docs/                 architecture and operational documentation
experiments/          research history and reproducible architecture experiments
workloads/            optional external-compute/infrastructure workloads
bases/                immutable accepted Base artifacts/manifests
adapters/             official/community adapters
registry/             generated Base/Adapter dependency registries
web/                  browser runtime / GitHub Pages app
```

Production code should live under `orbitune/`; research-only architecture probes stay under `experiments/` and are not required to train or generate with the local Compound Base.

## Current model-status boundary

The Compound path is runnable and production-shaped, but this repository does not claim that its final generated music already beats the legacy ~10M reference model. That comparison requires converged training and generated-MIDI evaluation, not short proxy runs.

Known corpus/model gates before publishing a final immutable Base include composition-aware near-duplicate handling, broader real-MIDI validation, and final long-run quality evaluation.

## Adapter and Base compatibility

A published Base is identified by a stable Base id plus exact checkpoint SHA-256. An Adapter targets exactly one compatible Base checkpoint. Changed checkpoint bytes require a new compatibility lineage rather than silently replacing a published Base.

See [`CONTRIBUTING_BASES.md`](CONTRIBUTING_BASES.md), [`CONTRIBUTING_ADAPTERS.md`](CONTRIBUTING_ADAPTERS.md), and [`docs/COMPOUND_BASE.md`](docs/COMPOUND_BASE.md).

## License

Orbitune source code is licensed under Apache-2.0. Each contributed Base and Adapter declares its own compatible license and training-data rights status.
