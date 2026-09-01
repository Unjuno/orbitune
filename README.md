# Orbitune

Orbitune is a local-first symbolic MIDI generation framework. The repository currently keeps two compatible model paths:

- **Compound Transformer Base** — the current Transformer-first architecture for new training and generation work.
- **Theory-REMI reference Base** — the existing ~10M operational path kept intact for checkpoint, LoRA and deployment compatibility.

The new Base does not rely on the short-lived Windowed-MLP proxy as its production composer.

## Current Compound Transformer Base

One Compound MIDI event is one temporal model step. The Base combines multiple temporal scales instead of forcing all history into one flat context window:

```text
Compound MIDI Event
        ↓
Factorized Event Embedding
        ↓
Local Causal Transformer ───────────────┐
        ↓                               │
Medium Summary Transformer ────────────┤
        ↓                               ├─ Fusion
Global Summary Transformer ────────────┤
        ↓                               │
Fast / Medium / Slow Recurrent Memory ─┘
        ↓
Intra-event Transformer
        ↓
Discrete + Continuous Attribute Heads
        ↓
Next Compound MIDI Event
```

Persistent generation state is bounded: recent local events, bounded medium/global summary histories and fixed-size recurrent memory. The checked-in config is approximately the same size class as the previous ~10M reference Base; the 280k models under `experiments/` are research proxies only.

## Local quick start

```bash
git clone https://github.com/Unjuno/orbitune.git
cd orbitune
git switch midi-gpt-base-complete

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

Prepare a directory of Standard MIDI files:

```bash
orbitune-compound prepare midi/
```

This creates the default training inputs:

```text
data/compound/train.jsonl
data/compound/validation.jsonl
data/compound/report.json
```

Inspect and train on CPU:

```bash
orbitune-compound info --config configs/compound_hierarchical_9m.json
orbitune-compound train --device cpu
```

Training writes `models/compound-base.pt` by default. Resume exactly from it with optimizer and RNG state restored:

```bash
orbitune-compound resume \
  --checkpoint models/compound-base.pt \
  --steps 20000 \
  --device cpu
```

Generate a Standard MIDI file locally:

```bash
orbitune-compound generate \
  --checkpoint models/compound-base.pt \
  --out generated.mid \
  --events 512 \
  --device cpu
```

Continue an existing MIDI file:

```bash
orbitune-compound generate \
  --checkpoint models/compound-base.pt \
  --primer-midi prompt.mid \
  --out continuation.mid \
  --events 512 \
  --device cpu
```

For the complete runtime contract and explicit command options, see [`docs/COMPOUND_BASE.md`](docs/COMPOUND_BASE.md).

## Compound representation

The current semantic MIDI schema covers:

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

MIDI is canonicalized deterministically. Same-onset/channel/pitch duplicate notes are merged, overlapping retriggers are truncated, unused fields are zeroed, and same-step state/control events precede NOTE events.

The serialized Compound record remains 12 fields so existing prepared corpora stay readable. The model uses categorical heads where the value is intrinsically discrete and continuous auxiliary/generative heads for ordered numeric attributes such as delta time, note duration, velocity and continuous controls.

Known corpus/representation gates before publishing a final immutable Base include composition-aware near-deduplication, broader real-MIDI validation and final handling of rare/meta MIDI semantics.

## CPU smoke before GPU training

Repository validation should not spend GPU time:

```bash
python -m pytest -q tests/test_compound_base.py tests/test_compound_cli.py
```

The Compound tests cover forward/backward, exact checkpoint restoration, bounded streaming state, MIDI write/read and the actual local command chain:

```text
prepare → train → resume → info → generate
```

Use GPU only for an actual corpus-scale training run after the CPU path is green.

## Legacy Theory-REMI path

The original `orbitune` CLI remains available and is intentionally separate:

```bash
orbitune prepare-split-corpus data/raw \
  --train-out data/tokens/train.tokens \
  --validation-out data/tokens/validation.tokens \
  --report data/tokens/split-report.json

orbitune train-base \
  --tokens data/tokens/train.tokens \
  --validation-tokens data/tokens/validation.tokens \
  --out models/legacy-base.pt
```

Existing LoRA adapters, Base registry, ONNX/Web export and legacy checkpoints continue to use this path. Do not discard the legacy Base merely because short proxy experiments favored another operator; final model selection requires converged real-corpus training and generated-MIDI comparison.

## Repository layout

```text
orbitune/              runtime/model/tokenizer package
configs/               checked-in model configurations
data/                   local prepared data (generated artifacts are not Bases)
models/                 local candidate checkpoints
bases/                  immutable accepted Base artifacts/manifests
adapters/               official/community LoRA adapters
experiments/            architecture research and reproducible proxies
workloads/              bounded external-compute workloads
scripts/                maintenance and legacy helper entrypoints
docs/                   architecture, audit, handoff and runtime documentation
tests/                  CPU unit/integration contracts
web/                    legacy local browser runtime
```

The distinction is intentional: code required to run the Compound Base lives under `orbitune/`; architecture experiments remain under `experiments/` and are not imported by the local Base runtime.

## LoRA and adapters

The existing Adapter ABI is tied to the legacy Theory-REMI architecture. The Compound Transformer Base keeps standard `nn.Linear` attention/decoder projections so a Compound-specific LoRA contract can be added without changing the Base architecture, but adapters should not be silently mixed across the two checkpoint ABIs.

## Remote/GPU infrastructure

`workloads/` contains bounded RunPod/GPU-control canaries and benchmarks. They are infrastructure tools, not a prerequisite for local training or generation. CPU CI should validate source, checkpoint and CLI contracts first; GPU compute is reserved for training workloads where measured CPU throughput is insufficient.

## CI

Primary checks include:

- `test.yml` — Python unit/integration tests, including Compound model contracts.
- `ml-smoke.yml` — legacy reference training/LoRA smoke.
- `runpod-canary-smoke.yml` — CPU contract for remote-GPU workload packaging.
- `continuous-smoke.yml` / `continuous-train.yml` — legacy resumable training path.
- `export-smoke.yml` — legacy ONNX/export staging.
- `validate-adapters.yml` — Base/Adapter manifest and compatibility checks.
- `web-test.yml` / `pages.yml` — legacy browser runtime and published assets.

## License

Orbitune source code is licensed under Apache-2.0. Each contributed Base and Adapter must declare its own compatible license and training-data rights status.
