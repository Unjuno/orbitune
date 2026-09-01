# Compound Transformer Base

`orbitune-compound` is the local-first training and generation path for Orbitune's Transformer-first Compound MIDI Base. It coexists with the legacy Theory-REMI `orbitune` path; existing checkpoints and adapter tooling are not replaced.

## Architecture

One Compound MIDI event is one temporal step. The current Base combines:

1. **Local causal Transformer** for exact recent-event context.
2. **Medium summary Transformer** over pooled local states.
3. **Global summary Transformer** over pooled medium states.
4. **Routed recurrent memory** with fast, medium and slow decayed states whose persistent size does not grow with song length.
5. **Intra-event Transformer** that autoregressively decodes the attributes of the next Compound event.
6. **Mixed output heads**: categorical heads for discrete MIDI state and bounded continuous heads for delta time, duration, velocity and continuous controls.

The checked-in `configs/compound_hierarchical_9m.json` is approximately the same model-size class as the previous ~10M reference Base. The 280k models under `experiments/` are research proxies and are not the runnable Base.

## Clone and install

```bash
git clone https://github.com/Unjuno/orbitune.git
cd orbitune

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

Everything below works on CPU. A GPU is optional and is not required for repository validation.

## 1. Prepare MIDI

Put training MIDI files under a directory such as `midi/`, then run:

```bash
orbitune-compound prepare midi/
```

Default outputs are deliberately aligned with the training command:

```text
data/compound/train.jsonl
data/compound/validation.jsonl
data/compound/report.json
```

The split is song-preserving and groups exact duplicate MIDI bytes by SHA-256 so identical files cannot cross train/validation. Composition-family near-deduplication is still a corpus-quality gate for a production training set.

Custom paths remain available:

```bash
orbitune-compound prepare /path/to/midi \
  --train-out data/compound/train.jsonl \
  --validation-out data/compound/validation.jsonl \
  --report data/compound/report.json \
  --validation-fraction 0.1 \
  --min-events 8
```

## 2. Inspect the model before training

```bash
orbitune-compound info --config configs/compound_hierarchical_9m.json
```

This prints the architecture ABI, parameter count and complete model configuration.

## 3. Train

The defaults line up with the files created by `prepare`:

```bash
orbitune-compound train --device cpu
```

Equivalent explicit form:

```bash
orbitune-compound train \
  --train-jsonl data/compound/train.jsonl \
  --validation-jsonl data/compound/validation.jsonl \
  --config configs/compound_hierarchical_9m.json \
  --checkpoint models/compound-base.pt \
  --steps 10000 \
  --batch-size 8 \
  --seq-len 256 \
  --device cpu
```

The checkpoint stores model weights, optimizer state, global step, model/tokenizer ABI, configuration, Python RNG, Torch RNG, CUDA RNG when applicable, sampler RNG and the source commit when `GITHUB_SHA` or `ORBITUNE_SOURCE_COMMIT` is present.

## 4. Resume

Resume is a first-class command; there is no need to reconstruct optimizer or RNG state manually:

```bash
orbitune-compound resume \
  --checkpoint models/compound-base.pt \
  --steps 20000 \
  --device cpu
```

`--steps` is the final target global step. If the checkpoint is at step 10000, the command above continues from 10001 through 20000.

Inspect the saved state with:

```bash
orbitune-compound info --checkpoint models/compound-base.pt
```

## 5. Generate MIDI

```bash
orbitune-compound generate \
  --checkpoint models/compound-base.pt \
  --out generated.mid \
  --events 512 \
  --device cpu
```

Continue an existing MIDI file with:

```bash
orbitune-compound generate \
  --checkpoint models/compound-base.pt \
  --primer-midi prompt.mid \
  --out continuation.mid \
  --events 512 \
  --device cpu
```

Generation keeps bounded local/medium/global histories plus fixed-size recurrent memory, so persistent runtime state does not grow with total song length.

## CPU repository smoke

Before spending GPU time:

```bash
python -m pytest -q tests/test_compound_base.py tests/test_compound_cli.py
```

The tests cover forward/backward, exact checkpoint restoration, bounded stream state, Standard MIDI roundtrip and the actual `prepare -> train -> resume -> info -> generate` CLI path using a tiny CPU model.

## Legacy Base

The previous Theory-REMI Base remains available through the `orbitune` command, including `orbitune train-base` and the existing LoRA/adapter tooling. Do not delete old checkpoints when evaluating this Base; final musical-quality comparison requires converged real-corpus training and generated-MIDI evaluation, not the short architecture-proxy runs.
