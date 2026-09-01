# Compound Hierarchical Base

`orbitune.compound_base` is the runnable Transformer-first Compound MIDI Base. It intentionally coexists with the legacy Theory-REMI `OrbituneGPT` rather than replacing it.

## Architecture

One Compound MIDI event is one temporal step. The model uses four Transformer paths plus fixed-size recurrent memory:

1. **Local Transformer** — causal attention over the recent event window.
2. **Medium summary Transformer** — pooled causal summaries every 8 events.
3. **Global summary Transformer** — pooled summaries of medium-scale states.
4. **Intra-event Transformer** — autoregressively decodes attributes of the next Compound event instead of predicting them all independently.
5. **Routed recurrent memory** — fast, medium and slow decayed GRU states. Persistent recurrent-state size does not grow with song length.

The decoder is mixed-type: event type, channel, pitch/state attributes are categorical; delta time, note duration, velocity, and continuous MIDI controls use bounded Gaussian heads and are quantized only when converted back into a Compound record/MIDI event.

The default config is approximately the scale of the previous ~10M reference Base, not the 280k proxy models used during architecture experiments.

## Local setup

```bash
git clone https://github.com/Unjuno/orbitune.git
cd orbitune
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
orbitune-compound inspect --config configs/compound_hierarchical_9m.json
```

The `inspect` command should work without MIDI data and is the quickest installation check.

## Prepare data

Put MIDI files under a directory such as `data/raw`, then run:

```bash
python scripts/prepare_compound_split.py \
  --source data/raw \
  --train-out data/compound/train.jsonl \
  --validation-out data/compound/validation.jsonl \
  --report-out data/compound/split-report.json \
  --validation-fraction 0.1 \
  --min-events 32
```

The split is song-preserving and exact MIDI byte duplicates are grouped by SHA-256 so they cannot cross train/validation. Composition-aware near-deduplication is still a corpus-quality gate before final production training.

## Train locally

CPU:

```bash
orbitune-compound train \
  --train-jsonl data/compound/train.jsonl \
  --validation-jsonl data/compound/validation.jsonl \
  --config configs/compound_hierarchical_9m.json \
  --checkpoint checkpoints/compound-base.pt \
  --device cpu \
  --steps 10000
```

If a local CUDA GPU is deliberately selected, change only `--device cuda`. The repository does not require RunPod or any remote GPU service for this path.

The checkpoint stores model/optimizer state, CPU and CUDA RNG state, Python RNG, sampler RNG, global step, configuration, tokenizer ABI, and source commit when `GITHUB_SHA` or `ORBITUNE_SOURCE_COMMIT` is set.

## Resume

```bash
orbitune-compound train \
  --train-jsonl data/compound/train.jsonl \
  --validation-jsonl data/compound/validation.jsonl \
  --checkpoint checkpoints/compound-base.pt \
  --resume checkpoints/compound-base.pt \
  --device cpu \
  --steps 20000
```

`--steps` is the final global step, so this continues from the saved step to step 20000 rather than training 20000 additional steps.

## Generate MIDI locally

Unconditional generation:

```bash
orbitune-compound generate \
  --checkpoint checkpoints/compound-base.pt \
  --out generated.mid \
  --events 512 \
  --device cpu
```

Continue an existing MIDI file:

```bash
orbitune-compound generate \
  --checkpoint checkpoints/compound-base.pt \
  --primer-midi prompt.mid \
  --out continuation.mid \
  --events 512 \
  --device cpu
```

Generation uses bounded local/summary histories plus fixed-size recurrent memory, so persistent runtime state does not grow with the total generated song length.

## Verification

Run the CPU test suite before a long training run:

```bash
python -m pytest -q tests/test_compound_base.py
python -m pytest -q
```

The Compound tests cover forward/backward, exact checkpoint resume, bounded streaming state, MIDI writer/parser roundtrip, and the default model-size contract.

## Scope

This is a production-shaped runnable Base path, not a claim that its final music quality already beats the legacy ~10M Transformer. The earlier short-budget proxy ranking is not used to select an MLP architecture here. Final model selection should use converged training and generated-MIDI listening/evaluation.
