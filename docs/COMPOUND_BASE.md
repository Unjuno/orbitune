# Compound Hierarchical Base

`orbitune.compound_base` is the runnable Transformer-first Compound MIDI Base.
It intentionally coexists with the legacy Theory-REMI `OrbituneGPT` rather than
replacing it.

## Architecture

One Compound MIDI event is one temporal step. The model uses four Transformer
paths plus fixed-size recurrent memory:

1. **Local Transformer** — causal attention over the recent event window.
2. **Medium summary Transformer** — pooled causal summaries every 8 events.
3. **Global summary Transformer** — pooled summaries of medium-scale states.
4. **Intra-event Transformer** — autoregressively decodes the attributes of the
   next Compound event instead of predicting all attributes independently.
5. **Routed recurrent memory** — fast, medium and slow decayed GRU states. Its
   persistent state size does not grow with song length.

The decoder is mixed-type: event type, channel, pitch/state attributes are
categorical; delta time, note duration, velocity, and continuous MIDI controls
use bounded Gaussian heads and are quantized only when converted back into a
Compound record/MIDI event.

The default config is approximately the scale of the previous ~10M reference
Base, not the 280k proxy models used during architecture experiments.

## Prepare data

Use the existing Compound corpus pipeline:

```bash
python scripts/prepare_compound_split.py \
  --source /path/to/midi \
  --train-jsonl data/train.jsonl \
  --validation-jsonl data/validation.jsonl \
  --split-json data/split.json
```

## Inspect

```bash
orbitune-compound inspect --config configs/compound_hierarchical_9m.json
```

## Train on CPU

```bash
orbitune-compound train \
  --train-jsonl data/train.jsonl \
  --validation-jsonl data/validation.jsonl \
  --config configs/compound_hierarchical_9m.json \
  --checkpoint checkpoints/compound-base.pt \
  --device cpu \
  --steps 10000
```

The checkpoint stores model/optimizer state, CPU and CUDA RNG state, Python RNG,
sampler RNG, global step, configuration, tokenizer ABI, and source commit when
`GITHUB_SHA` or `ORBITUNE_SOURCE_COMMIT` is set.

Resume without changing the checkpoint path:

```bash
orbitune-compound train \
  --train-jsonl data/train.jsonl \
  --validation-jsonl data/validation.jsonl \
  --checkpoint checkpoints/compound-base.pt \
  --resume checkpoints/compound-base.pt \
  --device cpu \
  --steps 20000
```

`--steps` is the final global step, so the command above continues from the
saved step to step 20000.

## Generate MIDI locally

```bash
orbitune-compound generate \
  --checkpoint checkpoints/compound-base.pt \
  --out generated.mid \
  --events 512 \
  --device cpu
```

To continue an existing MIDI file:

```bash
orbitune-compound generate \
  --checkpoint checkpoints/compound-base.pt \
  --primer-midi prompt.mid \
  --out continuation.mid \
  --events 512 \
  --device cpu
```

Generation uses bounded local/summary histories plus the fixed-size recurrent
memory state, so persistent runtime state does not grow with the total generated
song length.

## Current scope

This is a production-shaped runnable Base path, not a claim that its final music
quality beats the legacy ~10M Transformer. The previous 12-epoch proxy ranking
is not used to select an MLP architecture here. Final model selection should use
converged training and generated-MIDI listening/evaluation.
