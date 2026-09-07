# Usage Guide

## Generating MIDI

### Basic generation (CPU)

```bash
orbitune-compound generate \
  --checkpoint "C:\Users\junny\OneDrive\Desktop\MIDI-GPT\orbitune_clone\runs\research_nc_aria_gigamidi_v1\model.pt" \
  --out generated.mid \
  --events 512 \
  --device cpu \
  --seed 100 \
  --temperature 0.85
```

### Continue from a primer MIDI

```bash
orbitune-compound generate \
  --checkpoint "C:\Users\junny\OneDrive\Desktop\MIDI-GPT\orbitune_clone\runs\research_nc_aria_gigamidi_v1\model.pt" \
  --primer-midi prompt.mid \
  --out continuation.mid \
  --events 512 \
  --device cpu \
  --seed 100 \
  --temperature 0.85
```

### GPU generation

```bash
orbitune-compound generate \
  --checkpoint "C:\Users\junny\OneDrive\Desktop\MIDI-GPT\orbitune_clone\runs\research_nc_aria_gigamidi_v1\model.pt" \
  --out generated.mid \
  --events 2048 \
  --device cuda \
  --seed 200 \
  --temperature 0.95
```

## Loading the checkpoint programmatically

```python
import torch
from orbitune.compound_base import CompoundHierarchicalGPT, CompoundBaseConfig
from orbitune.compound_training import parse_compound_checkpoint

# Load checkpoint (Compound schema v2)
ckpt = torch.load(
    "C:\\Users\\junny\\OneDrive\\Desktop\\MIDI-GPT\\orbitune_clone\\runs"
    "\\research_nc_aria_gigamidi_v1\\model.pt",
    map_location="cpu",
    weights_only=False,
)
ckpt = parse_compound_checkpoint(ckpt)

# Reconstruct model from saved config
cfg = CompoundBaseConfig(**ckpt["config"])
model = CompoundHierarchicalGPT(cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Generation defaults
# - seed: 100, 200, 300, 400, 500 (validated in quality_validation_3way.json)
# - temperature: 0.85 (conservative) or 0.95 (creative)
# - events: 500 (standard generation length in validation)
```

## Resuming training (prospective — not for the frozen checkpoint)

```bash
orbitune-compound resume \
  --checkpoint "C:\path\to\model.pt" \
  --steps 50000 \
  --device cuda \
  --allow-runtime-change
```

> **Important**: The frozen checkpoint at step 100,000 has `sampler_rng_state` saved correctly (post-fix), but the **frozen training run itself** (steps 50,000→100,000) was affected by the pre-fix bug where the local training RNG was not restored. Resuming from this checkpoint will start from the correct RNG state for **future** steps but will not reproduce the exact original sampler sequence. See `reproducibility.md` for details.

## Validation corpus

The validation corpus identity for this checkpoint is:

```
9b8f61eaa156971ad3988a13945184708edcdbc39293ae9d4ba52c5c87831c2c
```

This SHA-256 is computed over the sorted list of validation song SHA-256 values across both Aria and GigaMIDI validation splits. If the validation corpus changes, `best_validation_loss` and `best_step` are automatically reset.

## Rights / license

- This model is **non-commercial only** (CC-BY-NC-SA-4.0).
- Training data: Aria-MIDI (CC-BY-NC-SA-4.0) + GigaMIDI (research-use only).
- Generated output inherits the model license: CC-BY-NC-SA-4.0.
- Commercial use is prohibited.
