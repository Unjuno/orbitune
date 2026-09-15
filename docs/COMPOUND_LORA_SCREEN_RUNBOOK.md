# A2-512 Compound LoRA screen runbook

This runbook is for the **target/rank ML screen only**. It does not establish a human-preference winner and it does not freeze the public Compound Adapter ABI.

## Inputs

Use the immutable A2-512 Base checkpoint and keep its exact SHA-256 binding:

- Base id: `orbitune-a2-512-research-nc`
- checkpoint SHA-256: `e5bd2080ccf084edaa33c0df9864e4d353b4fe184ed199a2ea89a1cc06324fe0`
- sweep spec: `experiments/compound_lora_a2_screen_v1.json`

The LoRA screen can now read the indexed Compound corpus directly. No giant intermediate JSONL export is required.

Expected commercial-safe corpus layout:

```text
data/corpora/commercial_v1/compound_indexed/
  train/
    index.json
    records.i32
    songs.jsonl
  validation/
    index.json
    records.i32
    songs.jsonl
```

The indexed source identity recorded in every Adapter artifact includes the indexed format/schema/tokenizer ABI, split, manifest SHA-256, index SHA-256, byte sizes and song/event counts. The sweep also verifies that every child candidate reports the same source identity as the parent sweep.

## First real five-candidate screen

Run on CUDA:

```bash
python scripts/compound_lora_sweep.py \
  --base-checkpoint /path/to/a2-512/model.pt \
  --base-id orbitune-a2-512-research-nc \
  --train-source data/corpora/commercial_v1/compound_indexed/train \
  --validation-source data/corpora/commercial_v1/compound_indexed/validation \
  --spec experiments/compound_lora_a2_screen_v1.json \
  --output-dir runs/a2-lora-screen-v1 \
  --steps 1000 \
  --batch-size 4 \
  --seq-len 128 \
  --validation-batches 16 \
  --seed 17 \
  --device cuda
```

The five checked-in candidates are:

1. `decoder-q-r4`
2. `decoder-qv-r4`
3. `decoder-qv-r8`
4. `context-qv-r4`
5. `context-decoder-qv-r4`

All candidates keep `alpha/rank = 2.0` and use the same Base bytes, source identities, optimizer settings, seed, step budget and validation protocol.

## Gate after the ML screen

`sweep.json` remains deliberately marked:

```text
status = experimental_ml_screen
preference_claim = false
```

Do not publish the lowest-loss candidate automatically. Retain several leaders and move them through:

1. fixed-seed generation using matched prompts/primers;
2. symbolic structural checks, including NOTE/PROGRAM/channel distribution and degeneration rates;
3. the pinned sampled SoundFont renderer;
4. blinded pairwise listening with randomized A/B side and ties/abstain;
5. only then decide whether ordinary LoRA SFT is sufficient or reward/preference optimization is justified.

For the current A2 Base, explicitly track the previously observed failure modes during this gate:

- Program 0 dominance among audible NOTE trajectories;
- nonzero-Program NOTE fraction;
- channel-10 drum NOTE fraction;
- active channel count;
- PROGRAM-only degeneration rate.

These are diagnostic metrics, not standalone aesthetic rewards.

## Backward compatibility

The older JSONL interface remains accepted:

```text
--train-jsonl ...
--validation-jsonl ...
```

New runs should prefer `--train-source` and `--validation-source`; each accepts either a Compound JSONL file or an indexed split directory / `index.json`.
