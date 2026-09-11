# Experimental Compound LoRA SFT

## Status

This is a bounded research/debug path for exercising post-Base LoRA adaptation on the Compound model family. It is **not** the frozen public Compound Adapter ABI and it does not change the policy in [COMPOUND_LORA_POLICY.md](COMPOUND_LORA_POLICY.md).

The purpose is to make the minimum post-Base path executable before the final Base and public Adapter ABI are frozen:

```text
immutable/local Compound checkpoint
+ existing Compound JSONL data
→ freeze Base parameters
→ inject explicitly configured experimental LoRA
→ SFT LoRA parameters only
→ validate on a separate JSONL split
→ save experimental Safetensors + Base binding metadata
```

This path is also intended to smoke-test the LoRA primitive that later reward-guided post-training experiments can reuse. Reward models, GRPO and preference optimization are deliberately outside this script.

## Prepare data

Use the same Compound preparation path as Base training. Do not introduce a separate Adapter tokenizer or data ABI.

```bash
orbitune-compound prepare /path/to/midi \
  --train-out data/compound/train.jsonl \
  --validation-out data/compound/validation.jsonl \
  --report data/compound/report.json
```

Keep training and validation data separate.

## Run a bounded experiment

Target modules, rank and alpha are required arguments on purpose. There is no claimed default Compound Adapter configuration yet.

```bash
python scripts/compound_lora_sft.py \
  --base-checkpoint /path/to/frozen-or-experimental-base.pt \
  --base-id local-compound-candidate \
  --train-jsonl data/compound/train.jsonl \
  --validation-jsonl data/compound/validation.jsonl \
  --output-dir runs/compound-lora-smoke \
  --target-module 'decoder.stack.blocks.*.attn.q_proj' \
  --rank 4 \
  --alpha 8 \
  --steps 100 \
  --device cpu
```

The target/rank values above are examples for exercising the code path, not recommended production settings and not the legacy Theory-REMI ABI.

The script refuses to overwrite a non-empty output directory. It records the exact Base checkpoint SHA-256, model/tokenizer ABI, resolved target modules, dataset hashes and training settings.

## Output

A successful run writes:

```text
runs/compound-lora-smoke/
  adapter.safetensors
  adapter.json
  training.json
  metrics.json
```

`adapter.json` is explicitly marked:

```text
schema = orbitune-compound-lora-experimental-v0
status = experimental
public_adapter_abi = false
```

The format may change before the public Compound Adapter ABI is frozen.

## Invariants

The experimental implementation enforces the following invariants:

- Base parameters are frozen before Adapter injection.
- Only `lora_A` and `lora_B` parameters are trainable.
- LoRA `B` starts at zero, so initial injection is a no-op.
- Base parameters are hashed before and after training and a mutation fails the run.
- Adapter artifacts are bound to an exact Base SHA-256.
- Loading against the wrong Base SHA fails closed.
- Adapter tensors are stored as Safetensors rather than executable pickle checkpoints.
- Existing Compound JSONL is reused for SFT; there is no LoRA-specific data encoding.

These checks exercise implementation safety. They do not establish musical quality, rights to redistribute a Base/Adapter, or compatibility with a future public Compound Adapter ABI.

## Tests

Run the bounded CPU tests with:

```bash
python -m pytest -q tests/test_compound_lora.py
```

The tests cover no-op injection, trainable-parameter scope, Base immutability, Safetensors save/reload, Base-SHA rejection and a tiny end-to-end SFT script invocation.

## Relationship to future reward-guided training

Reward-guided post-training should reuse the same low-level LoRA injection, Base freezing and artifact binding rather than implementing a second Adapter mechanism. A separate research repository can own rollout logging, MIDI-to-audio rendering, aesthetic reward evaluation and GRPO while importing the Compound LoRA primitive from Orbitune.

The public Adapter ABI should only be frozen after the final Base is selected and target modules/rank/scaling/merge behavior are measured, as required by the project policy.
