# Experimental Compound LoRA SFT

## Status

This is a bounded research/debug path for exercising post-Base LoRA adaptation on the Compound model family. It is **not** the frozen public Compound Adapter ABI and it does not change the policy in [COMPOUND_LORA_POLICY.md](COMPOUND_LORA_POLICY.md).

The current immutable Base target is the completed A2-512 release, `orbitune-a2-512-research-nc`. Experimental Adapters must remain bound to its exact checkpoint SHA (or to another explicitly named experimental Base); architecture or parameter-count similarity is never treated as compatibility.

The executable post-Base path is:

```text
immutable Compound checkpoint
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

## Controlled target/rank screening

Do not choose the Compound target modules or rank from the legacy Theory-REMI settings by analogy. `scripts/compound_lora_sweep.py` runs multiple experimental candidates through the same `compound_lora_sft.py` path while holding the comparison protocol fixed.

The checked-in first-pass grid is [compound_lora_a2_screen_v1.json](../experiments/compound_lora_a2_screen_v1.json). It compares decoder-only q/q+v adaptation, rank 4 versus rank 8 at constant `alpha/rank = 2`, hierarchical-context q+v adaptation, and combined context+decoder q+v adaptation. This is an experiment definition, not a production recommendation.

Example:

```bash
python scripts/compound_lora_sweep.py \
  --base-checkpoint /path/to/a2/model.pt \
  --base-id orbitune-a2-512-research-nc \
  --train-jsonl /path/to/adapter-train.jsonl \
  --validation-jsonl /path/to/adapter-validation.jsonl \
  --spec experiments/compound_lora_a2_screen_v1.json \
  --output-dir runs/a2-lora-screen-v1 \
  --steps 1000 \
  --batch-size 4 \
  --seq-len 128 \
  --validation-batches 16 \
  --seed 17 \
  --device cuda
```

Every candidate is launched with the same Base bytes, train/validation files, seed, optimization settings, step count and validation protocol. Because LoRA `B` is zero-initialized, the initial validation loss must agree across candidates; the sweep fails if that invariant is violated. The summary records exact Base/data/spec hashes, resolved targets, Adapter SHA-256, trainable parameter count, initial/final validation loss and a deterministic ML-only ranking.

The output `sweep.json` is deliberately marked:

```text
status = experimental_ml_screen
preference_claim = false
```

A lower held-out loss is not proof that listeners prefer the music. The leading candidates must advance to generated-MIDI structural checks and blinded human preference evaluation before any target/rank/scaling choice is frozen.

## Output

A successful single-candidate run writes:

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

## Pre-merged Web candidate

The browser does not need a dynamic LoRA ABI for the initial product path. A reviewed Adapter can be merged locally into its exact Base and then passed through the existing native-stream V2 export pipeline as an ordinary Compound model.

First verify and pre-merge the Adapter:

```bash
python scripts/merge_compound_lora.py \
  --base-checkpoint /path/to/model.pt \
  --base-sha256 <exact-base-sha256> \
  --base-model-id orbitune-a2-512-research-nc \
  --adapter-dir /path/to/adapter \
  --adapter-id <versioned-adapter-id> \
  --output-checkpoint runs/premerged/model.pt \
  --output-manifest runs/premerged/manifest.json
```

The merge command:

- rejects the wrong Base SHA or model id before applying the Adapter;
- uses the Adapter's exact target/rank/alpha/scaling metadata;
- verifies bounded numerical parity between each eval-mode LoRA wrapper and its merged linear;
- removes all LoRA wrappers so the result strictly reloads as a normal `CompoundHierarchicalGPT`;
- records Adapter manifest/tensor SHA-256 values and merged-checkpoint identity;
- writes `publication_eligible = false` intentionally.

A successful merge is therefore **not** a publication approval. It only creates a technically inspectable derivative candidate.

Next run the existing V2 export/parity path against the exact merged checkpoint:

```bash
python scripts/export_compound_web_v2.py \
  --checkpoint runs/premerged/model.pt \
  --checkpoint-sha256 <merged-checkpoint-sha256> \
  --out-dir runs/premerged/web
```

The exact derivative must then pass native, ONNX Runtime and onnxruntime-web/WASM validation, generated-MIDI quality comparison, Adapter-data rights review and the inherited A2 research/non-commercial licensing boundary.

After those gates, an approved Web release may use `kind = "lora-premerged"` and must include a versioned `adapter.id`. `scripts/build_compound_web_runtime_config.py` can aggregate reviewed additional LoRA releases with the canonical Base release, but rejects wrong-Base, wrong-runtime-ABI, unreviewed, commercialized, duplicate-id or missing-Adapter-identity variants.

Changing Base/LoRA variant in the PWA starts a new stream. Reusing recurrent stream state across independently merged variants is not assumed to be valid.

## Tests

Run the bounded CPU tests with:

```bash
python -m pytest -q \
  tests/test_compound_lora.py \
  tests/test_compound_lora_merge.py \
  tests/test_compound_lora_sweep.py \
  tests/test_compound_web_release.py
```

Coverage includes no-op injection, trainable-parameter scope, Base immutability, Safetensors save/reload, Base-SHA rejection, tiny end-to-end SFT invocation, controlled sweep reproducibility/invariants, wrapper-to-merged numerical parity, clean-model reload and strict `lora-premerged` Web release aggregation.

## Relationship to future reward-guided training

Reward-guided post-training should reuse the same low-level LoRA injection, Base freezing and artifact binding rather than implementing a second Adapter mechanism. A separate research repository can own rollout logging, MIDI-to-audio rendering, aesthetic reward evaluation and GRPO while importing the Compound LoRA primitive from Orbitune.

The public dynamic Adapter ABI should only be frozen after target modules/rank/scaling/merge behavior are measured on the immutable A2 Base. A pre-merged Web derivative does not require freezing that dynamic browser ABI, but it still requires exact Adapter identity, evaluation, rights review and parity on its own exported bytes.
