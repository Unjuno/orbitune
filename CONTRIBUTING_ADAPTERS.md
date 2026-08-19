# Contributing Orbitune Adapters

Orbitune accepts small LoRA adapters only for the immutable published `orbitune-base` checkpoint.

## Non-negotiable compatibility rule

An Adapter is compatible only when its `base_sha256` exactly matches the official Base checkpoint SHA-256. Matching architecture, parameter count, or tokenizer is not enough.

Both `manifest.json` and `adapter.safetensors` metadata must contain the same exact Base hash. Python, browser runtime, registry generation, and CI reject a mismatch.

The official Base checkpoint is never replaced after publication. If a different Base is ever created, it belongs to a separate compatibility lineage and cannot silently inherit the existing Adapter catalog.

## Required layout

```text
adapters/community/<adapter-id>/
  manifest.json
  adapter.safetensors
  demo.mid
  README.md
```

## Required compatibility metadata

Every Adapter declares:

```text
base_model       orbitune-base
base_sha256      <exact 64-hex checkpoint SHA-256>
architecture     orbitune-midi-gpt-v0
tokenizer        theory-remi-v0
adapter_type     lora
rank             4
target_modules   q_proj + v_proj
```

The `v0` strings above identify protocol/file ABIs; they do not identify a replaceable Base-weight version.

## Training workflow

Train against the verified official Base file:

```bash
orbitune train-adapter \
  --base models/orbitune-base.pt \
  --tokens data/tokens/style-train.tokens \
  --validation-tokens data/tokens/style-validation.tokens \
  --validation-interval 50 \
  --out adapters/community/my-style-v0/adapter.safetensors
```

The resulting Safetensors metadata embeds the exact Base SHA-256. Copy the same value into `manifest.json` before submission.

## Size policy

Recommended: one Adapter directory <= 1 MiB. Direct repository submissions above 5 MiB require maintainer review. Base checkpoints and large datasets are never committed to Git history.

## Rights and quality

Every Adapter must declare its Adapter license and training-data rights status. `rights_confirmed` must be true before acceptance. A non-empty generated `demo.mid` and non-empty `README.md` are mandatory.

## Pull request checklist

- [ ] `base_model` is `orbitune-base`
- [ ] `base_sha256` exactly matches the official Base checkpoint
- [ ] manifest Base hash equals Safetensors Base hash
- [ ] rank is 4 and targets are `q_proj`, `v_proj`
- [ ] `demo.mid` is playable and non-empty
- [ ] README, license and training-data declaration are complete
- [ ] Adapter validation CI passes

The registry is generated automatically; contributors should not create a second Base lineage by manually editing registry metadata.
