# Contributing Orbitune Adapters

Orbitune accepts small LoRA adapters for compatible immutable Base checkpoints registered under `bases/`.

## Compatibility rule

An Adapter is bound to exactly one Base checkpoint by both Base id and SHA-256:

```text
base_model   <registered Base id>
base_sha256  <exact checkpoint SHA-256>
```

Matching architecture or parameter count alone is not enough. Both `manifest.json` and `adapter.safetensors` metadata must carry the same Base hash. Registry generation, Python loading, and the browser runtime reject mismatches.

## Required layout

```text
adapters/community/<adapter-id>/
  manifest.json
  adapter.safetensors
  demo.mid
  README.md
```

## Current operational Adapter ABI

```text
architecture     orbitune-midi-gpt-v0
tokenizer        theory-remi-v0
adapter format   orbitune-lora-v0
rank             4
target_modules   q_proj + v_proj
reference shape  4 layers / hidden 448 / 7 heads / context 1024
```

The old hidden-240 ~3M configuration is historical and is not the current reference shape.

The Compound production path is still experimental (`orbitune-compound-v0-experimental`). Do not publish community adapters against it until a Compound Base architecture/tokenizer/Adapter ABI is frozen. A future Compound Adapter ABI may use different target modules, packing or rank defaults.

## Create and train a current reference Adapter

The default scaffold targets `orbitune-base`. For another registered compatible Base, set `base_model` in the manifest to that Base id and use its exact checkpoint SHA-256.

```bash
orbitune train-adapter \
  --base bases/my-base/model.pt \
  --tokens data/tokens/style-train.tokens \
  --validation-tokens data/tokens/style-validation.tokens \
  --validation-interval 50 \
  --out adapters/community/my-style-v0/adapter.safetensors
```

Training embeds the actual Base checkpoint SHA-256 in the Safetensors metadata. Copy the same value into the Adapter manifest.

## Size policy

Recommended: one Adapter directory <= 1 MiB. The hard CI threshold is 5 MiB.

## Rights and quality

Every Adapter must declare its license and training-data rights status. `rights_confirmed` must be true. A non-empty generated `demo.mid` and non-empty `README.md` are mandatory.

## Pull request checklist

- [ ] `base_model` exists in `bases/`
- [ ] `base_sha256` exactly matches that Base checkpoint
- [ ] manifest Base hash equals Safetensors Base hash
- [ ] Adapter architecture/tokenizer/format ABI is compatible with the selected Base
- [ ] `demo.mid` is playable and non-empty
- [ ] README, license, and training-data declaration are complete
- [ ] Base/Adapter dependency validation CI passes
