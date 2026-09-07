# Contributing Orbitune Adapters

Orbitune currently has **two separate model families with different Adapter status**:

- **Legacy Theory-REMI** — operational LoRA/Adapter ABI; community Adapter contributions may target compatible immutable Bases registered under `bases/`.
- **Compound Transformer** — trained Base/runtime path exists, but its public Adapter ABI is not frozen yet. See [Compound LoRA policy](docs/COMPOUND_LORA_POLICY.md).

Do not mix these ABIs.

## Legacy Theory-REMI compatibility rule

A legacy Adapter is bound to exactly one Base checkpoint by both Base id and SHA-256:

```text
base_model   <registered Base id>
base_sha256  <exact checkpoint SHA-256>
```

Matching architecture or parameter count alone is not enough. Both `manifest.json` and `adapter.safetensors` metadata must carry the same Base hash. Registry generation, Python loading, and the legacy browser runtime reject mismatches.

## Legacy required layout

```text
adapters/community/<adapter-id>/
  manifest.json
  adapter.safetensors
  demo.mid
  README.md
```

## Current operational legacy Adapter ABI

```text
architecture     orbitune-midi-gpt-v0
tokenizer        theory-remi-v0
adapter format   orbitune-lora-v0
rank             4
target_modules   q_proj + v_proj
reference shape  4 layers / hidden 448 / 7 heads / context 1024
```

The old hidden-240 ~3M configuration is historical and is not the current reference shape.

## Compound Adapter status

The Compound tokenizer/model/runtime use a different architecture and inference contract. The legacy `orbitune-lora-v0` tensor packing, rank and target-module assumptions are **not** a Compound Adapter ABI.

Current public policy:

- Base pretraining remains full-parameter training.
- Mutable continuation checkpoints are not public Adapter compatibility targets.
- A public Compound Adapter must target an immutable Base id + exact checkpoint SHA-256.
- Compound target modules, rank, scaling and artifact tensor layout must be measured and frozen under a **new Adapter ABI identifier**.
- The first validated browser strategy is a pre-merged LoRA variant exported as a matched Compound `stream` + `decoder_prefix` graph pair.
- Community Compound Adapter binaries are not accepted into the production registry until those gates are closed.

Experimental Compound LoRA code, bounded rank/target experiments, tests and ABI proposals are welcome when clearly labeled experimental and when they do not publish restricted Base/model artifacts.

See [docs/COMPOUND_LORA_POLICY.md](docs/COMPOUND_LORA_POLICY.md) for the complete Base-freeze, rights, training and release policy.

## Create and train a current legacy Adapter

The default scaffold targets a compatible registered legacy Base. For another registered compatible Base, set `base_model` in the manifest to that Base id and use its exact checkpoint SHA-256.

```bash
orbitune train-adapter \
  --base bases/my-base/model.pt \
  --tokens data/tokens/style-train.tokens \
  --validation-tokens data/tokens/style-validation.tokens \
  --validation-interval 50 \
  --out adapters/community/my-style-v0/adapter.safetensors
```

Training embeds the actual Base checkpoint SHA-256 in the Safetensors metadata. Copy the same value into the Adapter manifest.

This command belongs to the legacy Theory-REMI path. It must not be presented as Compound LoRA training.

## Legacy Adapter size policy

Recommended: one Adapter directory <= 1 MiB. The hard CI threshold is 5 MiB.

No Compound Adapter size limit is frozen yet; that limit should be selected together with the future Compound Adapter ABI.

## Rights and quality

Every public Adapter must declare its license and training-data rights status. `rights_confirmed` must be true for the existing legacy manifest contract. A non-empty generated `demo.mid` and non-empty `README.md` are mandatory for the existing legacy contribution path.

An Adapter cannot broaden the permissions of its Base. In particular, an Adapter or pre-merged derivative of a research-NC Compound Base remains noncommercial even when the Adapter's own training material is permissively licensed.

## Legacy pull request checklist

- [ ] `base_model` exists in `bases/`
- [ ] `base_sha256` exactly matches that Base checkpoint
- [ ] manifest Base hash equals Safetensors Base hash
- [ ] Adapter architecture/tokenizer/format ABI is compatible with the selected Base
- [ ] `demo.mid` is playable and non-empty
- [ ] README, license, and training-data declaration are complete
- [ ] Base/Adapter dependency validation CI passes

For Compound LoRA work, use the separate acceptance gates in [Compound LoRA policy](docs/COMPOUND_LORA_POLICY.md); do not force a Compound artifact through this legacy checklist.
