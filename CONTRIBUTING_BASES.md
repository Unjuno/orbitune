# Contributing Orbitune Bases

Orbitune accepts compact Base models directly in this repository.

## Layout

```text
bases/<base-id>/
  manifest.json
  model.pt
  web.onnx
  README.md
```

A Base id is an immutable compatibility lineage. Never replace the checkpoint bytes of an existing Base id after adapters have been published against it. Improvements are contributed under a new id.

## Current repository limits

- parameter count: <= 100,000,000
- each Base binary artifact: <= 95 MiB
- checkpoint and Web artifact must both be committed under the Base directory for the currently supported contribution path
- exact SHA-256 and byte counts are mandatory in `manifest.json`

## ABI status

The currently operational public contribution tooling is still the **legacy/reference** ABI:

```text
architecture     orbitune-midi-gpt-v0
tokenizer        theory-remi-v0
reference shape  4 layers / hidden 448 / 7 heads / context 1024
adapter format   orbitune-lora-v0
```

The historical hidden-240 ~3M configuration is not the current reference Base.

The production-candidate Compound path uses `orbitune-compound-v0-experimental` while it is under validation. **Do not publish an immutable community Base against that experimental ABI yet.** A future accepted Compound Base must declare a frozen architecture/tokenizer/runtime ABI before adapters target it.

A differently shaped Base may coexist in the registry only when its runtime/export path is actually supported. Listing an arbitrary architecture string does not by itself make the current browser or Adapter loader compatible with it.

## Staging a current legacy/reference ABI Base

After training and ONNX export, stage only after you have actually checked the model/data rights. The staging command requires an explicit acknowledgement and will not manufacture `rights_confirmed=true` implicitly:

```bash
python scripts/add_base.py \
  --id my-base \
  --display-name "My Base" \
  --checkpoint models/my-base.pt \
  --web-onnx my-base-web.onnx \
  --license Apache-2.0 \
  --training-license original \
  --rights-confirmed
```

This copies the artifacts into `bases/my-base/`, computes SHA-256 values, and creates the manifest/README. `--id` is validated before any output path is created, so Base ids cannot escape the configured output root.

Then regenerate registries:

```bash
PYTHONPATH=. python scripts/build_registry.py \
  --bases bases \
  --adapters adapters \
  --base-out registry/bases.json \
  --adapter-out registry/adapters.json
```

## Adapter relationship

Every Adapter names both:

```json
{
  "base_model": "my-base",
  "base_sha256": "<exact checkpoint sha256>"
}
```

CI rejects unknown Base ids and SHA mismatches. The Adapter's declared architecture/tokenizer/format must also be compatible with the selected Base and runtime.

## Rights and provenance

A Base contribution must declare its code/model license and training-data rights status. `training_data.rights_confirmed` must be true before the Base can be accepted. The CLI's `--rights-confirmed` flag is an acknowledgement, not evidence by itself; reviewers must still verify the provenance declaration.

For official Orbitune Bases, rights confirmation is necessary but not sufficient: the corpus pipeline must also record provenance, parsing/quality filtering, deduplication and train/validation separation. Exact-byte hash grouping alone is not considered composition-level deduplication.
