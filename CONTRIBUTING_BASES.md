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
- checkpoint and web ONNX must both be committed under the Base directory
- exact SHA-256 and byte counts are mandatory in `manifest.json`

The current rank-4 LoRA Adapter ABI supports the `orbitune-midi-gpt-v0` / `theory-remi-v0` 4x240 architecture. A differently shaped Base may still be listed as a Base, but it needs a compatible Adapter ABI before community LoRA adapters can target it.

## Staging a current-ABI Base

After training and ONNX export:

```bash
python scripts/add_base.py \
  --id my-base \
  --display-name "My Base" \
  --checkpoint models/my-base.pt \
  --web-onnx my-base-web.onnx \
  --license Apache-2.0 \
  --training-license original
```

This copies the artifacts into `bases/my-base/`, computes SHA-256 values, and creates the manifest/README.

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

CI rejects unknown Base ids and SHA mismatches.

## Rights

A Base contribution must declare its code/model license and training-data rights status. `training_data.rights_confirmed` must be true before the Base can be accepted.
