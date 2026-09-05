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
- `bases/` is a **public distribution path**; restricted/internal-only checkpoints must never be staged or committed here

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

## Checkpoint rights lineage

Every Base must carry an explicit `lineage` object. This is separate from the license on the checkpoint bytes themselves.

Required lineage fields are:

```json
{
  "parent_checkpoint": null,
  "commercial_eligible": true,
  "distribution_scope": "commercial",
  "license_policy": "prod-only",
  "corpus_registry": "configs/pretrain_corpus_commercial_v5.json",
  "corpus_manifest_sha256": "<64-char sha256>",
  "restricted_source_ids": [],
  "rights_summary": "PROD-only commercial-safe corpus"
}
```

`parent_checkpoint`, when present, must contain both a Base id and the exact parent checkpoint SHA-256.

Allowed lineage policies are:

- `prod-only`: only commercial-safe/PROD ancestry;
- `research-nc`: includes non-commercial research ancestry;
- `restricted`: internal/restricted ancestry whose checkpoint must not enter this public repository path.

Allowed distribution scopes are:

- `commercial`;
- `noncommercial`;
- `internal-only`.

The manifest validator enforces the following hard boundaries:

- `commercial_eligible=true` requires `license_policy=prod-only`;
- `commercial_eligible=true` requires `distribution_scope=commercial`;
- a commercial-eligible Base may not list restricted source ids;
- `research-nc` and `restricted` Bases must be `commercial_eligible=false`;
- a `restricted` Base must use `distribution_scope=internal-only`;
- a noncommercial/internal-only Base may not use a standard checkpoint license such as Apache-2.0, MIT, GPL, CC0 or CC-BY that grants commercial use;
- a commercial Base may not declare a checkpoint license containing a noncommercial restriction.

The public registry adds a second fail-closed boundary: `restricted` or `internal-only` Base manifests are rejected even if their generic manifest is otherwise structurally valid. Internal checkpoints belong in a separate, non-public artifact store and are not staged with `scripts/add_base.py`.

This metadata does not manufacture rights. It records the result of the source/corpus audit and makes lineage mistakes mechanically visible.

A research-NC or restricted checkpoint must never be merged, distilled, backported, or otherwise represented as part of a commercial-eligible lineage.

## Staging a current legacy/reference ABI Base

After training and ONNX export, stage only after you have actually checked the model/data rights. The staging command requires explicit rights and lineage declarations and will not manufacture `rights_confirmed=true` implicitly.

Commercial example:

```bash
python scripts/add_base.py \
  --id my-commercial-base \
  --display-name "My Commercial Base" \
  --checkpoint models/my-base.pt \
  --web-onnx my-base-web.onnx \
  --license Apache-2.0 \
  --training-license commercial-safe \
  --rights-confirmed \
  --commercial-eligible true \
  --distribution-scope commercial \
  --license-policy prod-only \
  --corpus-registry configs/pretrain_corpus_commercial_v5.json \
  --corpus-manifest-sha256 <sha256> \
  --rights-summary "PROD-only commercial-safe corpus"
```

Research-NC descendant example:

```bash
python scripts/add_base.py \
  --id my-research-nc-base \
  --display-name "My Research-NC Base" \
  --checkpoint models/my-research.pt \
  --web-onnx my-research-web.onnx \
  --license CC-BY-NC-SA-4.0 \
  --training-license "mixed PROD + NC research" \
  --rights-confirmed \
  --commercial-eligible false \
  --distribution-scope noncommercial \
  --license-policy research-nc \
  --corpus-registry configs/pretrain_corpus_research_nc_v1.json \
  --corpus-manifest-sha256 <sha256> \
  --parent-base-id my-commercial-base \
  --parent-checkpoint-sha256 <parent-checkpoint-sha256> \
  --restricted-source-id gigamidi \
  --rights-summary "Research-only descendant; contains non-commercial training data"
```

The research example uses an explicitly noncommercial checkpoint license so the checkpoint license does not contradict `distribution_scope=noncommercial`. A source-specific audit may require different noncommercial/custom terms; do not infer checkpoint licensing from the dataset license alone.

This copies the public-distributable artifacts into `bases/<id>/`, computes SHA-256 values, validates the rights contract, and creates the manifest/README. `--id` is validated before any output path is created, so Base ids cannot escape the configured output root. The staging CLI rejects `restricted` and `internal-only` Bases before it copies any artifacts.

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

`lineage.corpus_registry` and `lineage.corpus_manifest_sha256` bind the Base to the exact corpus state that produced it. `commercial_eligible`, `distribution_scope`, and `license_policy` describe the resulting checkpoint's permitted lineage; they do not override source licenses or applicable law.

For official Orbitune Bases, rights confirmation is necessary but not sufficient: the corpus pipeline must also record provenance, parsing/quality filtering, normalized-event deduplication and train/validation separation. Exact-byte hash grouping alone is not considered production deduplication.
