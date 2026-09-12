# Publication status and boundaries

## Scope

The repository is public source code with documentation of trained research models. Source and model licenses remain separate. The completed A2-512 checkpoint is published externally on Hugging Face; model binaries remain outside Git.

The historical 100k checkpoint remains documentation-only. The separately identified A2-512 checkpoint is distributed under CC-BY-NC-SA-4.0 for research/non-commercial use. This release does not publish training data, an ONNX browser export, or a Compound Adapter ABI.

## Public artifact contract

| Artifact | Status |
| --- | --- |
| Python/runtime source | Available in the repository under the existing source-code license |
| Research model documentation | Historical record under `models/research_nc_aria_gigamidi_v1/`; A2-512 release under `models/research_nc_aria_gigamidi_a2_512_v1/` |
| Compound browser runtime source | Available; separate native-stream V2 ABI with fail-closed publication config |
| Compound LoRA policy | Available; public compatibility policy, **not** a frozen Compound Adapter ABI |
| A2-512 research checkpoint | [Published on Hugging Face](https://huggingface.co/Unjuno/orbitune-a2-512); exact SHA and Hub revision recorded in Git |
| Historical 100k checkpoint | Documentation-only; not distributed |
| Compound ONNX stream/decoder graphs | Locally validated/referenced by handoff work; **not distributed by this repository** |
| Compound Web model variants | None configured; generation remains disabled until a separately reviewed release |
| Production Compound Adapter binaries | Not accepted/published yet; Compound Adapter ABI is not frozen |
| Training data and memmap indexes | Not redistributed |
| Local audit JSONs / generated MIDI package | Referenced by historical reports; not bundled public evidence |

`models/research_nc_aria_gigamidi_v1/publication.json` remains the documentation-only model publication record. It disallows a model download URL or public-registry eligibility in its current status. The browser-side [runtime config](../web/compound-runtime-config.json) likewise contains an empty `variants` list and `redistribution_review: pending`.

`models/research_nc_aria_gigamidi_a2_512_v1/manifest.json` records the separately published checkpoint, immutable artifact SHA-256, byte size, lineage, validation protocol and pinned Hugging Face revision. Publication of the PyTorch checkpoint does not make it a browser-runtime variant.

The browser code adds another hard gate rather than weakening this boundary: an available Compound variant is rejected unless redistribution review is explicitly marked complete, publication status is switched to the published state, the variant is bound to the exact Base checkpoint SHA, and its architecture/tokenizer/runtime ABI match. See [Compound browser runtime](COMPOUND_WEB_RUNTIME.md).

The original research `manifest.json` is retained unchanged as a historical run record. It is not compatible with the strict [legacy Base schema](../schemas/base_manifest.schema.json): its additional fields, absent ONNX hash, zero ONNX bytes, and oversized checkpoint must not be hidden by loosening that production schema.

## Base pretraining and Adapter publication

For Compound, Base pretraining and LoRA adaptation are separate publication stages.

```text
mutable full-parameter training state
→ completed/evaluated checkpoint
→ immutable Base id + SHA-256
→ versioned Compound Adapter ABI
→ frozen-Base LoRA training
→ Adapter / merged-variant evaluation
→ rights review
→ optional artifact publication
```

A mutable continuation checkpoint must not become a public Adapter compatibility target. Intermediate checkpoints may be used for local research, but public compatibility requires an immutable Base identity.

The existing Theory-REMI `orbitune-lora-v0` ABI is not a shortcut around this gate. Compound target modules, rank/scaling and serialization must be frozen under a new ABI after the exact target Base is selected. See [Compound LoRA policy](COMPOUND_LORA_POLICY.md).

An Adapter cannot broaden the distribution rights of its Base. A research-NC Base yields research-NC/noncommercial Adapter/merged descendants at most; permissive Adapter training data does not convert the Base lineage into a commercial one.

## Corrections made for public readers

- Removed nonexistent `generate --seed` and `resume --allow-runtime-change` usage examples; current CLI arguments are tested.
- Separated approximately 4.069 billion corpus target pairs from actual training exposure and unique coverage.
- Preserved the historical sampler-RNG limitation; current fixes do not retroactively repair saved state.
- Corrected the 126-song discrepancy being added as another rejection category. Missing inputs remain a limitation, not a proved quality exclusion.
- Marked local evidence as local instead of presenting missing files as public deliverables.
- Separated the Apache-2.0 code license, the reported NC checkpoint declaration, source terms, Adapter terms and generated-output rights. No license is changed by source/runtime cleanup.
- Separated the native Compound browser ABI from the legacy Theory-REMI Web/LoRA ABI; Compound LoRA must not silently reuse the legacy Adapter contract.
- Marked older design/handoff documents as historical through the current documentation index rather than treating their pre-implementation `NEXT` lists as present status.

## Before a Base weight or ONNX release

A maintainer should close these gates for the exact bytes being released:

1. Verify checkpoint hash/size from the artifact, parent identities, loadability, and runtime compatibility. Verify the exact exported stream/decoder graph hashes and sizes against the artifact intended for release.
2. Review rights and required attribution for every source and for checkpoint/derived-model redistribution. Preserve research-NC restrictions. Do not infer redistribution permission merely because runtime source is public.
3. Provide a real, versioned artifact location with CORS behavior suitable for browser loading, plus exact SHA-256 values. The browser must verify bytes before creating ONNX sessions.
4. Attach suitably redacted, source-addressable evaluation and provenance evidence. Resolve or explicitly retain limitations on missing inputs, source accounting, dtype/losslessness, split isolation, and historical resume fidelity. Do not rename missing evidence to PASS.
5. Test the exact released stream/decoder pair in a clean onnxruntime-web/WASM environment and run a native-generation golden fixture. Distinguish technical MIDI validity from listening/quality claims.
6. Change `redistribution_review` and `publication_status` only as part of that reviewed release, then add a fully ABI-bound Base or pre-merged-LoRA variant. The current source tree must remain generation-disabled before that point.
7. Keep the legacy Base/Adapter registry unchanged unless a separately reviewed artifact actually satisfies its different ABI and artifact contract.

## Before a Compound LoRA / Adapter release

In addition to any applicable Base-release gates:

1. Select one immutable Base model id and exact checkpoint SHA-256.
2. Freeze a new versioned Compound Adapter ABI; do not use `orbitune-lora-v0`.
3. Freeze exact target-module names/shapes, rank/alpha/scaling and Safetensors metadata/tensor layout.
4. Train with Base weights frozen and record enough configuration/source identity to reproduce the Adapter run.
5. Verify strict rejection of wrong-Base, wrong-ABI, missing and duplicate Adapter tensors.
6. Record held-out evaluation and generated-MIDI evidence for Base-only versus Adapter behavior.
7. Review Adapter training-data rights separately and confirm that Adapter terms do not exceed the Base's distribution scope.
8. If a browser variant is offered, merge only against the exact Base, export the matched V2 graph pair and rerun native/Web parity on those exact merged bytes.
9. Publish real hashes/URLs and Adapter identity; never use an unversioned mutable training checkpoint as the dependency.

Until these conditions are met, Compound LoRA remains an experimental/local adaptation path rather than a public community Adapter release channel.

Source/runtime/Adapter-policy development can finish while artifact-release gates remain open.

## Source hygiene and its limits

[check_publication.py](../scripts/check_publication.py) reads tracked files, checks selected public documentation links, validates the publication record, and detects common credential and local-artifact mistakes. [Tests](../tests/test_publication.py) cover its failure cases. [Security guidance](../SECURITY.md) describes safe reporting.

The scan is **not** a comprehensive secret scanner, malware analysis, legal review, or Git-history audit. `.gitignore` prevents some future accidental additions; it does not remove already committed data or revoke leaked credentials. Historical audit files and existing source pins are preserved rather than rewritten.

The Web runtime adds unit tests for numeric ABI rules, MIDI conversion, model-variant binding, allowed URL schemes, and release-gate state. These tests validate source contracts only; without a released ONNX graph pair they are not an independent re-verification of model quality or ONNX parity.

## Local checks

```bash
python -m pip install -e ".[dev]"
python scripts/check_publication.py
python -m pytest -q tests/test_publication.py
node --test web/*.test.mjs
```

The publication check performs no dataset processing, weight deserialization, Adapter training or Base training. Results apply to the inspected checkout, not to unavailable local training artifacts.
