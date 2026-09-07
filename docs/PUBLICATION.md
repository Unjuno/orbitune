# Publication status and boundaries

## Scope

The repository is public source code with documentation of a locally trained research model. **Source publication is not weight publication.** The repository may contain browser/runtime code that knows how a future verified model artifact should be loaded, while the model binary itself remains unavailable.

The documented research checkpoint is not tracked, no release asset is created by the current runtime work, and no model download URL is invented. Repository/runtime changes must not be interpreted as permission to redistribute the research weights or training data.

## Public artifact contract

| Artifact | Status |
| --- | --- |
| Python/runtime source | Available in the repository under the existing source-code license |
| Research model documentation | Available under `models/research_nc_aria_gigamidi_v1/` |
| Compound browser runtime source | Available; separate native-stream V2 ABI with fail-closed publication config |
| Research checkpoint | Reported as locally frozen; not distributed by this repository |
| Compound ONNX stream/decoder graphs | Locally validated/referenced by handoff work; **not distributed by this repository** |
| Compound Web model variants | None configured; generation remains disabled until a separately reviewed release |
| Training data and memmap indexes | Not redistributed |
| Local audit JSONs / generated MIDI package | Referenced by historical reports; not bundled public evidence |

`models/research_nc_aria_gigamidi_v1/publication.json` remains the documentation-only model publication record. It disallows a model download URL or public-registry eligibility in its current status. The browser-side [runtime config](../web/compound-runtime-config.json) likewise contains an empty `variants` list and `redistribution_review: pending`.

The browser code adds another hard gate rather than weakening this boundary: an available Compound variant is rejected unless redistribution review is explicitly marked complete, publication status is switched to the published state, the variant is bound to the exact Base checkpoint SHA, and its architecture/tokenizer/runtime ABI match. See [Compound browser runtime](COMPOUND_WEB_RUNTIME.md).

The original research `manifest.json` is retained unchanged as a historical run record. It is not compatible with the strict [legacy Base schema](../schemas/base_manifest.schema.json): its additional fields, absent ONNX hash, zero ONNX bytes, and oversized checkpoint must not be hidden by loosening that production schema.

## Corrections made for public readers

- Removed nonexistent `generate --seed` and `resume --allow-runtime-change` usage examples; current CLI arguments are tested.
- Separated approximately 4.069 billion corpus target pairs from actual training exposure and unique coverage.
- Preserved the historical sampler-RNG limitation; current fixes do not retroactively repair saved state.
- Corrected the 126-song discrepancy being added as another rejection category. Missing inputs remain a limitation, not a proved quality exclusion.
- Marked local evidence as local instead of presenting missing files as public deliverables.
- Separated the Apache-2.0 code license, the reported NC checkpoint declaration, source terms, and generated-output rights. No license is changed by source/runtime cleanup.
- Separated the native Compound browser ABI from the legacy Theory-REMI Web/LoRA ABI; Compound LoRA must not silently reuse the legacy adapter contract.

## Before a weight or ONNX release

A maintainer should close these gates for the exact bytes being released:

1. Verify checkpoint hash/size from the artifact, parent identities, loadability, and runtime compatibility. Verify the exact exported stream/decoder graph hashes and sizes against the artifact intended for release.
2. Review rights and required attribution for every source and for checkpoint/derived-model redistribution. Preserve research-NC restrictions. Do not infer redistribution permission merely because runtime source is public.
3. Provide a real, versioned artifact location with CORS behavior suitable for browser loading, plus exact SHA-256 values. The browser must verify bytes before creating ONNX sessions.
4. Attach suitably redacted, source-addressable evaluation and provenance evidence. Resolve or explicitly retain limitations on missing inputs, source accounting, dtype/losslessness, split isolation, and historical resume fidelity. Do not rename missing evidence to PASS.
5. Test the exact released stream/decoder pair in a clean onnxruntime-web/WASM environment and run a native-generation golden fixture. Distinguish technical MIDI validity from listening/quality claims.
6. Change `redistribution_review` and `publication_status` only as part of that reviewed release, then add a fully ABI-bound Base or pre-merged-LoRA variant. The current source tree must remain generation-disabled before that point.
7. Keep the legacy Base/Adapter registry unchanged unless a separately reviewed artifact actually satisfies its different ABI and artifact contract.

Source/runtime development can finish while these artifact-release gates remain open.

## Source hygiene and its limits

[check_publication.py](../scripts/check_publication.py) reads tracked files, checks selected public documentation links, validates the publication record, and detects common credential and local-artifact mistakes. [Tests](../tests/test_publication.py) cover its failure cases. [Security guidance](../SECURITY.md) describes safe reporting.

The scan is **not** a comprehensive secret scanner, malware analysis, legal review, or Git-history audit. `.gitignore` prevents some future accidental additions; it does not remove already committed data or revoke leaked credentials. Historical audit files and existing source pins are preserved rather than rewritten.

The Web runtime adds unit tests for numeric ABI rules, MIDI conversion, model-variant binding, allowed URL schemes, and release-gate state. These tests validate source contracts only; without the unpublished model files they are not an independent re-verification of model quality or ONNX parity.

## Local checks

```bash
python -m pip install -e ".[dev]"
python scripts/check_publication.py
python -m pytest -q tests/test_publication.py
node --test web/*.test.mjs
```

The publication check performs no dataset processing, weight deserialization, or training. Results apply to the inspected checkout, not to unavailable local training artifacts.
