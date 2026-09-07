# Publication status and boundaries

## Scope

The repository is public source code with documentation of a locally trained research model. **Source publication is not weight publication.** This cleanup does not upload checkpoints, raw corpora, indexes, or private evidence, and does not change repository visibility or create a release tag.

Baseline reviewed: `bc190faaadf13d07e0ac211ba87a8f21af5f7d57` (PR #44). At that review, the GitHub Releases API returned no releases and the research checkpoint was not tracked. No new model download URL is invented.

## Public artifact contract

| Artifact | Status |
| --- | --- |
| Python/runtime source | Available in the repository under the existing source-code license |
| Research model documentation | Available under `models/research_nc_aria_gigamidi_v1/` |
| Research checkpoint | Reported as locally frozen; not distributed by this repository |
| Compound ONNX / browser model | Not published |
| Training data and memmap indexes | Not redistributed |
| Local audit JSONs / generated MIDI package | Referenced by historical reports; not bundled public evidence |

`models/research_nc_aria_gigamidi_v1/publication.json` uses the new [documentation-only schema](../schemas/research_publication.schema.json). It explicitly disallows a download URL or public-registry eligibility in this status.

The original research `manifest.json` is retained unchanged as a historical run record. It is not compatible with the strict [legacy Base schema](../schemas/base_manifest.schema.json): its additional fields, absent ONNX hash, zero ONNX bytes, and oversized checkpoint must not be hidden by loosening that production schema.

## Corrections made for public readers

- Removed nonexistent `generate --seed` and `resume --allow-runtime-change` usage examples; current CLI arguments are tested.
- Separated approximately 4.069 billion corpus target pairs from actual training exposure and unique coverage.
- Preserved the historical sampler-RNG limitation; current fixes do not retroactively repair saved state.
- Corrected the 126-song discrepancy being added as another rejection category. Missing inputs remain a limitation, not a proved quality exclusion.
- Marked local evidence as local instead of presenting missing files as public deliverables.
- Separated the Apache-2.0 code license, the reported NC checkpoint declaration, source terms, and generated-output rights. No license is changed by this cleanup.

## Before a weight release

A maintainer should close these gates for the exact bytes being released:

1. Verify checkpoint hash/size from the artifact, parent identities, loadability, and runtime compatibility. Provide a real, versioned download location and checksum file; do not upload weights through this source-cleanup PR.
2. Review rights and required attribution for every source and for checkpoint redistribution. Preserve research-NC restrictions. Do not infer a blanket output license or broader commercial permission from the source-code license.
3. Attach suitably redacted, source-addressable evaluation and provenance evidence. Resolve or explicitly retain limitations on missing inputs, source accounting, dtype/losslessness, split isolation, and historical resume fidelity. Do not rename missing evidence to PASS.
4. Test generation in a clean environment with the exact released checkpoint. Distinguish technical MIDI validity from listening/quality claims.
5. Keep the legacy Base registry unchanged unless a separately reviewed, compatible export actually satisfies its ABI and artifact contract.

Source cleanup can finish while these weight-release gates remain open.

## Source hygiene and its limits

[check_publication.py](../scripts/check_publication.py) reads tracked files, checks selected public documentation links, validates the publication record, and detects common credential and local-artifact mistakes. [Tests](../tests/test_publication.py) cover its failure cases. [Security guidance](../SECURITY.md) describes safe reporting.

The scan is **not** a comprehensive secret scanner, malware analysis, legal review, or Git-history audit. `.gitignore` prevents some future accidental additions; it does not remove already committed data or revoke leaked credentials. This change preserves old audit files and existing source pins rather than rewriting history.

## Local checks

```bash
python -m pip install -e ".[dev]"
python scripts/check_publication.py
python -m pytest -q tests/test_publication.py
```

The publication check performs no network access, dataset processing, weight deserialization, or training. Results apply to the inspected checkout, not to unavailable local training artifacts.
