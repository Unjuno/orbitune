# Reproducibility status

## Evidence boundary

The [historical manifest](manifest.json) records a locally frozen step-100,000 checkpoint with SHA-256:

```text
8bf20a1198c4f5ee086ca13fd89521af6cfaa1fb28dd6602b1eb8f0b104629dc
```

This cleanup changes documentation and publication checks only. It does not load, rewrite, retrain, or independently rehash that local checkpoint. The binary and the local `evidence/*.json` files referenced by the training report are not bundled public evidence. Do not treat their relative names as working download links.

The historical manifest is preserved byte-for-byte. It contains additional local metadata, a null/zero placeholder ONNX artifact, and a checkpoint above the public Base-artifact size limit. It must not be passed off as a schema-valid entry for the legacy `bases/` registry. [publication.json](publication.json) supplies a separate documentation-only contract.

## Historical sampler-resume defect

The checkpoint builder previously saved the local sampler RNG under `python_rng_state` and left `sampler_rng_state` empty. Restoring only the global Python random module did not restore the independent sampler `random.Random` instance. The recorded frozen continuation is therefore **not bit-exact**.

Current source separates global Python RNG state from sampler-local state, and regression tests exercise uninterrupted versus resumed sampling. This fix is prospective. Do not claim the frozen binary has a populated, corrected sampler state merely because current source is fixed. Legacy checkpoints remain loadable, but absent state cannot be recovered by relabeling metadata.

Reinitializing an RNG can repeat its random-number stream. It does not by itself prove identical MIDI songs/windows were replayed across different corpora, eligibility sets, or batch shapes. Exact data-exposure claims need the actual sample IDs.

Tests: [sampler RNG regression tests](../../tests/test_sampler_rng_resume.py). Validation-corpus changes also require separately scoped best-loss state; losses from different validation sets are not directly comparable.

## Data and evaluation caveats

[training_data.md](training_data.md) records missing input files, OverflowError exclusions, the unresolved 126-song discrepancy, and the distinction between corpus size and training exposure. Accounting-counter balance and file hashes alone do not demonstrate intended-source completeness or lossless uint8 encoding.

The training report summarizes 30 parse-valid MIDI samples. No listening-panel result, broad benchmark, or complete public sample package is established here. Treat the reported metrics as technical observations, not a musical-quality guarantee.

## Test reporting

The historical local report stated 254 passed tests, five Windows-related failures, and one missing dependency (`jsonschema`). These are historical observations, not the current CI result. Failed and environment-blocked tests are not counted as passes, and the attribution of a failure requires evidence.

Use the checks attached to the actual PR/commit for current source validation. The publication workflow does not run corpus downloads or real-model training and does not validate the unpublished weight artifact.

## Future reproduction requirements

Preserve the exact checkpoint and parent hashes, source code commit and dirty diff, training configuration, dataset revisions, source and retained manifests, split policy, representation/dtype, index identities, sampler state, and validation protocol. Report any absent historical field as unknown.

For this recorded model, the published metadata is enough to identify the claimed artifact, but not to independently reproduce or certify the entire training run. [Publication status](../../docs/PUBLICATION.md) lists the remaining weight-release prerequisites.
