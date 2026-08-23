# Continuous training dataset gate

The scheduled `continuous-train` workflow is intentionally idle by default.

It currently trains the **legacy Theory-REMI reference path**, not the experimental Compound production path. This distinction is deliberate: reviewed Compound corpus files must not accidentally start an obsolete training loop.

To arm legacy/reference continuous training intentionally, all three files must exist and be non-empty:

```text
data/continuous/ENABLE_LEGACY_REFERENCE_TRAINING
data/continuous/train.tokens
data/continuous/validation.tokens
```

Do not add the marker merely to make CI run. The token corpora must be reviewed for provenance/license, parsing quality, deduplication and train/validation separation.

Once explicitly armed, GitHub Actions resumes the previous legacy ~10M training state every six hours. Mutable optimizer/model state is stored in Actions cache and mirrored to the `continuous-training` prerelease. That mutable state is never a published Base compatibility target.

The future Compound continuous-training path must have a separate dataset/ABI gate after the Compound Base model, production corpus and checkpoint format are accepted.
