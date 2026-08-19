# Continuous training dataset gate

The scheduled `continuous-train` workflow is intentionally idle until dataset research is complete.

To arm real training, provide two non-empty Theory-REMI token corpora:

```text
data/continuous/train.tokens
data/continuous/validation.tokens
```

Do not place unreviewed or license-unclear data here. The final data source must be documented with provenance, license/rights status, deduplication procedure, and train/validation separation.

Once both files exist, GitHub Actions resumes the previous 10M training state every six hours. Mutable optimizer/model state is stored in Actions cache and mirrored to the repository prerelease tag `continuous-training`; it is not a published Base compatibility target.
