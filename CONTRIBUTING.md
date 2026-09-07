# Contributing

## Choose the right change

Runtime source belongs under `orbitune/`; bounded regression tests belong under `tests/`. Keep corpus acquisition, experiment runs, model artifacts, and deployment changes separate from documentation fixes. See [the documentation index](docs/README.md).

Orbitune currently has two contribution tracks:

- legacy [Base contributions](CONTRIBUTING_BASES.md) and [Adapter contributions](CONTRIBUTING_ADAPTERS.md), which have an existing Theory-REMI manifest/ABI/artifact contract;
- Compound model/runtime development, whose research model documentation lives under `models/` and whose LoRA policy is defined separately in [docs/COMPOUND_LORA_POLICY.md](docs/COMPOUND_LORA_POLICY.md).

Research documentation in `models/` is not automatically eligible for the legacy `bases/` or `adapters/` registries. Do not route Compound artifacts through the Theory-REMI compatibility schema merely to make validation pass.

## Development

Use a virtual environment and install from the repository root:

```bash
python -m pip install -e ".[dev]"
python scripts/check_publication.py
python -m pytest -q tests/test_publication.py
python -m pytest -q
node --test web/*.test.mjs
```

The wider suite may need platform-specific facilities. Report exact pass/fail/skip/block counts and environment details; do not call blocked tests passed. Do not start a corpus-scale training run just to validate a source change.

## Base pretraining versus LoRA work

For the Compound family, full-parameter Base pretraining and LoRA adaptation are intentionally separate stages. A mutable long-run checkpoint is not a public Adapter compatibility target. Freeze and document the exact Base checkpoint first; only then may a versioned Compound Adapter ABI be frozen and used for public Adapter compatibility.

Until that ABI exists, Compound LoRA contributions should be limited to clearly experimental code, target/rank measurements, tests and compatibility proposals. Public Compound Adapter binaries and merged derivatives remain gated by [the Compound LoRA policy](docs/COMPOUND_LORA_POLICY.md) and [publication boundaries](docs/PUBLICATION.md).

## Before opening a PR

Review `git diff --stat`, `git diff --check`, and the actual staged content. Include the change purpose, tests, known limitations, and any public-facing CLI or schema implications. Keep source code, documentation and fixtures small and reviewable.

Never commit access tokens, cookies, private keys, raw third-party corpora, indexes, private run outputs, or unreviewed model binaries. `.gitignore` is not a substitute for inspecting already tracked files and commits. Publish a model or derived binary only through an explicitly reviewed artifact-release process.

Preserve frozen checkpoint bytes and historical evidence. Clarify mistakes in an additional audit or current documentation rather than silently manufacturing a successful historical run. Test prospective resume fixes without altering saved model identity.

Source-code licensing does not grant rights to training data, Base checkpoints, Adapters or merged model artifacts. Keep commercial and research-NC lineage declarations separate. An Adapter cannot broaden the permissions of its Base.

For potential vulnerabilities or credentials, use [SECURITY.md](SECURITY.md), not a public PR containing the sensitive material.
