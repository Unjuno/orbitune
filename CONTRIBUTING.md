# Contributing

## Choose the right change

Runtime source belongs under `orbitune/`; bounded regression tests belong under `tests/`. Keep corpus acquisition, experiment runs, and deployment changes separate from documentation fixes. See [the documentation index](docs/README.md).

Legacy [Base contributions](CONTRIBUTING_BASES.md) and [Adapter contributions](CONTRIBUTING_ADAPTERS.md) have their own manifest, ABI, size, and rights contracts. Research documentation in `models/` is not automatically eligible for those registries.

## Development

Use a virtual environment and install from the repository root:

```bash
python -m pip install -e ".[dev]"
python scripts/check_publication.py
python -m pytest -q tests/test_publication.py
python -m pytest -q
```

The wider suite may need platform-specific facilities. Report exact pass/fail/skip/block counts and environment details; do not call blocked tests passed. Do not start a corpus-scale training run just to validate a source change.

## Before opening a PR

Review `git diff --stat`, `git diff --check`, and the actual staged content. Include the change purpose, tests, known limitations, and any public-facing CLI or schema implications. Keep source code, documentation, and fixtures small and reviewable.

Never commit access tokens, cookies, private keys, raw third-party corpora, indexes, or private run outputs. `.gitignore` is not a substitute for inspecting already tracked files and commits. Publish a model binary only through an explicitly reviewed artifact-release process.

Preserve frozen checkpoint bytes and historical evidence. Clarify mistakes in an additional audit or current documentation rather than silently manufacturing a successful historical run. Test prospective resume fixes without altering saved model identity.

Source-code licensing does not grant rights to training data or model artifacts. Keep commercial and research-NC lineage declarations separate and follow [publication boundaries](docs/PUBLICATION.md).

For potential vulnerabilities or credentials, use [SECURITY.md](SECURITY.md), not a public PR containing the sensitive material.
