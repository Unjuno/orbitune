# Security

## Report safely

Do not include tokens, credentials, private dataset content, or exploit payloads in a public issue or PR. Use GitHub's private "Report a vulnerability" option if it is enabled for this repository. If it is unavailable, ask the maintainer for a private reporting channel without posting sensitive details. No response-time or supported-version guarantee is made by this document.

If a credential was exposed, revoke or rotate it with its provider. Deleting a file from the latest commit or adding an ignore rule does not remove it from history or invalidate it.

## Model and data artifacts

Load only trusted checkpoints. PyTorch checkpoint loading can deserialize Python objects; a documented SHA-256 verifies identity against a trusted reference, not safety of an arbitrary producer. See [PyTorch's loading warning](https://docs.pytorch.org/docs/stable/generated/torch.load.html).

Do not bypass gated access or redistribute datasets with source-code releases. Keep tokens in appropriate local credential storage, never command examples with real values or committed logs.

Training, dataset builds, infrastructure deployment, model uploads, and source-only checks are separate operations. Source-only CI must not implicitly start a large training job or publish model/data artifacts.

## Automated checks

The publication checker detects common tracked credential paths, token-like strings, and large/local artifacts without printing matched secret values. It is a limited prevention check on the current tracked tree, not a complete secret/history or security audit.
