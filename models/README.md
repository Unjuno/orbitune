# Local training outputs

`models/` is a local workspace for checkpoints that are still being trained, compared, or rejected.

Accepted Base models are committed under:

```text
bases/<base-id>/
  manifest.json
  model.pt
  web.onnx
  README.md
```

Typical flow:

```text
models/candidate.pt
  -> evaluate / export
  -> python scripts/add_base.py ...
  -> bases/<base-id>/
```

The `.gitignore` policy keeps arbitrary training outputs under `models/` out of Git history. Only deliberately staged Base artifacts under `bases/` are repository-managed.
