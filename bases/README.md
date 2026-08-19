# Orbitune Bases

Base models are managed in this repository alongside adapters.

Each Base has its own immutable identity:

```text
bases/<base-id>/
  manifest.json
  model.pt
  web.onnx
  README.md
```

Rules:

- `base-id` is a stable compatibility lineage, not a rolling version slot.
- Once adapters are published against a Base checkpoint SHA-256, that checkpoint must never be replaced in-place.
- A contributor may add another Base under a new id.
- Each individual checkpoint/ONNX file must remain below 95 MiB in the current repository policy.
- The current rank-4 LoRA ABI supports Bases implementing `orbitune-midi-gpt-v0` / `theory-remi-v0` with the 4x240 model shape. Other Base architectures may be registered, but require their own Adapter ABI before LoRA adapters are accepted.
- `manifest.json` records exact checkpoint and web ONNX SHA-256 values.

`registry/bases.json` is generated from these directories; do not hand-edit generated registry entries after the builder is enabled in CI.
