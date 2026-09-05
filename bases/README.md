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
- The current legacy/reference rank-4 LoRA ABI supports Bases implementing `orbitune-midi-gpt-v0` / `theory-remi-v0` with the 4x448 model shape. The experimental Compound production direction is not yet a frozen Base/Adapter ABI. Other Base architectures may be registered, but require a compatible Adapter ABI and runtime implementation before adapters are accepted or browser inference is enabled.
- `manifest.json` records exact checkpoint and web ONNX SHA-256 values.
- `manifest.json` also records immutable corpus/checkpoint rights lineage: parent checkpoint identity, corpus registry and manifest SHA-256, commercial eligibility, distribution scope, license policy, restricted source ids, and a rights summary.
- `commercial_eligible=true` is only valid for `license_policy=prod-only` with `distribution_scope=commercial` and no restricted source ids.
- A Base containing `research-nc` ancestry must remain `commercial_eligible=false`; those weights must not be backported, merged, or distilled into a commercial-eligible lineage.
- `bases/` and the generated Base registry are public distribution paths. `restricted` or `internal-only` checkpoints must not be committed here; registry generation fails closed if one is present.
- A noncommercial Base must use checkpoint licensing/terms that are compatible with its noncommercial distribution scope.

`registry/bases.json` is generated from these directories; do not hand-edit generated registry entries.
