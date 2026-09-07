# Using an independently obtained checkpoint

## Availability and trust

The repository does **not** include the trained checkpoint and does not currently provide a public download URL for it. The commands below are for a user who already has an authorized local copy. They do not download data or start training.

Install the source checkout as described in the [root README](../../README.md). Run commands from the repository root. Only load checkpoints from a trusted source: [PyTorch warns that checkpoint loading uses an unpickler](https://docs.pytorch.org/docs/stable/generated/torch.load.html). A matching hash detects byte differences against a trusted reference; it is not a safety review of an unknown producer.

## Verify bytes without loading the model

Windows PowerShell:

```powershell
$checkpoint = 'path/to/model.pt'
$expected = '8bf20a1198c4f5ee086ca13fd89521af6cfaa1fb28dd6602b1eb8f0b104629dc'
$actual = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw 'Checkpoint SHA-256 mismatch; do not load it.' }
```

Portable Python check (all platforms):

```python
import hashlib
from pathlib import Path
checkpoint = Path("path/to/model.pt")
expected = "8bf20a1198c4f5ee086ca13fd89521af6cfaa1fb28dd6602b1eb8f0b104629dc"
hash_state = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        hash_state.update(block)
if hash_state.hexdigest() != expected:
    raise SystemExit("Checkpoint SHA-256 mismatch; do not load it.")
print("Checkpoint hash matches the documented reference.")
```

## Inspect and generate

Replace `path/to/model.pt` with the verified local path. These flags are defined by the checked-in `orbitune.compound_cli` parser:

```bash
orbitune-compound info --checkpoint path/to/model.pt
orbitune-compound generate --checkpoint path/to/model.pt --out generated.mid --events 512 --device cpu --temperature 0.85 --top-p 0.92
```

For a MIDI primer:

```bash
orbitune-compound generate --checkpoint path/to/model.pt --primer-midi prompt.mid --out continuation.mid --events 512 --device cpu --temperature 0.85 --top-p 0.92
```

With a compatible CUDA-enabled environment, `--device cuda` is available. CPU is the portable documented starting point; macOS GPU acceleration and Compound browser/ONNX compatibility are not claimed here.

The current public `generate` command has **no `--seed` flag**. Do not add it to these commands. The historical fixed-seed quality report used separate local tooling. This cleanup checks argument compatibility in CI; it cannot test inference against weights that are not distributed here.

## Continuing training

Inference does not require the training corpus. Training resumption does. Do not point a resume command at an immutable baseline and assume it writes elsewhere; inspect the specific trainer's output and step semantics first.

The public `orbitune-compound resume` parser does **not** accept `--allow-runtime-change`. The CFE training helper has a different interface. Consult the relevant `--help`, use a new output/run, preserve the exact parent hash and noncommercial lineage, and validate corpus/tokenizer/runtime compatibility.

Current code can save sampler-local RNG state for future checkpoints. That does not add missing RNG state to the frozen checkpoint. Legacy state must be inspected and resume fidelity reported honestly. See [reproducibility.md](reproducibility.md).

## Rights

The documented research model remains noncommercial under project policy. The historical checkpoint license declaration is CC-BY-NC-SA-4.0. No public release or additional license grant is made by these instructions; generated-output rights are not automatically established by this guide.
