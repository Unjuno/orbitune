# Orbitune Audit Continuation — 2026-08-24

Status: **ACTIVE CONTINUATION RECORD**

Read after `docs/AUDIT_2026-08-23.md`. This file records issues found and fixed after the first full-repository audit, plus the initial `Unjuno/gpu-control` / RunPod workload integration.

## Executive result

The continued audit found several non-model failures that would have affected contribution CI, Pages deployment, reproducibility, continuous training, and Base staging. These were fixed before starting the next Compound-model implementation stage.

A bounded remote-GPU training canary now exists under `workloads/runpod-training-canary/`. It deliberately uses the operational legacy/reference 10.2M Theory-REMI model as an infrastructure probe; it is not production-model evidence.

## Fixed during continuation

### C1 — stale deleted downloader test

`tests/test_download_base_model.py` remained after the obsolete GitHub Release downloader was deleted. It imported a non-existent script and could break full pytest. Deleted.

### C2 — Base Web compatibility trusted strings instead of shape

`web_runtime_compatible` previously depended only on architecture/tokenizer strings. A differently shaped checkpoint could claim the same strings and be exposed to the fixed 4x448 Web runtime.

Current-reference Base validation now loads the checkpoint, validates the actual config/parameter count and marks Web compatibility true only for the known supported reference shape.

### C3 — Base manifest parameter count was not bound to checkpoint contents

The registry now compares manifest `parameter_count` with the actual loaded current-reference checkpoint.

### C4 — checkpoint allocation DoS boundary

A malicious small checkpoint could declare an enormous config and trigger model allocation before validation. `OrbituneGPT.load_checkpoint()` now validates config shape/types and enforces the repository parameter budget before instantiating the model.

### C5 — `validate-assets` missing runtime dependencies

The asset-validation workflow executed registry code whose import chain needs PyTorch/Safetensors but did not install the package/dependencies. The workflow now installs its required runtime before validation.

### C6 — Pages had the same missing-dependency problem

The Pages workflow also called the registry builder without installing the package. Fixed.

### C7 — stale CI path filter

`ml-smoke.yml` still watched deleted `orbitune/release.py`. Removed and read-only permissions were made explicit where appropriate.

### C8 — training seed did not control initial weights

`train_base()` constructed `OrbituneGPT` before applying `TrainConfig.seed`; Adapter LoRA initialization had the same problem. Same seed/data/config therefore did not imply the same initialization.

The seed is now applied before Base and LoRA construction. A deterministic Base-training regression test was added.

### C9 — continuous-training fresh initialization had the same seed bug

Fresh scheduled training seeded Torch after model creation. Fixed.

### C10 — continuous spike detector forgot history at every scheduled run

Rolling loss history and consecutive-spike state were local variables only. Every resume therefore created a new warm-up blind interval. They are now included in durable state and restored across runs, while older state remains readable.

### C11 — continuous state lacked explicit tokenizer/state schema

Durable state now records state format and tokenizer identity in addition to architecture/config.

### C12 — unrestricted pickle loads in continuous state path

State loads were moved toward restricted `weights_only=True` where the saved payload is compatible with it. Workflow snapshot inspection no longer needs unrestricted loading.

### C13 — continuous release upload could hide the actual rollback reason

If rollback happened before the first best checkpoint existed, unconditional `best.pt` upload could fail first. Persistence now treats best checkpoint as optional until it exists and preserves the actual health failure signal.

### C14 — Base staging path traversal

`scripts/add_base.py` used `--id` in the output path before validating it. Base id is now validated before output directory creation.

### C15 — Base staging manufactured rights confirmation

The staging tool previously emitted `training_data.rights_confirmed=true` without an explicit acknowledgement. It now requires `--rights-confirmed`. This flag is an acknowledgement, not provenance evidence.

### C16 — CLI numeric contract gaps

Zero/negative training steps, batch size, sequence length, invalid learning rate/weight decay/dropout and related inputs could reach deeper code and fail with unrelated exceptions. The CLI now rejects invalid values at the boundary.

### C17 — Compound canonical fields were under-specified in code

The schema documentation said unused `a1..a4` slots must be zero, but validation did not enforce that. Event-specific unused fields, global metadata channel, tempo range and time-signature denominator invariants are now enforced.

## Initial gpu-control / RunPod workload

The companion control-plane repository `Unjuno/gpu-control` explicitly requires staged escalation:

```text
local/container minimum
→ immutable public source verification
→ isolated container verification
→ dry run
→ live price verification
→ cleanup guarantee
→ explicit paid authorization
→ ApprovedExecutionPlan
→ provider execution
```

Its initial `cheap-24gb` policy allows one GPU with at least 24 GB VRAM, at most 30 minutes and at most $0.30. Live provider calls are still gated in `gpu-control`; Orbitune does not bypass those gates.

Orbitune now supplies the workload side:

```text
workloads/runpod-training-canary/Dockerfile
workloads/runpod-training-canary/run.py
workloads/runpod-training-canary/README.md
.github/workflows/runpod-canary-smoke.yml
```

### Fixed canary training volume

```text
model                legacy/reference OrbituneGPT
parameters           10,200,960
steps                250
batch                 8
sequence              256
training tokens       512,000
validation interval   50 steps
validation points     5
seed                  20260824
```

Rationale: this amount is sufficient to sustain CUDA forward/backward, AdamW, validation and checkpoint I/O while remaining tiny relative to pretraining. The first paid run must not be enlarged merely because a GPU can fit a larger batch/model.

### Canary data

The canary generates deterministic synthetic Theory-REMI tokens internally. It has no runtime dataset download or private-data dependency. Its losses are **not model-quality evidence**.

### Container contract

The Docker base is PyTorch 2.10.0 CUDA 12.8/cuDNN 9 pinned by immutable manifest digest. Runtime entrypoint is finite and writes:

```text
/outputs/result.json
/outputs/canary-base.pt
```

The result records device/CUDA identity, throughput, peak allocated VRAM, training/validation losses and checkpoint digest metadata.

### Paid canary PASS

A provider run is not accepted merely because the workload exits zero. Required evidence includes:

```text
result.status == pass
result.device_type == cuda
result.cuda_available == true
parameters == 10,200,960
tokens_processed == 512,000
5 finite validation points
final validation < first validation
checkpoint SHA-256 matches
provider lifecycle finalized
cleanup confirmed
```

CPU PASS is only a local/CI workload-contract smoke.

## Remaining audit work

### R1 — contributed ONNX semantic validation

Current staging/registry validation binds checkpoint SHA/config more strongly than Web ONNX semantics. A contributed current-reference ONNX should be checked for expected inputs, dimensions, output vocabulary shape and a minimal execution before it is marked Web compatible.

### R2 — browser Safetensors parser should fail closed

The JavaScript parser is looser than Python packaging validation for malformed/overlapping ranges, unexpected tensors and non-finite values. Harden before treating arbitrary user-supplied adapters as trusted runtime input.

### R3 — JSON Schema / Python validator exact parity

Keep manifest schema and Python validators mechanically aligned. Differences are currently small but two independent contracts should not drift.

### R4 — action dependencies are version-tag pinned

GitHub Actions still use tags such as `actions/checkout@v4`. Commit-SHA pinning remains a supply-chain hardening option.

### R5 — CI green after the latest audit commits

The available connector exposes commit status but not all push-triggered Action runs reliably. Inspect the actual Actions UI/logs before declaring the complete latest suite green.

## Model/research blockers unchanged

The infrastructure canary does not resolve the production-model blockers:

1. timing above 1536 steps without silent clipping;
2. Compound BOS/EOS/start semantics;
3. one field-cardinality/mask schema;
4. Compound Base + masked losses;
5. synthetic then tiny-real overfit;
6. real-corpus timing/pedal/provenance/dedup validation;
7. 5M/10M/20M real-MIDI scale sweep;
8. long-memory/ControlField/device quantization validation;
9. policy-learning necessity only after Base rollout.

## Correct next separation of work

Two tracks may now proceed independently:

```text
Track A — infrastructure
  runpod-canary-smoke CI
  gpu-control image-publish/live adapter gates
  bounded 512k-token paid RunPod canary

Track B — production model
  long-time experiment
  BOS/EOS + field schema
  Compound Base implementation
  synthetic/tiny-real overfit
```

Do not use success in Track A to skip correctness gates in Track B, and do not wait for the Compound model before proving the GPU infrastructure with the legacy canary.
