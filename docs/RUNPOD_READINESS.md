# Orbitune-side RunPod readiness

This document covers only the workload-repository conditions that Orbitune must satisfy before `Unjuno/gpu-control` may be considered for a paid RunPod execution. It does not authorize spending and it does not replace gpu-control policy.

## Scope

The first live workload is the legacy/reference 10.2M Theory-REMI GPU training canary under:

```text
workloads/runpod-training-canary/
```

Its purpose is infrastructure validation, not production model-quality validation.

## Required Orbitune-side conditions

### R1 — public immutable source

- repository is public;
- exact 40-character commit SHA is selected;
- the selected SHA contains the canary Dockerfile, bounded entrypoint, runner, workload contract, and smoke workflow;
- `gpu-control verify-source` succeeds against that exact SHA.

Status: **implemented; per-run SHA selection/verification remains required**.

### R2 — deterministic finite workload

- finite non-interactive entrypoint;
- deterministic seed;
- no external dataset download;
- no runtime network dependency;
- no provider secret or Docker socket inside the workload;
- bounded runtime and output size;
- meaningful non-zero exit on workload failure;
- bounded `result.json` failure evidence is written when the Python runner exits before producing its normal result.

Status: **implemented**.

### R3 — local/CI smoke before paid compute

The CPU smoke must exercise model construction, one optimizer update, validation, checkpoint save and result serialization without provider access. It also deliberately triggers a CUDA-required failure on a CPU runner and verifies that the bounded entrypoint emits a failure `result.json` while preserving a non-zero process exit.

Status: **implemented by `.github/workflows/runpod-canary-smoke.yml`; a green run on the exact candidate SHA must be observed before paid escalation**.

### R4 — immutable published image

The exact source SHA must be published as a linux/amd64 image and referenced by immutable `sha256:` digest. Mutable tags may be used only for discovery.

Publication is manual through:

```text
.github/workflows/publish-runpod-canary.yml
```

The workflow checks out the exact requested SHA, passes that SHA into the Docker build as `ORBITUNE_SOURCE_SHA`, publishes to:

```text
ghcr.io/unjuno/orbitune-runpod-canary
```

and emits `orbitune-published-image-evidence-v1` containing the exact source SHA, Dockerfile, workload id, platform and digest-pinned image reference. The image stores the same SHA in the OCI revision label and runtime environment; the successful workload result records it as `source_sha`.

Status: **publication mechanism implemented; an actual candidate image has not yet been published/verified**.

### R5 — machine-readable workload identity

`workloads/runpod-training-canary/workload.json` is the workload repository's stable machine-readable contract for resource bounds, training amount, output files and acceptance criteria. It requires baked source-SHA correlation and image-digest correlation with the future approved plan.

Status: **implemented**.

### R6 — bounded result contract

Required output directory:

```text
/outputs
```

Successful execution requires:

```text
result.json
canary-base.pt
```

If the underlying runner exits before producing its normal result, the bounded entrypoint still writes a compact failure `result.json` and returns the runner's non-zero exit code.

A successful result includes baked source identity, CUDA/device identity, training amount, timing/throughput, validation history, checkpoint size and checkpoint SHA-256. A checkpoint larger than the bounded collection policy becomes reference-only rather than silently bypassing collection limits.

Status: **implemented**.

### R7 — economic ceiling

The first full canary request is bounded by:

```text
gpu profile          cheap-24gb
GPU count            1
max runtime          30 minutes
max requested cost   $0.30
training tokens      512,000
```

These are ceilings, not spending targets. Live catalog pricing must show a cheaper/equal viable option before submission. A smaller paid micro-canary should be preferred first if it can validate CUDA/image/provider compatibility at lower expected cost.

Status: **defined; live price/availability evidence belongs to gpu-control and is not yet executed**.

### R8 — authenticated completion evidence boundary

Orbitune does not define the authentication mechanism for paid workload completion. The baked source SHA and normal/failure result metadata are correlation/debugging evidence only. They must not be treated as authenticated workload completion by themselves.

That authentication mechanism is a control-plane/provider contract and must be implemented and verified by `gpu-control` before live result collection is enabled.

Status: **intentionally blocked on gpu-control design/implementation**.

## Candidate-SHA freeze rule

An Orbitune commit may be frozen as the first RunPod candidate only when all of the following are true for that exact SHA:

1. the canary source, Dockerfile, entrypoint and workload contract are internally consistent;
2. `runpod-canary-smoke` is green on that SHA;
3. no known unresolved Orbitune-side blocker affects the infrastructure-canary purpose;
4. the exact SHA is then passed to the manual image-publication workflow;
5. the resulting image evidence reports the same SHA and an immutable `sha256:` digest.

Documentation-only changes after image publication do not mutate the frozen candidate. Any change to canary source, copied Orbitune model/tokenizer code, Dockerfile, workload contract, or smoke workflow requires a new source SHA and a new image publication.

## External prerequisites not owned by Orbitune

The following are prerequisites from `gpu-control` and cannot be satisfied merely by changing this repository:

- gpu-control leaves parked mode through reviewed change after an explicit human request;
- gpu-control `main` has required branch protection and status checks;
- protected owner-reviewed `paid-runpod` GitHub Environment exists;
- `RUNPOD_API_KEY` exists only as an environment-scoped secret in that environment;
- live RunPod adapter/workflow remains fail-closed until enabled by reviewed policy/code changes;
- immutable published-image evidence is consumed and bound to the approved plan;
- fresh catalog pricing/availability evidence exists;
- provider account occupancy is empty and rechecked immediately before create;
- authenticated workload-completion evidence exists;
- reliable cleanup/termination path exists and cleanup failure is visible;
- explicit paid authorization is provided for the particular execution.

## Paid-run sequence

```text
Orbitune exact SHA selected
→ exact-SHA CPU smoke green
→ freeze candidate SHA
→ exact-SHA image publication
→ capture digest evidence
→ gpu-control validate
→ gpu-control verify-source
→ container verification bound to source + digest
→ live catalog/economic decision evidence
→ explicit human paid authorization
→ ApprovedExecutionPlan
→ smallest useful paid GPU micro-canary
→ verify authenticated completion + cleanup
→ only then consider the 512k-token canary
```

Success of the micro-canary does not authorize the larger canary automatically. The larger execution requires a current rationale and new authorization under gpu-control policy.

## Current readiness summary

Orbitune is currently **workload-ready but not paid-execution-ready**.

Closed on this repository:

- deterministic finite canary implementation;
- bounded wrapper for normal and early-runner-failure evidence;
- machine-readable workload contract;
- source-SHA identity baked into published images/results;
- CPU success/failure contract smoke workflow;
- exact-commit image publication workflow;
- bounded output/result contract;
- fixed initial resource/training ceiling.

Still required before any paid RunPod action:

1. observe green CPU smoke on the exact chosen source SHA;
2. freeze that candidate SHA;
3. manually publish and verify its GHCR image and digest evidence;
4. complete gpu-control's external GitHub protection/environment/secret prerequisites;
5. implement/verify authenticated completion evidence and live provider wiring in gpu-control;
6. perform fresh pricing/availability and economic decision checks;
7. obtain explicit paid authorization for the specific run.
