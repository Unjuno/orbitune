# Local GPU training canary

This is the preferred first real-GPU execution path for Orbitune while paid-provider execution remains parked.

The current target host is `VLab16` with an NVIDIA RTX 3080. The launcher intentionally uses conservative defaults suitable for either common RTX 3080 VRAM variant and records the actual GPU name and peak allocated VRAM in `result.json`.

## Preconditions

- Linux host with NVIDIA driver and `nvidia-smi` working.
- Docker with NVIDIA Container Toolkit configured so `docker run --gpus all ...` works.
- A clean tracked Git working tree at the exact commit to test.
- No RunPod credential is required or used.

## Run

From the repository root:

```bash
bash workloads/local-gpu-canary/run-local.sh
```

Defaults:

- host label: `VLab16`
- expected GPU substring: `RTX 3080`
- steps: `250`
- batch size: `4`
- sequence length: `256`
- validation interval: `50`
- processed tokens: `256,000`

The launcher builds the pinned CUDA/PyTorch container from the exact local commit, runs it with CUDA required, networking disabled, a read-only root filesystem, dropped Linux capabilities, and only a writable output bind mount.

Outputs are written beneath:

```text
outputs/local-gpu-canary/VLab16/<source-sha>/
```

Expected artifacts:

- `result.json`
- `canary-base.pt`

The host-side acceptance check requires CUDA, a GPU name containing `RTX 3080`, positive peak VRAM use, a passing canary result, exact source SHA, finite losses, the expected token count, and a non-empty checkpoint.

## Small smoke run

To validate Docker/CUDA wiring before the full bounded canary:

```bash
STEPS=10 BATCH_SIZE=1 SEQ_LEN=128 VALIDATION_INTERVAL=10 \
  bash workloads/local-gpu-canary/run-local.sh
```

## Increasing throughput

Do not increase the default batch size until the first result records actual `peak_vram_bytes`. If there is substantial headroom, rerun with `BATCH_SIZE=8`. The infrastructure canary is not a musical-quality benchmark, so maximizing GPU utilization is not a success criterion.

## Scope

This path validates the local GPU/container/training/checkpoint stack only. It does not activate RunPod, authorize paid compute, publish an immutable registry image, or satisfy provider lifecycle/result-collection prerequisites.
