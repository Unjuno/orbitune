# RunPod training canary

This workload is the **first paid-GPU integration target** for Orbitune and `Unjuno/gpu-control`.

It is intentionally not a production Base training run and not a Compound-model quality experiment. Its purpose is to prove the remote execution path end to end:

```text
immutable Orbitune commit
→ exact-SHA image build with source SHA baked into the image
→ digest-pinned CUDA image
→ RunPod single GPU
→ CUDA-visible PyTorch
→ forward/backward
→ AdamW update
→ deterministic validation
→ checkpoint save
→ /outputs/result.json
→ authenticated bounded result/completion log collection
→ provider cleanup
```

The first canary does **not** transfer the checkpoint bytes off the Pod. It proves that the exact workload saved a checkpoint and authenticated the resulting checkpoint metadata in `result.json`. A later artifact-transfer contract is required before any checkpoint may be represented by `gpu-control` as a collected artifact.

## Why the legacy/reference 10.2M model is used

The production-candidate Compound Base is not implemented yet. Using it for the infrastructure canary would couple two independent unknowns: the new model and the remote GPU control plane.

The existing `orbitune-midi-gpt-v0` / `theory-remi-v0` 10.2M model is already executable and checkpointed, so it is the correct canary for isolating RunPod/container failures. A successful canary does **not** approve Theory-REMI as the production tokenizer.

## Fixed remote workload size

Default container execution uses:

```text
steps                 250
batch_size               8
sequence_length         256
tokens / optimizer step 2048
total training tokens 512000
validation interval      50 steps
validation points         5
model parameters   10200960
seed                20260824
```

The 512k-token amount is deliberate: it is large enough to exercise sustained CUDA kernels, optimizer state, validation and checkpoint I/O, but tiny relative to a real pretraining run. It should fit comfortably inside the `gpu-control` `cheap-24gb` profile (one GPU, >=24 GB VRAM, <=30 minutes, <=$0.30), subject to live provider price verification.

Do not increase the first paid run merely to consume available GPU time. If this canary fails, the useful signal is infrastructure/debugging, not additional training.

## Data

The workload generates a deterministic synthetic Theory-REMI token stream inside the container. It has:

- no dataset download;
- no runtime network dependency;
- no private data;
- no corpus-license ambiguity;
- no claim of musical-quality validity.

This keeps the first live run focused on the GPU/control-plane boundary.

## Image and source identity

The Dockerfile uses the PyTorch 2.10.0 CUDA 12.8/cuDNN 9 runtime image pinned by manifest digest:

```text
pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca
```

The exact Orbitune source SHA is supplied at image build time as `ORBITUNE_SOURCE_SHA`. The image stores that value in both an OCI revision label and the runtime environment. `result.json` records the baked `source_sha`.

A local direct Python smoke uses the explicit sentinel `unbaked-local`. A paid RunPod result must instead contain the exact 40-character SHA bound to the published image and ApprovedExecutionPlan.

The image entrypoint is finite and non-interactive. At runtime it needs no network access.

Image publication is manual through `.github/workflows/publish-runpod-canary.yml`. It checks out the requested exact SHA, builds only that source, passes the same SHA into the Docker build, and emits publication evidence containing the source SHA and immutable image digest.

## Completion signer / training process boundary

Authenticated completion uses privilege separation inside the container.

The image entrypoint starts as UID 0 because it is the only process allowed to receive and use the ephemeral `GPU_CONTROL_COMPLETION_KEY_B64`. It captures the completion context, removes every `GPU_CONTROL_COMPLETION_*` value from its live Python environment, and launches the actual training program as numeric UID/GID `10001:10001` with no supplementary groups and umask `0077`.

This separation is intentional. Merely omitting the secret from the child environment is insufficient when the wrapper and child run under the same UID, because a same-UID process may be able to inspect the parent process through `/proc`. The canary CI therefore includes a Linux-level regression test that the UID 10001 training process cannot read the root signer's `/proc/$PPID/environ`.

Authenticated mode also fixes the writable result directory to exactly `/outputs`; arbitrary output-directory overrides are rejected. Before training, the wrapper grants UID/GID 10001 temporary ownership of `/outputs`. Immediately after the training process exits, the wrapper reclaims the directory as root before reading or signing any result. `result.json` is opened with a no-follow regular-file check, so a symlink cannot be substituted for signed result content.

The wrapper snapshots the exact `result.json` bytes once, hashes those bytes into the completion envelope, and emits the same byte snapshot through the bounded stdout marker. This avoids a hash/read time-of-check-to-time-of-use mismatch. After signing and marker emission, `result.json`, `completion.json`, and the local checkpoint are root-owned read-only files and `/outputs` is sealed read-only for the remainder of the process lifetime.

The HMAC key is still considered ephemeral secret material. The workload never persists it, logs it, forwards it to the training process, or includes it in result/completion JSON.

## gpu-control source gate

`gpu-control` requires a public repository, exact 40-character commit SHA, exact Dockerfile path, bounded GPU profile/runtime/cost, and verifies the Dockerfile at that immutable SHA before later execution gates.

After choosing the exact Orbitune commit, the source-validation handoff is:

```bash
ORBITUNE_SHA=<40-character-commit-sha>

gpu-control verify-source \
  --target-repo Unjuno/orbitune \
  --target-sha "$ORBITUNE_SHA" \
  --dockerfile-path workloads/runpod-training-canary/Dockerfile \
  --gpu-profile cheap-24gb \
  --max-runtime-minutes 30 \
  --max-cost-usd 0.30
```

This is still a dry/source verification step; it does not authorize or create a paid Pod. Image publication, live pricing, cleanup guarantees, explicit paid-compute authorization, and the remaining RunPod adapter gates stay owned by `gpu-control`.

## Outputs and collection boundary

The workload writes these primary files to `/outputs`:

```text
/outputs/result.json
/outputs/canary-base.pt
```

When authenticated completion is enabled it also writes:

```text
/outputs/completion.json
```

`result.json` includes:

- workload/schema identity;
- baked Orbitune source SHA;
- architecture/tokenizer identity;
- CUDA availability and selected device;
- GPU name;
- PyTorch/CUDA versions;
- steps/batch/sequence/tokens processed;
- elapsed time and throughput;
- peak allocated VRAM;
- training and validation losses;
- checkpoint byte size and SHA-256;
- an explicit `transport: container-local-only` marker for the checkpoint metadata.

The bounded log protocol transports the exact bytes of `result.json` and, for an authenticated paid execution, `completion.json`. Each complete encoded marker, including its prefix, is limited to 16 KiB. The checkpoint itself is **not** included in this log transport.

Therefore `gpu-control` may use the authenticated result to verify that the trusted workload reports a checkpoint save, size and SHA-256, but it must not mark `canary-base.pt` as a collected artifact unless a separate byte-transfer path actually retrieves and verifies that file.

The 10.2M FP32 checkpoint is expected to remain below 64 MiB. That limit constrains the local canary output and a possible future bounded artifact-transfer path; it does not imply that the current log collector transferred the checkpoint.

## Acceptance gates

### Local CPU smoke

The code path may be tested without a GPU using a very small override:

```bash
python workloads/runpod-training-canary/run.py \
  --output-dir /tmp/orbitune-canary \
  --device cpu \
  --steps 1 \
  --batch-size 1 \
  --seq-len 16 \
  --validation-interval 1
```

This validates imports, model construction, one optimizer update, validation, checkpointing and result serialization. It does **not** validate CUDA. The direct local result records `source_sha=unbaked-local`. The same contract is run automatically by `.github/workflows/runpod-canary-smoke.yml`.

### First paid RunPod canary

Use the image's default arguments. The control-plane result is accepted only if all of the following are independently checked where the transport permits independent checking:

```text
container exit code == 0
result.status == pass
result.source_sha == exact approved Orbitune SHA
published image digest == ApprovedExecutionPlan image digest
result.device_type == cuda
result.cuda_available == true
result.parameters == 10200960
result.tokens_processed == 512000
validation_history has 5 finite points
last validation loss < first validation loss
authenticated result reports a non-empty checkpoint
reported checkpoint bytes <= 64 MiB
authenticated result contains checkpoint SHA-256 metadata
provider lifecycle finalized
cleanup confirmed
```

The control plane independently authenticates the exact collected `result.json` bytes and their SHA-256 through the completion envelope. It does **not** independently re-hash `canary-base.pt` off-Pod in the first canary because the checkpoint bytes are not transported. The checkpoint claim is an authenticated statement made by the exact immutable workload that created and hashed the local file.

The baked source SHA is correlation evidence, not workload-completion authentication. `gpu-control` must still supply the authenticated completion-evidence mechanism required by its own policy.

A CPU `pass` result is valid only as a local/container smoke; it is a **FAIL** for the paid GPU canary.

## Relation to future Compound training

After this infrastructure canary passes, do not immediately start a large official Base. The next GPU workloads should be:

1. Compound Base synthetic overfit, once the Compound model exists;
2. tiny real-MIDI overfit;
3. controlled 5M/10M/20M scale calibration;
4. only then corpus-scale pretraining.

This preserves the debugging hierarchy: infrastructure first, model correctness second, data/model-scale research third, expensive pretraining last.
