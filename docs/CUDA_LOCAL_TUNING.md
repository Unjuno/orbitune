# Local CUDA tuning for the Compound Base

Use this path before any long local training run. The goal is **training throughput with safe VRAM headroom**, not merely filling memory.

The CUDA tuner/trainer lives in `scripts/compound_cuda_train.py`. It keeps the model/checkpoint ABI unchanged, so checkpoints remain loadable by `orbitune-compound` for inspection and MIDI generation.

## 1. Verify the GPU and runtime

```bash
python scripts/compound_cuda_train.py hardware
```

Record the GPU name, compute capability, total VRAM, PyTorch/CUDA versions, and BF16 support. If the reported device or VRAM does not match the intended local GPU, stop before training.

## 2. Prepare real Compound data

```bash
python scripts/prepare_compound_split.py \
  --source data/raw \
  --train-out data/compound/train.jsonl \
  --validation-out data/compound/validation.jsonl \
  --report-out data/compound/split-report.json \
  --validation-fraction 0.1 \
  --min-events 32
```

## 3. Tune microbatch size on real training steps

Start at sequence length 256:

```bash
python scripts/compound_cuda_train.py tune \
  --train-jsonl data/compound/train.jsonl \
  --config configs/compound_hierarchical_9m.json \
  --seq-len 256 \
  --batch-sizes 4,8,16,32,64,96,128 \
  --precision auto \
  --max-vram-fraction 0.92 \
  --out runs/cuda-tune-256.json
```

The tuner executes real forward/backward/AdamW steps, so optimizer state is included in peak VRAM. It reports steps/s, events/s, peak allocated/reserved VRAM, peak reserved fraction, and GPU/memory utilization where the runtime exposes them.

Use the highest stable **events/s** result under the headroom limit. Do not select a configuration only because it fills more VRAM.

If the throughput winner uses less than about 60% of VRAM, run additional sweeps with longer sequence lengths, for example 384, 512, and 768. Prefer a longer useful sequence over blindly increasing batch size when it improves throughput and temporal training coverage.

Compare BF16 and FP16 explicitly if their throughput is close:

```bash
python scripts/compound_cuda_train.py tune ... --precision bf16
python scripts/compound_cuda_train.py tune ... --precision fp16
```

## 4. Train with the measured configuration

Example only; replace batch/sequence/precision with measured values:

```bash
ORBITUNE_SOURCE_COMMIT="$(git rev-parse HEAD)" \
python scripts/compound_cuda_train.py train \
  --train-jsonl data/compound/train.jsonl \
  --config configs/compound_hierarchical_9m.json \
  --checkpoint checkpoints/compound-base.pt \
  --precision auto \
  --batch-size 32 \
  --seq-len 256 \
  --steps 10000 \
  --log-every 25 \
  --checkpoint-every 250
```

The CUDA path uses Tensor-Core-friendly autocast, FP16 GradScaler when needed, fused AdamW when available, a tensorized song-window sampler, reduced per-step host synchronization, and CUDA throughput/memory telemetry.

A `low_vram_utilization` warning after warmup means the run should be investigated rather than left running for hours.

## 5. Resume

```bash
ORBITUNE_SOURCE_COMMIT="$(git rev-parse HEAD)" \
python scripts/compound_cuda_train.py train \
  --train-jsonl data/compound/train.jsonl \
  --checkpoint checkpoints/compound-base.pt \
  --resume checkpoints/compound-base.pt \
  --steps 20000 \
  --batch-size 32 \
  --seq-len 256 \
  --precision auto
```

CUDA checkpoints extend the existing checkpoint payload with AMP scaler state and runtime settings. Model/optimizer/RNG/sampler/global-step state continues to use the existing checkpoint ABI.

## 6. Optional `torch.compile`

Do not assume compilation is faster. Benchmark it after the ordinary AMP/fused-optimizer path is stable:

```bash
python scripts/compound_cuda_train.py train ... --compile --compile-mode default
```

Compare steady-state events/s after compilation warmup. Keep it only if it wins and checkpoint/resume still passes.

## 7. If utilization remains low

If VRAM is reasonably occupied but GPU utilization is persistently low, do not keep increasing batch size blindly. Profile the implementation. The first suspect in the current architecture is the sequential recurrent-memory scan (`GRUCell` per event), which can become launch/serialization bound on GPU.

At that point compare:

1. BF16 vs FP16;
2. batch and sequence-length sweeps;
3. fused AdamW behavior;
4. `torch.compile` on/off;
5. a profiler trace focused on the recurrent scan and attention kernels.

Any recurrent-memory kernel rewrite must preserve the numerical recurrence and be checked against the reference implementation before replacement.

## 8. Acceptance gate before a multi-hour run

Do not start a long Base training run until these are recorded:

- exact GPU identity and total VRAM;
- selected precision/batch/sequence length;
- peak VRAM and headroom;
- steady-state events/s after warmup;
- GPU utilization samples;
- finite loss;
- successful checkpoint save;
- successful resume from the saved global step;
- MIDI generation from the resumed checkpoint.

A 16 GiB GPU sitting near 3 GiB for hours is acceptable only if measured throughput has already plateaued and profiling shows a compute/serialization bottleneck that larger batch/sequence length does not improve. Otherwise treat it as an optimization failure.
