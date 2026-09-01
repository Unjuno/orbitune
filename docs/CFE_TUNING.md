# Context Fit Envelope (CFE) tuning

`Context Fit Envelope` is an Orbitune project term, not a standard term from the literature.

It means the measured Pareto region of training configurations that fit the target GPU while maximizing useful training throughput. For the Compound Base, the CFE dimensions are:

- attention head geometry (`n_head`, therefore `head_dim`)
- attention kernel eligibility / causal fast-path
- precision (BF16 / FP16 / FP32)
- sequence length
- microbatch size
- optional `torch.compile` mode
- peak allocated/reserved VRAM and safe headroom

The objective is **not maximum VRAM occupancy**. The primary objective is stable events/sec with good GPU utilization and sufficient VRAM headroom.

## Why this was added

The checked-in Compound config uses `d_model=224, n_head=8`, so `head_dim=28`.

That geometry is valid mathematically, but it is suspicious for NVIDIA Ampere training. NVIDIA cuDNN SDPA documentation states that FP16/BF16 head dimensions should be multiples of 8, and PyTorch's fused SDPA/FlashAttention fast paths have historically had the same alignment requirement. `head_dim=28` therefore risks falling back to a slower attention backend.

Orbitune can change only the head partition while leaving all Q/K/V/output projection parameter counts unchanged:

| n_head | head_dim | d_model | parameter-count effect |
| ---: | ---: | ---: | --- |
| 8 | 28 | 224 | baseline |
| 7 | 32 | 224 | unchanged |
| 14 | 16 | 224 | unchanged |

Do **not** assume `n_head=7` is better musically. The CFE tuner measures whether the aligned geometry is materially faster on the actual GPU; model-quality comparisons remain a separate validation concern.

## Paper / implementation findings applied

### 1. FlashAttention: IO awareness matters

FlashAttention shows that reducing HBM reads/writes and avoiding materialization of the attention matrix can improve both speed and memory efficiency. FlashAttention-2 further improves GPU work partitioning and occupancy.

References:

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022: https://papers.nips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, 2023: https://tridao.me/publications/flash2/flash2.pdf

Application in Orbitune:

- `scripts/compound_cfe_train.py` can replace a materialized full causal bias with SDPA's `is_causal=True` path.
- Sliding-window local attention keeps its explicit local mask, because changing its semantics is not acceptable.
- The tuner compares fast-path ON/OFF on the real training step instead of assuming a backend is faster.

### 2. Ampere fused-attention geometry must be measured

NVIDIA cuDNN attention documentation requires FP16/BF16 head dimensions to be multiples of 8 for supported fused SDPA paths on Ampere/Ada. PyTorch SDPA automatically dispatches among Flash, memory-efficient/cuDNN, and math backends depending on input constraints.

References:

- NVIDIA cuDNN Attention: https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html
- PyTorch scaled dot product attention: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- PyTorch accelerated transformers / SDPA: https://pytorch.org/blog/accelerated-pytorch-2/

Application in Orbitune:

- `hardware` probes actual Flash / efficient / cuDNN backend eligibility for each candidate head geometry.
- default CFE candidates are `n_head=8,7,14` (`head_dim=28,32,16`).
- parameter count is unchanged by this head-count sweep.

### 3. Fixed shapes are useful for compiler/kernel reuse

Sequence-packing literature notes that changing sequence/batch shapes frequently can cause recompilation and kernel-selection overhead, while fixed shapes are friendly to compiled training. Orbitune already samples fixed-size song-local windows, so it does not pay padding cost inside each sampled window.

References:

- Krell et al., *Efficient Sequence Packing without Cross-contamination*, 2021/2022: https://arxiv.org/abs/2107.02027
- NVIDIA NeMo packed sequence guidance: https://docs.nvidia.com/nemo-framework/user-guide/25.09/sft_peft/packed_sequence.html

Application in Orbitune:

- CFE search explores a small set of fixed `(batch, seq_len)` shapes.
- after selecting a winner, long training should stay on the selected fixed shape unless a deliberate retune is performed.
- sequence packing is **not a current priority** because the current `TensorSampler` creates dense fixed-length windows with no padding tokens. Packing becomes relevant only if a future state-carry/song-stream trainer batches variable-length chunks.

### 4. Activation checkpointing is a Pareto tool, not a default

PyTorch documents activation checkpointing as a speed-memory tradeoff. Newer selective activation checkpointing and memory-budget mechanisms can find better points on that Pareto curve, but recomputation costs compute.

Reference:

- PyTorch, *Current and New Activation Checkpointing Techniques in PyTorch*, 2025: https://pytorch.org/blog/activation-checkpointing-techniques/

Application in Orbitune:

- do not enable activation checkpointing while the 9M model still has unused VRAM.
- first increase useful microbatch/context and use fused attention.
- only introduce selective checkpointing if the best useful `(batch, seq_len)` cannot fit and a larger context/batch produces enough throughput/learning benefit to compensate for recompute.

### 5. `torch.compile` is benchmark-gated

PyTorch `torch.compile` can fuse pointwise work and optionally use CUDA graphs / autotuned kernels. `reduce-overhead` may consume more memory because CUDA graph workspaces are cached.

Reference:

- PyTorch `torch.compile`: https://docs.pytorch.org/docs/stable/generated/torch.compile.html

Application in Orbitune:

- eager remains the baseline.
- compare `default`, `reduce-overhead`, and if worthwhile `max-autotune` only after compilation warmup.
- do not include compile time in steady-state events/sec.

## Commands

### Hardware / fused-backend probe

```bash
python scripts/compound_cfe_train.py hardware \
  --config configs/compound_hierarchical_9m.json \
  --precision auto \
  --head-counts 8,7,14
```

For a 224-wide model, the important observation is whether the target Ampere GPU reports a fused backend for head_dim 32/16 while head_dim 28 falls back.

### Context Fit Envelope sweep

Start with a controlled search:

```bash
python scripts/compound_cfe_train.py cfe \
  --train-jsonl data/compound/train.jsonl \
  --config configs/compound_hierarchical_9m.json \
  --precision auto \
  --head-counts 8,7,14 \
  --fastpaths false,true \
  --seq-lens 256,512 \
  --batch-sizes 4,8,16,32,64,96,128 \
  --warmup-steps 5 \
  --measure-steps 20 \
  --max-vram-fraction 0.92 \
  --out runs/cfe.json
```

The output records:

- `n_head` / `head_dim`
- causal fast-path ON/OFF
- sequence length
- batch size and events per microbatch
- steps/sec and events/sec
- parameter count
- fused AdamW availability
- peak allocated/reserved VRAM
- GPU utilization metrics when PyTorch exposes them
- OOM boundary

The recommended row is the safe candidate with maximum measured events/sec. Inspect the top frontier instead of trusting a single noisy sample; rerun the top 2-3 rows with a longer measurement interval.

## Training the selected CFE point

Example only; substitute values from `runs/cfe.json`:

```bash
ORBITUNE_SOURCE_COMMIT="$(git rev-parse HEAD)" \
python scripts/compound_cfe_train.py train \
  --train-jsonl data/compound/train.jsonl \
  --validation-jsonl data/compound/validation.jsonl \
  --config configs/compound_hierarchical_9m.json \
  --checkpoint models/compound-base.pt \
  --n-head 7 \
  --causal-fastpath \
  --precision bf16 \
  --batch-size 32 \
  --seq-len 512 \
  --steps 250 \
  --eval-every 50 \
  --checkpoint-every 50
```

Do not copy the example values without running the CFE sweep on the target GPU.

## Acceptance criteria before long training

A long run is acceptable only when all are known:

- actual GPU model and CUDA/PyTorch versions
- actual VRAM capacity
- fused attention backend eligibility for the selected head geometry
- selected precision
- selected `n_head` / `head_dim`
- selected fixed sequence length and microbatch
- eager vs compile result
- peak allocated and reserved VRAM
- stable events/sec after warmup
- representative GPU utilization / power / thermals from `nvidia-smi`
- validation loss remains finite
- checkpoint/resume succeeds
- generated MIDI reparses successfully

If VRAM remains low **and** GPU utilization/events/sec remain low, profile the recurrent `GRUCell` event scan and attention kernels before increasing training duration. If GPU utilization is already high and larger batches do not improve events/sec, low VRAM usage alone is not a failure.

## Not yet applied automatically

### FlexAttention for the 64-event sliding window

FlexAttention research shows large gains for sparse/sliding-window masks compared with materialized SDPA masks, and supports block-sparse causal sliding windows.

References:

- Dong et al., *Flex Attention: a Programming Model for Generating Optimized Attention Kernels*, MLSys 2025 / arXiv: https://arxiv.org/abs/2412.05496
- PyTorch FlexAttention overview: https://pytorch.org/blog/flexattention/

Orbitune does **not** automatically replace local SDPA with FlexAttention yet. The current model uses attention dropout during training, and changing attention implementation must preserve the training semantics and be benchmarked on the target PyTorch/Ampere stack. Treat FlexAttention as the next kernel-level A/B only if the profiler shows local sliding-window attention is a material bottleneck.
