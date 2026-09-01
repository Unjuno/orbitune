#!/usr/bin/env pwsh
# Context Fit Envelope (CFE) optimized launch command for the Compound Base
# on an NVIDIA RTX 3080 (16 GB) laptop GPU, BF16, PyTorch 2.5.1+cu124.
#
# Chosen config (vs default n_head=8):
#   - n_head=7, head_dim=32 (multi-of-8 -> cuDNN SDPA fast-path on Ampere)
#   - seq_len=256, microbatch=128  (sweet spot for 16 GB)
#   - causal-fastpath enabled (SDPA is_causal=True)
#   - precision=auto -> bf16
#
# See docs/CFE_REPORT_3080.md for the full report.
param(
    [string]$TrainJsonl = "data/continuous/synthetic_compound_train.jsonl",
    [string]$ValJsonl   = "data/continuous/synthetic_compound_val.jsonl",
    [string]$Config     = "configs/compound_hierarchical_9m_nhead7.json",
    [string]$Checkpoint = "runs/compound_9m_nhead7.pt",
    [int]$Steps         = 10000,
    [int]$BatchSize     = 128,
    [int]$SeqLen        = 256
)

$ErrorActionPreference = "Stop"
$env:PYTHONWARNINGS = "ignore"
$venv = ".\venv_cuda\Scripts\python.exe"

& $venv -W ignore scripts/compound_cfe_train.py train `
    --train-jsonl $TrainJsonl `
    --validation-jsonl $ValJsonl `
    --config $Config `
    --checkpoint $Checkpoint `
    --steps $Steps `
    --batch-size $BatchSize `
    --seq-len $SeqLen `
    --n-head 7 `
    --causal-fastpath `
    --precision auto `
    --learning-rate 3e-4 `
    --weight-decay 0.01 `
    --grad-clip 1.0 `
    --log-every 25 `
    --eval-every 250 `
    --validation-batches 4 `
    --validation-batch-size 4 `
    --checkpoint-every 250 `
    --seed 1