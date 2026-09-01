#!/usr/bin/env pwsh
# Production launcher for Compound Base long-run training on RTX 3080 16GB.
#
# This script is the recommended entry point for any multi-hour training
# run. It enforces:
#   - explicit --TrainJsonl and --ValJsonl (no silent synthetic fallback)
#   - synthetic JSONL rejection unless --AllowSynthetic is set
#   - strict runtime drift detection on resume (use --AllowRuntimeChange
#     to override; the drift is recorded in the run output)
#   - git HEAD SHA capture for audit trail
#
# Recommended settings for an RTX 3080 16GB (from the extended CFE sweep):
#   - BatchSize 144 (under 92% VRAM ceiling)
#   - SeqLen    256
#   - Precision auto -> BF16
#   - n_head=7 / head_dim=32 (cuDNN SDPA fast-path on Ampere)

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$TrainJsonl,

    [Parameter(Mandatory = $true)]
    [string]$ValJsonl,

    [string]$Config = "configs/compound_hierarchical_9m_nhead7.json",

    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [string]$Resume,

    [int]$Steps = 50000,

    [int]$BatchSize = 144,

    [int]$SeqLen = 256,

    [int]$NHead = 7,

    [switch]$CausalFastpath = $true,

    [ValidateSet("auto", "bf16", "fp16", "fp32")]
    [string]$Precision = "auto",

    [double]$LearningRate = 3e-4,

    [double]$WeightDecay = 0.01,

    [double]$GradClip = 1.0,

    [int]$CheckpointEvery = 250,

    [int]$LogEvery = 25,

    [int]$EvalEvery = 250,

    [int]$ValidationBatches = 4,

    [int]$ValidationBatchSize = 4,

    [int]$ValidationSeed = 10001,

    [int]$Seed = 1,

    [switch]$AllowSynthetic,

    [switch]$AllowRuntimeChange,

    [switch]$Compile,

    [ValidateSet("default", "reduce-overhead", "max-autotune")]
    [string]$CompileMode = "default"
)

$ErrorActionPreference = "Stop"
$env:PYTHONWARNINGS = "ignore"
$venv = ".\venv_cuda\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venv)) {
    throw "Python venv not found at $venv. Run tools/install_cuda_env.ps1 first."
}

# Capture git HEAD SHA for audit trail.
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot ".")
$gitHead = "unknown"
try {
    Push-Location $repoRoot
    $gitHead = (& git rev-parse HEAD 2>$null) | Select-Object -First 1
    Pop-Location
} catch {
    Pop-Location -ErrorAction SilentlyContinue
}
Write-Host "[launcher] git HEAD: $gitHead"
Write-Host "[launcher] train: $TrainJsonl"
Write-Host "[launcher] val:   $ValJsonl"
Write-Host "[launcher] ckpt:  $Checkpoint (resume: $Resume)"

# Enforce n_head consistency with the chosen config. The shipped
# compound_hierarchical_9m_nhead7.json already pins n_head=7. If the user
# overrides NHead they must use a matching config; we surface that here.
$configText = Get-Content -LiteralPath $Config -Raw -ErrorAction Stop
if ($configText -notmatch "`"n_head`":\s*$NHead\b") {
    Write-Warning "[launcher] n_head=$NHead does not match config $Config. Double-check the config file before continuing."
}

$argsList = @(
    "train"
    "--train-jsonl", $TrainJsonl
    "--validation-jsonl", $ValJsonl
    "--config", $Config
    "--checkpoint", $Checkpoint
    "--steps", $Steps
    "--batch-size", $BatchSize
    "--seq-len", $SeqLen
    "--n-head", $NHead
    "--learning-rate", $LearningRate
    "--weight-decay", $WeightDecay
    "--grad-clip", $GradClip
    "--checkpoint-every", $CheckpointEvery
    "--log-every", $LogEvery
    "--eval-every", $EvalEvery
    "--validation-batches", $ValidationBatches
    "--validation-batch-size", $ValidationBatchSize
    "--validation-seed", $ValidationSeed
    "--seed", $Seed
)
if ($CausalFastpath) { $argsList += "--causal-fastpath" } else { $argsList += "--no-causal-fastpath" }
$argsList += @("--precision", $Precision)
if ($Resume) { $argsList += @("--resume", $Resume) }
if ($AllowSynthetic) { $argsList += "--allow-synthetic" }
if ($AllowRuntimeChange) { $argsList += "--allow-runtime-change" }
if ($Compile) {
    $argsList += "--compile"
    $argsList += @("--compile-mode", $CompileMode)
}

& $venv -W ignore scripts/compound_cfe_train.py @argsList