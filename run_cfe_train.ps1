#!/usr/bin/env pwsh
# Production launcher for Compound Base training on the measured RTX 3080 CFE.
#
# The CFE geometry remains n_head=7 / head_dim=32 / seq=256 / batch=144.
# This launcher deliberately uses scripts/compound_longrun_train.py rather
# than the benchmark-oriented CFE trainer so exact resume and health gates
# run before optimizer mutation.

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

    [Parameter(Mandatory = $true)]
    [int]$Steps,

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

    # When -Resume is set, overwrite the optimizer's param_group 'lr' to
    # this value after the checkpoint's optimizer state is loaded. Without
    # this flag, -LearningRate is silently ignored on resume (the saved
    # optimizer state restores the original LR). Recorded in the new
    # checkpoint's runtime dict for audit. Requires -Resume.
    [double]$OverrideResumeLr,

    # Required until state-carry TBPTT is implemented. This prevents a
    # multi-hour run from silently implying train/generation state equivalence.
    [switch]$AllowFixedWindowTraining
)

$ErrorActionPreference = "Stop"
$env:PYTHONWARNINGS = "ignore"
$venv = ".\venv_cuda\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venv)) {
    throw "Python venv not found at $venv. Install the CUDA environment first."
}
if (-not (Test-Path -LiteralPath $TrainJsonl)) {
    throw "Training JSONL not found: $TrainJsonl"
}
if (-not (Test-Path -LiteralPath $ValJsonl)) {
    throw "Validation JSONL not found: $ValJsonl"
}
if (-not $AllowFixedWindowTraining) {
    throw "Fixed-window training resets pre-window local/medium/global/recurrent history. Read docs/STATE_CARRY_AUDIT.md and pass -AllowFixedWindowTraining explicitly for the current training mode."
}

$configObject = Get-Content -LiteralPath $Config -Raw -ErrorAction Stop | ConvertFrom-Json
if ([int]$configObject.n_head -ne $NHead) {
    throw "n_head=$NHead does not match config n_head=$($configObject.n_head) in $Config"
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot ".")
$gitHead = "unknown"
try {
    Push-Location $repoRoot
    $gitHead = (& git rev-parse HEAD 2>$null) | Select-Object -First 1
} finally {
    Pop-Location -ErrorAction SilentlyContinue
}
$env:ORBITUNE_SOURCE_COMMIT = $gitHead

Write-Host "[launcher] git HEAD: $gitHead"
Write-Host "[launcher] train:    $TrainJsonl"
Write-Host "[launcher] val:      $ValJsonl"
Write-Host "[launcher] ckpt:     $Checkpoint"
Write-Host "[launcher] resume:   $Resume"
Write-Host "[launcher] geometry: n_head=$NHead batch=$BatchSize seq=$SeqLen precision=$Precision"
Write-Warning "[launcher] fixed-window mode is explicitly enabled; state-carry TBPTT remains a documented follow-up."

$argsList = @(
    "--train-jsonl", $TrainJsonl
    "--validation-jsonl", $ValJsonl
    "--config", $Config
    "--checkpoint", $Checkpoint
    "--steps", $Steps
    "--batch-size", $BatchSize
    "--seq-len", $SeqLen
    "--n-head", $NHead
    "--precision", $Precision
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
    "--allow-fixed-window-training"
)
if ($CausalFastpath) { $argsList += "--causal-fastpath" } else { $argsList += "--no-causal-fastpath" }
if ($Resume) { $argsList += @("--resume", $Resume) }
if ($AllowSynthetic) { $argsList += "--allow-synthetic" }
if ($AllowRuntimeChange) { $argsList += "--allow-runtime-change" }
if ($PSBoundParameters.ContainsKey('OverrideResumeLr')) {
    if (-not $Resume) {
        throw "-OverrideResumeLr requires -Resume; refusing to silently retune a fresh run."
    }
    $argsList += @("--override-resume-lr", $OverrideResumeLr)
}

& $venv -W ignore scripts/compound_longrun_train.py @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Compound long-run trainer exited with code $LASTEXITCODE"
}
