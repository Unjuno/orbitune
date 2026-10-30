#!/usr/bin/env pwsh
# Detached launcher for tools/idle_train.py.
#
# The supervisor keeps the Compound Base training whenever the GPU is free and
# yields it back to any other job that turns up. It reuses the production
# watchdog's checkpoint-validation, resume-parent and snapshot policy, so this
# launcher does not restate any of it.
#
# Stop it by creating the STOP sentinel in the run directory. The sentinel makes
# the supervisor ask the trainer to stand down at the next step boundary and
# write a final checkpoint; it does not kill a running step.

[CmdletBinding()]
param(
    [string]$Watchdog = "C:\orbitune-production\epoch_a2_watchdog.py",

    [int]$GpuIndex = 0,
    [double]$UtilThreshold = 5.0,

    # A slice peaks near 15.3 GiB reserved of 16 GiB on this geometry, so the GPU
    # has to be essentially empty before a slice starts.
    [double]$MinFreeMib = 14000,

    [double]$MaxCpuPercent = 0,
    [switch]$AllowForeignProcesses,
    [double]$IdleSeconds = 60,
    [double]$PollSeconds = 20,
    [double]$YieldAfterBusySeconds = 40,
    [double]$GracefulTimeout = 900,
    [double]$StartupTimeoutSeconds = 300,
    [double]$NoProgressBackoff = 120,
    [double]$HeartbeatSeconds = 600,
    [int]$MaxFailures = 5,
    [string]$Until,

    [switch]$DryRun,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"
$env:PYTHONWARNINGS = "ignore"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot ".")
$venv = Join-Path $repoRoot "orbitune_clone\venv_cuda\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venv)) {
    throw "CUDA venv not found at $venv"
}
if (-not (Test-Path -LiteralPath $Watchdog)) {
    throw "Watchdog module not found: $Watchdog"
}

# Pull the run directory out of the watchdog so the stop instructions below and
# the status command cannot drift from the run the supervisor will actually use.
$watchdogText = Get-Content -LiteralPath $Watchdog -Raw -ErrorAction Stop
$runMatch = [regex]::Match($watchdogText, '(?m)^RUN\s*=\s*Path\(r"([^"]+)"\)')
if (-not $runMatch.Success) {
    throw "Could not read RUN from $Watchdog"
}
$run = $runMatch.Groups[1].Value
New-Item -ItemType Directory -Path $run -Force | Out-Null
$statusLog = Join-Path $run "watchdog.jsonl"
$stdout = Join-Path $run "idle_supervisor.stdout.log"
$stderr = Join-Path $run "idle_supervisor.stderr.log"
$pidFile = Join-Path $run "idle_supervisor.pid"
$stopFile = Join-Path $run "STOP"

if (Test-Path -LiteralPath $stopFile) {
    throw "STOP sentinel already present at $stopFile. The previous run was stopped on purpose; remove it only if training should continue: Remove-Item -LiteralPath '$stopFile'"
}

$argsList = @(
    "tools\idle_train.py"
    "--watchdog", $Watchdog
    "--gpu-index", $GpuIndex
    "--util-threshold", $UtilThreshold
    "--min-free-mib", $MinFreeMib
    "--max-cpu-percent", $MaxCpuPercent
    "--idle-seconds", $IdleSeconds
    "--poll-seconds", $PollSeconds
    "--yield-after-busy-seconds", $YieldAfterBusySeconds
    "--graceful-timeout", $GracefulTimeout
    "--startup-timeout-seconds", $StartupTimeoutSeconds
    "--no-progress-backoff", $NoProgressBackoff
    "--heartbeat-seconds", $HeartbeatSeconds
    "--max-failures", $MaxFailures
)
if ($Until) { $argsList += @("--until", $Until) }
if ($AllowForeignProcesses) { $argsList += "--allow-foreign-processes" }
if ($DryRun) { $argsList += "--dry-run" }

Write-Host "[launcher] run:        $run"
Write-Host "[launcher] watchdog:   $Watchdog"
Write-Host "[launcher] start when: no other CUDA process, util <= $UtilThreshold%, free >= $MinFreeMib MiB, held $IdleSeconds s"
Write-Host "[launcher] yield when: another CUDA process stays resident for $YieldAfterBusySeconds s"
if ($AllowForeignProcesses) {
    Write-Warning "[launcher] foreign CUDA processes are being tolerated. This geometry reserves ~15.3 GiB of 16 GiB, so a co-resident job can be starved."
}
Write-Host "[launcher] status:     Get-Content '$statusLog' -Tail 5"
Write-Host "[launcher] stop:       New-Item -ItemType File -Path '$stopFile' -Force"

if ($DryRun) {
    & $venv -W ignore @argsList
    if ($LASTEXITCODE -ne 0) { throw "idle_train.py --dry-run exited with code $LASTEXITCODE" }
    return
}

if ($Foreground) {
    & $venv -W ignore @argsList
    if ($LASTEXITCODE -ne 0) { throw "idle_train.py exited with code $LASTEXITCODE" }
    return
}

$process = Start-Process -FilePath $venv -ArgumentList (@("-W", "ignore") + $argsList) `
    -WorkingDirectory $repoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii

Write-Host "[launcher] pid:        $($process.Id)"
Write-Host "[launcher] stdout:     $stdout"
Write-Host "[launcher] stderr:     $stderr"