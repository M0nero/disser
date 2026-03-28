param(
    [string]$PythonExe = "",
    [string]$OutRoot = "outputs\bio_out_v4",
    [string]$RunDir = "outputs\runs\bio_v4_run",
    [string]$TrainConfig = "bio\configs\bio_default.json",
    [string]$ValConfig = "bio\configs\bio_val.json",
    [int]$BatchSize = 256,
    [int]$NumWorkers = 1,
    [int]$SynthWorkers = 0,
    [int]$TrainSynthSamples = 100000,
    [int]$ValSynthSamples = 10000,
    [int]$ShardSize = 256,
    [switch]$FastDebug,
    [switch]$Step2Only
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-PythonExe {
    param([string]$Requested)

    if ($Requested) {
        return $Requested
    }
    if (Test-Path ".\venv\Scripts\python.exe") {
        return ".\venv\Scripts\python.exe"
    }
    if (Test-Path ".\.venv\Scripts\python.exe") {
        return ".\.venv\Scripts\python.exe"
    }
    return "python"
}

function Invoke-Checked {
    param(
        [string]$Exe,
        [string[]]$CommandArgs,
        [string]$Title
    )

    if (-not $CommandArgs -or $CommandArgs.Count -eq 0) {
        throw "'$Title' has no command arguments"
    }

    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
    Write-Host "$Exe $($CommandArgs -join ' ')" -ForegroundColor DarkGray
    & $Exe @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "'$Title' failed with exit code $LASTEXITCODE"
    }
}

$PythonExe = Resolve-PythonExe -Requested $PythonExe

if ($FastDebug) {
    $TrainSynthSamples = 20000
    $ValSynthSamples = 2000
}

if (-not $Step2Only) {
    $buildArgs = @(
        "-m", "bio", "build-dataset",
        "--out_root", $OutRoot,
        "--slovo_prelabel_config", $TrainConfig,
        "--synth_train_config", $TrainConfig,
        "--synth_val_config", $ValConfig,
        "--train_num_samples", "$TrainSynthSamples",
        "--val_num_samples", "$ValSynthSamples",
        "--shard_size", "$ShardSize"
    )
    if ($SynthWorkers -gt 0) {
        $buildArgs += @("--synth_workers", "$SynthWorkers")
    }
    Invoke-Checked -Exe $PythonExe -CommandArgs $buildArgs -Title "BIO build-dataset"
} else {
    $trainPrelabels = Join-Path $OutRoot "prelabels_slovo_train"
    $valPrelabels = Join-Path $OutRoot "prelabels_slovo_val"
    $trainNoev = Join-Path $OutRoot "prelabels_slovo_noev_train"
    $valNoev = Join-Path $OutRoot "prelabels_slovo_noev_val"
    $trainSynth = Join-Path $OutRoot "synth_train"
    $valSynth = Join-Path $OutRoot "synth_val"

    $synthTrainArgs = @(
        "-m", "bio", "synth-build",
        "--config", $TrainConfig,
        "--prelabel_dir", $trainPrelabels,
        "--preferred_noev_prelabel_dir", $trainNoev,
        "--out_dir", $trainSynth,
        "--num_samples", "$TrainSynthSamples",
        "--shard_size", "$ShardSize"
    )
    if ($SynthWorkers -gt 0) {
        $synthTrainArgs += @("--workers", "$SynthWorkers", "--no_auto_workers")
    } else {
        $synthTrainArgs += @("--auto_workers")
    }
    $synthValArgs = @(
        "-m", "bio", "synth-build",
        "--config", $ValConfig,
        "--prelabel_dir", $valPrelabels,
        "--preferred_noev_prelabel_dir", $valNoev,
        "--out_dir", $valSynth,
        "--num_samples", "$ValSynthSamples",
        "--shard_size", "$ShardSize"
    )
    if ($SynthWorkers -gt 0) {
        $synthValArgs += @("--workers", "$SynthWorkers", "--no_auto_workers")
    } else {
        $synthValArgs += @("--auto_workers")
    }

    Invoke-Checked -Exe $PythonExe -CommandArgs $synthTrainArgs -Title "BIO synth-build train"
    Invoke-Checked -Exe $PythonExe -CommandArgs $synthValArgs -Title "BIO synth-build val"
}

$trainDir = Join-Path $OutRoot "synth_train"
$valDir = Join-Path $OutRoot "synth_val"
$trainArgs = @(
    "-m", "bio", "train",
    "--train_dir", $trainDir,
    "--val_dir", $valDir,
    "--out_dir", $RunDir,
    "--config", $TrainConfig,
    "--batch_size", "$BatchSize",
    "--num_workers", "$NumWorkers",
    "--no_auto_workers",
    "--tensorboard",
    "--save_analysis_artifacts"
)

Invoke-Checked -Exe $PythonExe -CommandArgs $trainArgs -Title "BIO train"

Write-Host ""
Write-Host "BIO rebuild/train pipeline finished." -ForegroundColor Green
Write-Host "Dataset root: $OutRoot" -ForegroundColor Green
Write-Host "Run dir: $RunDir" -ForegroundColor Green
Write-Host "Train synth samples: $TrainSynthSamples" -ForegroundColor Green
Write-Host "Val synth samples: $ValSynthSamples" -ForegroundColor Green
if ($SynthWorkers -gt 0) {
    Write-Host "Synth workers: $SynthWorkers (manual)" -ForegroundColor Green
} else {
    Write-Host "Synth workers: auto" -ForegroundColor Green
}
