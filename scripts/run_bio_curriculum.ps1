param(
    [string]$PythonExe = "",
    [string]$SignerSplitDir = "datasets\data\slovo_signer_split",
    [string]$OutRoot = "outputs\bio_out_v8",
    [string]$RunDir = "outputs\runs\bio_v8_curriculum",
    [string]$TrainConfig = "bio\configs\bio_default.json",
    [string]$ValConfig = "bio\configs\bio_val.json",
    [int]$SynthWorkers = 0,
    [int]$DenseSignerMinClips = 8,
    [switch]$Tensorboard,
    [switch]$SaveAnalysisArtifacts
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-PythonExe {
    param([string]$Requested)

    if ($Requested) { return $Requested }
    if (Test-Path ".\venv\Scripts\python.exe") { return ".\venv\Scripts\python.exe" }
    if (Test-Path ".\.venv\Scripts\python.exe") { return ".\.venv\Scripts\python.exe" }
    return "python"
}

function Invoke-Checked {
    param(
        [string]$Exe,
        [string[]]$CommandArgs,
        [string]$Title
    )

    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
    Write-Host "$Exe $($CommandArgs -join ' ')" -ForegroundColor DarkGray
    & $Exe @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "'$Title' failed with exit code $LASTEXITCODE"
    }
}

$PythonExe = Resolve-PythonExe -Requested $PythonExe

$splitArgs = @(
    "-m", "bio", "signer-split",
    "--csv", "datasets/data/annotations.csv",
    "--csv", "datasets/data/annotations_no_event.csv",
    "--out_dir", $SignerSplitDir
)
Invoke-Checked -Exe $PythonExe -CommandArgs $splitArgs -Title "BIO signer-split"

$buildArgs = @(
    "-m", "bio", "build-dataset",
    "--out_root", $OutRoot,
    "--slovo_csv", (Join-Path $SignerSplitDir "annotations.csv"),
    "--slovo_no_event_csv", (Join-Path $SignerSplitDir "annotations_no_event.csv"),
    "--synth_train_config", $TrainConfig,
    "--synth_val_config", $ValConfig,
    "--emit_warmup_dataset",
    "--dense_signer_min_clips", "$DenseSignerMinClips"
)
if ($SynthWorkers -gt 0) {
    $buildArgs += @("--synth_workers", "$SynthWorkers")
}
Invoke-Checked -Exe $PythonExe -CommandArgs $buildArgs -Title "BIO build-dataset curriculum"

$curriculumArgs = @(
    "-m", "bio", "train-curriculum",
    "--train_warmup_dir", (Join-Path $OutRoot "synth_train_warmup"),
    "--val_warmup_dir", (Join-Path $OutRoot "synth_val_warmup"),
    "--train_dir", (Join-Path $OutRoot "synth_train"),
    "--val_dir", (Join-Path $OutRoot "synth_val"),
    "--out_dir", $RunDir,
    "--config", $TrainConfig
)
if ($Tensorboard) {
    $curriculumArgs += "--tensorboard"
}
if ($SaveAnalysisArtifacts) {
    $curriculumArgs += "--save_analysis_artifacts"
}
Invoke-Checked -Exe $PythonExe -CommandArgs $curriculumArgs -Title "BIO train-curriculum"

Write-Host ""
Write-Host "BIO signer-aware curriculum finished." -ForegroundColor Green
Write-Host "Signer split: $SignerSplitDir" -ForegroundColor Green
Write-Host "Dataset root: $OutRoot" -ForegroundColor Green
Write-Host "Run dir: $RunDir" -ForegroundColor Green
