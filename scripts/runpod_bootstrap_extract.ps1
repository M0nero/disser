[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$IpAddress,

    [Parameter(Mandatory = $true)]
    [int]$Port,

    [Parameter(Mandatory = $true)]
    [int]$Jobs
)

$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/M0nero/disser.git"
$SshKeyPath = Join-Path $HOME ".ssh\id_ed25519"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LocalSubsetPath = (Resolve-Path (Join-Path $RepoRoot "test_out\runpod_phoenix_100")).Path

$RemoteRepoPath = "/workspace/disser"
$RemoteDataRoot = "/workspace/data"
$RemoteSubsetPath = "/workspace/data/phoenix_100"
$RemoteOutputRoot = "/workspace/out"
$RemoteSubsetManifest = "$RemoteSubsetPath/subset_manifest.txt"

function Invoke-Ssh {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command
    )

    & ssh "-p" $Port "-i" $SshKeyPath "root@$IpAddress" $Command
    if ($LASTEXITCODE -ne 0) {
        throw "SSH command failed: $Command"
    }
}

function Invoke-SshCapture {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command
    )

    $output = & ssh "-p" $Port "-i" $SshKeyPath "root@$IpAddress" $Command
    if ($LASTEXITCODE -ne 0) {
        throw "SSH command failed: $Command"
    }
    return ($output | Out-String).Trim()
}

function Invoke-ScpCopyDir {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LocalPath,

        [Parameter(Mandatory = $true)]
        [string]$RemotePath
    )

    & scp "-P" $Port "-i" $SshKeyPath "-r" $LocalPath "root@$IpAddress`:$RemotePath"
    if ($LASTEXITCODE -ne 0) {
        throw "SCP copy failed: $LocalPath -> $RemotePath"
    }
}

function Invoke-ScpCopyFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LocalPath,

        [Parameter(Mandatory = $true)]
        [string]$RemotePath
    )

    & scp "-P" $Port "-i" $SshKeyPath $LocalPath "root@$IpAddress`:$RemotePath"
    if ($LASTEXITCODE -ne 0) {
        throw "SCP copy failed: $LocalPath -> $RemotePath"
    }
}

if (-not (Test-Path $SshKeyPath)) {
    throw "SSH key not found: $SshKeyPath"
}

if (-not (Test-Path $LocalSubsetPath)) {
    throw "Local video subset not found: $LocalSubsetPath"
}

Write-Host "Connecting to pod ${IpAddress}:$Port" -ForegroundColor Cyan
Invoke-Ssh "mkdir -p $RemoteDataRoot $RemoteOutputRoot /workspace"
Invoke-Ssh "bash -lc 'if [ -d $RemoteDataRoot/runpod_phoenix_100 ] && [ ! -e $RemoteSubsetPath ]; then mv $RemoteDataRoot/runpod_phoenix_100 $RemoteSubsetPath; fi'"

$remoteBootstrap = @"
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y python3 python3-pip python3-venv git ffmpeg zstd rsync libglib2.0-0 libgl1 libsm6 libxext6 libxrender1

if [ -d "$RemoteRepoPath/.git" ]; then
  git -C "$RemoteRepoPath" fetch --all --prune
  git -C "$RemoteRepoPath" checkout main
  git -C "$RemoteRepoPath" pull --ff-only
else
  rm -rf "$RemoteRepoPath"
  git clone "$RepoUrl" "$RemoteRepoPath"
fi

cd "$RemoteRepoPath"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.runpod.txt
python3 -m pip install opencv-contrib-python
python3 -c "import cv2, mediapipe, pyarrow, zarr; print('runtime ok')"
"@

$bootstrapScript = [System.IO.Path]::GetTempFileName() + ".sh"
Set-Content -Path $bootstrapScript -Value $remoteBootstrap -NoNewline
Invoke-ScpCopyFile -LocalPath $bootstrapScript -RemotePath "/tmp/runpod_bootstrap_extract_bootstrap.sh"
Invoke-Ssh "bash /tmp/runpod_bootstrap_extract_bootstrap.sh"
Remove-Item -Path $bootstrapScript -Force
if ($LASTEXITCODE -ne 0) {
    throw "Remote bootstrap failed"
}

$hasSubset = Invoke-SshCapture "bash -lc 'test -f $RemoteSubsetManifest && echo yes || echo no'"
if ($hasSubset -ne "yes") {
    Write-Host "Copying video subset to pod" -ForegroundColor Cyan
    Invoke-ScpCopyDir -LocalPath $LocalSubsetPath -RemotePath $RemoteDataRoot
    Invoke-Ssh "bash -lc 'if [ -d $RemoteDataRoot/runpod_phoenix_100 ] && [ ! -e $RemoteSubsetPath ]; then mv $RemoteDataRoot/runpod_phoenix_100 $RemoteSubsetPath; fi'"
} else {
    Write-Host "Video subset already exists on pod, skipping copy" -ForegroundColor Yellow
}

$remoteRun = @"
set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  DELEGATE=gpu
else
  DELEGATE=cpu
fi

RUN_TS=\$(date +%Y%m%d_%H%M%S)
OUT_DIR="$RemoteOutputRoot/phoenix_100_\${DELEGATE}_\${RUN_TS}"

echo "delegate=\$DELEGATE"
echo "jobs=$Jobs"
echo "out_dir=\$OUT_DIR"

cd "$RemoteRepoPath"
python3 scripts/extract_keypoints.py \
  --in-dir "$RemoteSubsetPath" \
  --pattern "**/*.mp4" \
  --out-dir "\$OUT_DIR" \
  --seed 0 \
  --image-coords \
  --stride 1 \
  --pose-every 1 \
  --keep-pose-indices 0,9,10,11,12,13,14,15,16,23,24 \
  --min-det 0.40 \
  --min-track 0.35 \
  --second-pass \
  --min-hand-score 0.10 \
  --hand-score-lo 0.40 \
  --hand-score-hi 0.80 \
  --hand-score-source presence \
  --anchor-score 0.85 \
  --tracker-init-score 0.75 \
  --tracker-update-score 0.65 \
  --pose-dist-qual-min 0.55 \
  --pose-side-reassign-ratio 0.60 \
  --occ-hyst-frames 20 \
  --occ-return-k 1.30 \
  --track-max-gap 20 \
  --track-score-decay 0.93 \
  --track-reset-ms 300 \
  --sp-trigger-below 0.80 \
  --sp-roi-frac 0.30 \
  --sp-margin 0.45 \
  --sp-escalate-step 0.25 \
  --sp-escalate-max 2.0 \
  --sp-overlap-iou 0.12 \
  --sp-overlap-shrink 0.60 \
  --sp-overlap-penalty-mult 1.5 \
  --sp-center-penalty 0.20 \
  --sp-label-relax 0.25 \
  --sanity-scale-range "0.70,1.35" \
  --sanity-wrist-k 2.0 \
  --sanity-bone-tol 0.30 \
  --sanity-anchor-max-gap 30 \
  --interp-hold 7 \
  --postprocess \
  --pp-max-gap 20 \
  --pp-smoother ema \
  --mp-backend tasks \
  --mp-tasks-delegate "\$DELEGATE" \
  --execution-mode auto \
  --gpu-prefetch-frames 32 \
  --jobs $Jobs
"@

Write-Host "Starting extraction on pod" -ForegroundColor Cyan
$runScript = [System.IO.Path]::GetTempFileName() + ".sh"
Set-Content -Path $runScript -Value $remoteRun -NoNewline
Invoke-ScpCopyFile -LocalPath $runScript -RemotePath "/tmp/runpod_bootstrap_extract_run.sh"
Invoke-Ssh "bash /tmp/runpod_bootstrap_extract_run.sh"
Remove-Item -Path $runScript -Force
if ($LASTEXITCODE -ne 0) {
    throw "Remote extraction failed"
}
