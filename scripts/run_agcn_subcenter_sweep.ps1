param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$baseArgs = @(
    "-m", "msagcn.train",
    "--json", "datasets\skeletons",
    "--csv", "datasets\data\annotations.csv",
    "--logdir", "outputs\runs",
    "--max_frames", "64",
    "--temporal_crop", "resample",
    "--streams", "joints,bones,velocity",
    "--include_pose",
    "--pose_keep", "0,9,10,11,12,13,14,15,16,23,24",
    "--connect_cross_edges",
    "--center",
    "--center_mode", "masked_mean",
    "--normalize",
    "--norm_method", "p95",
    "--norm_scale", "1.0",
    "--augment",
    "--mirror_swap_only",
    "--mirror_prob", "0.5",
    "--rot_deg", "10",
    "--scale_jitter", "0.1",
    "--noise_sigma", "0.01",
    "--epochs", "60",
    "--batch", "64",
    "--lr", "5e-4",
    "--wd", "5e-4",
    "--grad_clip", "1.0",
    "--label_smoothing", "0.05",
    "--depths", "64,128,256,320",
    "--temp_ks", "9,7,5,5",
    "--drop", "0.05",
    "--droppath", "0.03",
    "--stream_drop_p", "0.05",
    "--use_logit_adjustment",
    "--use_cosine_head",
    "--cosine_margin", "0.2",
    "--cosine_scale", "30",
    "--ema_decay", "0.999",
    "--warmup_frac", "0.1",
    "--tensorboard",
    "--tb_full_logging",
    "--auto_workers",
    "--use_ctr_hand_refine",
    "--ctr_groups", "4",
    "--ctr_alpha_init", "0.0",
    "--ctr_in_stream_encoder",
    "--use_decoded_skeleton_cache",
    "--use_supcon",
    "--supcon_weight", "0.03",
    "--supcon_temp", "0.07",
    "--supcon_mixed_batch",
    "--supcon_mixed_repeated_classes", "16",
    "--supcon_mixed_repeated_samples", "2"
)

$runs = @(
    @{
        Name = "agcn_supcon_mixed_w003_t007_16x2"
        ExtraArgs = @(
            "--cosine_subcenters", "1"
        )
    },
    @{
        Name = "agcn_supcon_mixed_w003_t007_16x2_sc2"
        ExtraArgs = @(
            "--cosine_subcenters", "2"
        )
    },
    @{
        Name = "agcn_supcon_mixed_w003_t007_16x2_sc3"
        ExtraArgs = @(
            "--cosine_subcenters", "3"
        )
    }
)

foreach ($run in $runs) {
    $runName = [string]$run.Name
    $outDir = "outputs\runs\$runName"
    $cmdArgs = @(
        $baseArgs +
        @("--out", $outDir, "--run_name", $runName) +
        [string[]]$run.ExtraArgs
    )

    Write-Host ""
    Write-Host "=== Starting $runName ===" -ForegroundColor Cyan
    Write-Host "$PythonExe $($cmdArgs -join ' ')" -ForegroundColor DarkGray

    & $PythonExe @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Run '$runName' failed with exit code $LASTEXITCODE"
    }

    Write-Host "=== Finished $runName ===" -ForegroundColor Green
}

Write-Host ""
Write-Host "All sub-center cosine sweep runs finished." -ForegroundColor Green
