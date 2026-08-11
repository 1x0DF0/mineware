# One-shot: dataset -> train -> print next steps
# Run from PowerShell in this folder:
#   .\bootstrap.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== 1/3  pip deps ===" -ForegroundColor Cyan
py -m pip install ultralytics opencv-python numpy --quiet

Write-Host "=== 2/3  build dataset (local auto-label; use ROBOFLOW_API_KEY for real Roboflow set) ===" -ForegroundColor Cyan
if ($env:ROBOFLOW_API_KEY) {
    py -m pip install roboflow --quiet
    py .\setup_dataset.py
} else {
    py .\setup_dataset.py --local-only
}

Write-Host "=== 3/3  train YOLOv8n ===" -ForegroundColor Cyan
py .\train_yolo.py --epochs 40 --batch 8

$best = Join-Path $PSScriptRoot "minecraft_yolo\run1\weights\best.pt"
if (Test-Path $best) {
    Write-Host ""
    Write-Host "SUCCESS: $best" -ForegroundColor Green
    Write-Host "Next:"
    Write-Host "  py .\detect_live.py"
    Write-Host "  py .\agent.py"
} else {
    Write-Host "Training finished but best.pt not found — check console errors." -ForegroundColor Red
    exit 1
}
