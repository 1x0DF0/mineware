# Fast YOLO train on Windows + NVIDIA GPU
# Run:
#   powershell -ExecutionPolicy Bypass -File .\train_gpu.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== Python / CUDA check ===" -ForegroundColor Cyan
py -c "import sys; print(sys.version)"
py -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"

$torchVer = py -c "import torch; print(torch.__version__)"
$cudaOk = py -c "import torch; print(torch.cuda.is_available())"

if ($torchVer -match "\+cpu" -or $cudaOk -notmatch "True") {
    Write-Host ""
    Write-Host "You have CPU-only torch ($torchVer). CUDA is NOT available." -ForegroundColor Yellow
    Write-Host "Python 3.14 often has NO official CUDA wheels yet." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "RECOMMENDED: install Python 3.12 (or 3.11), then:" -ForegroundColor Cyan
    Write-Host "  py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
    Write-Host "  py -3.12 -m pip install ultralytics opencv-python numpy"
    Write-Host "  py -3.12 .\train_yolo.py --epochs 40 --name run2 --device 0 --batch 0 --cache"
    Write-Host ""
    Write-Host "Trying CUDA install on current Python anyway..." -ForegroundColor Yellow
    py -m pip uninstall -y torch torchvision torchaudio
    py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    py -m pip install -U ultralytics

    $cudaOk2 = py -c "import torch; print(torch.cuda.is_available())"
    $torchVer2 = py -c "import torch; print(torch.__version__)"
    Write-Host "After reinstall: torch=$torchVer2 cuda=$cudaOk2"
    if ($cudaOk2 -notmatch "True") {
        Write-Host "FAILED: still no CUDA. Install Python 3.12 from python.org and use py -3.12" -ForegroundColor Red
        Write-Host "Or check GPU driver: nvidia-smi" -ForegroundColor Red
        exit 1
    }
}

Write-Host "=== nvidia-smi (driver) ===" -ForegroundColor Cyan
nvidia-smi
if ($LASTEXITCODE -ne 0) {
    Write-Host "nvidia-smi failed - install/update NVIDIA drivers first." -ForegroundColor Red
    exit 1
}

Write-Host "=== Train run2 on GPU ===" -ForegroundColor Cyan
py .\train_yolo.py --epochs 40 --name run2 --device 0 --batch 0 --cache
if ($LASTEXITCODE -ne 0) {
    Write-Host "Training failed (exit $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Done. Weights: minecraft_yolo\run2\weights\best.pt" -ForegroundColor Green
Write-Host "Test: py .\detect_live.py"
Write-Host "Agent: py .\agent.py --conf 0.35 --chops 1"
