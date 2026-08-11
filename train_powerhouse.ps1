# Train YOLO using FULL PC power (AMD DirectML and/or all Ryzen cores)
#   powershell -ExecutionPolicy Bypass -File .\train_powerhouse.ps1
#
# Your scan: Ryzen 9 5900X (24 threads), 128GB RAM, RX 9060 XT (AMD), Win11
# CUDA will NOT work (NVIDIA-only). We use DirectML for AMD, else max CPU.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " MINEWARE POWERHOUSE TRAINER"
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n=== Hardware ===" -ForegroundColor Green
$cpu = Get-CimInstance Win32_Processor
$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB, 0)
$gpu = Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name
Write-Host "CPU : $($cpu.Name)  ($($cpu.NumberOfCores)C/$($cpu.NumberOfLogicalProcessors)T)"
Write-Host "RAM : $ram GB"
Write-Host "GPU : $gpu"

$isAmd = ($gpu -match "AMD|Radeon")
$isNvidia = ($gpu -match "NVIDIA|GeForce|RTX|GTX")

Write-Host "`n=== Python / torch ===" -ForegroundColor Green
py -c "import sys; print(sys.version)"
py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>$null

# --- Backend pick ---
$backend = "cpu"
if ($isNvidia) {
    $cuda = py -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cuda -match "True") {
        $backend = "cuda"
    } else {
        Write-Host "NVIDIA GPU but torch has no CUDA. Install cu124 torch (use Python 3.12 if 3.14 fails)." -ForegroundColor Yellow
        Write-Host "  py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
    }
}

if ($backend -ne "cuda" -and $isAmd) {
    Write-Host "`nAMD GPU detected - trying torch-directml (DirectML)..." -ForegroundColor Yellow
    py -m pip install torch-directml -q 2>$null
    $dml = py -c "import torch_directml; d=torch_directml.device(); import torch; x=torch.zeros(1,device=d); print('DML_OK', d)" 2>$null
    if ($dml -match "DML_OK") {
        $backend = "dml"
        Write-Host "DirectML ready: $dml" -ForegroundColor Green
    } else {
        Write-Host "DirectML install/test failed - using FULL CPU instead (still strong on 5900X + 128GB)." -ForegroundColor Yellow
        $backend = "cpu"
    }
}

Write-Host "`n=== Backend: $backend ===" -ForegroundColor Cyan

# Ensure ultralytics
py -m pip install -U ultralytics opencv-python numpy -q

# Dataset check
if (-not (Test-Path ".\minecraft_dataset\data.yaml")) {
    Write-Host "Building dataset corpus..." -ForegroundColor Yellow
    py .\build_training_corpus.py
}

if ($backend -eq "cuda") {
    py .\train_yolo.py --epochs 40 --name run2 --device 0 --batch 0 --cache
} elseif ($backend -eq "dml") {
    # DirectML path inside train_yolo
    py .\train_yolo.py --epochs 40 --name run2 --device dml --batch 16 --cache
} else {
    # Full Ryzen + RAM: big batch, many workers, cache images
    Write-Host "CPU POWER MODE: 24 threads, large batch, RAM cache" -ForegroundColor Yellow
    $env:OMP_NUM_THREADS = "22"
    $env:MKL_NUM_THREADS = "22"
    py .\train_yolo.py --epochs 40 --name run2 --device cpu --batch 48 --workers 16 --cache
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Training FAILED exit=$LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`nSUCCESS: minecraft_yolo\run2\weights\best.pt" -ForegroundColor Green
Write-Host "py .\detect_live.py --conf 0.35"
Write-Host "py .\agent.py --conf 0.35 --chops 1"
