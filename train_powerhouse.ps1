# Train YOLO using FULL PC power on AMD systems
#   powershell -ExecutionPolicy Bypass -File .\train_powerhouse.ps1
#
# Hardware: Ryzen 9 5900X (24T), 128GB RAM, RX 9060 XT (AMD)
# CUDA = NVIDIA only (won't work here)
# torch-directml = no wheels for Python 3.14 yet
# => use ALL CPU cores + large batch + RAM cache (this box is great for that)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " MINEWARE POWERHOUSE TRAINER (AMD)"
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`n=== Hardware ===" -ForegroundColor Green
$cpu = Get-CimInstance Win32_Processor
$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 0)
$gpu = (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ", "
Write-Host "CPU : $($cpu.Name)  ($($cpu.NumberOfCores)C/$($cpu.NumberOfLogicalProcessors)T)"
Write-Host "RAM : $ram GB"
Write-Host "GPU : $gpu"

Write-Host "`n=== Python / torch ===" -ForegroundColor Green
py -c "import sys; print(sys.version)"
py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

$pyVer = py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Python minor: $pyVer"

# --- Try CUDA (won't work on AMD, but harmless) ---
$backend = "cpu"
$cuda = py -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($cuda -match "True") {
    $backend = "cuda"
    Write-Host "CUDA available - using GPU 0" -ForegroundColor Green
}

# --- Try DirectML only on Python < 3.13 (no 3.14 wheels) ---
if ($backend -ne "cuda" -and $gpu -match "AMD|Radeon") {
    if ($pyVer -match "^3\.(10|11|12)") {
        Write-Host "Trying torch-directml for AMD..." -ForegroundColor Yellow
        py -m pip install torch-directml 2>&1 | Out-Host
        $dml = py -c "import torch_directml, torch; d=torch_directml.device(); torch.zeros(1, device=d); print('DML_OK')" 2>$null
        if ($dml -match "DML_OK") {
            $backend = "dml"
            Write-Host "DirectML OK" -ForegroundColor Green
        } else {
            Write-Host "DirectML not working - using full CPU" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Python $pyVer : torch-directml has no wheels (need 3.10-3.12 for AMD GPU torch)." -ForegroundColor Yellow
        Write-Host "Using FULL CPU mode on Ryzen 9 5900X + 128GB RAM (still very fast for YOLOv8n)." -ForegroundColor Yellow
        Write-Host "Optional later: install Python 3.12 and re-run for DirectML GPU train." -ForegroundColor DarkYellow
    }
}

py -m pip install -U ultralytics opencv-python numpy 2>&1 | Select-Object -Last 5

if (-not (Test-Path ".\minecraft_dataset\data.yaml")) {
    Write-Host "Building dataset..." -ForegroundColor Yellow
    py .\build_training_corpus.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "`n=== Backend: $backend ===" -ForegroundColor Cyan

# Use almost all logical cores
$env:OMP_NUM_THREADS = "22"
$env:MKL_NUM_THREADS = "22"
$env:NUMEXPR_NUM_THREADS = "22"

$ErrorActionPreference = "Stop"

if ($backend -eq "cuda") {
    py .\train_yolo.py --epochs 40 --name run2 --device 0 --batch 0 --cache
} elseif ($backend -eq "dml") {
    py .\train_yolo.py --epochs 40 --name run2 --device dml --batch 16 --cache
} else {
    Write-Host "CPU POWER MODE: 22 threads, batch 48, workers 16, RAM cache" -ForegroundColor Yellow
    py .\train_yolo.py --epochs 40 --name run2 --device cpu --batch 48 --workers 16 --cache
}

$code = $LASTEXITCODE
if ($code -ne 0) {
    Write-Host "Training FAILED exit=$code" -ForegroundColor Red
    exit $code
}

Write-Host ""
Write-Host "SUCCESS: minecraft_yolo\run2\weights\best.pt" -ForegroundColor Green
Write-Host "py .\detect_live.py --conf 0.35"
Write-Host "py .\agent.py --conf 0.35 --chops 1"
