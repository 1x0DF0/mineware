# Full Windows hardware report for mineware training
#   powershell -ExecutionPolicy Bypass -File .\scan_hw.ps1

$ErrorActionPreference = "Continue"
Write-Host "========== MINEWARE HARDWARE SCAN ==========" -ForegroundColor Cyan

Write-Host "`n=== CPU ===" -ForegroundColor Green
Get-CimInstance Win32_Processor | Format-List Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed, L2CacheSize, L3CacheSize

Write-Host "=== RAM ===" -ForegroundColor Green
$cs = Get-CimInstance Win32_ComputerSystem
"{0:N1} GB total  |  {1}  |  {2}" -f ($cs.TotalPhysicalMemory/1GB), $cs.Manufacturer, $cs.Model
Get-CimInstance Win32_PhysicalMemory | Select-Object @{N='GB';E={[math]::Round($_.Capacity/1GB,0)}}, Speed, Manufacturer, PartNumber | Format-Table -AutoSize

Write-Host "=== GPU ===" -ForegroundColor Green
Get-CimInstance Win32_VideoController | Format-List Name, AdapterRAM, DriverVersion, VideoProcessor, Status, DriverDate
Write-Host "nvidia-smi (NVIDIA only - may fail on AMD):"
nvidia-smi 2>&1 | Select-Object -First 15

Write-Host "`n=== DISK ===" -ForegroundColor Green
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" |
  Select-Object DeviceID, @{N='SizeGB';E={[math]::Round($_.Size/1GB,1)}}, @{N='FreeGB';E={[math]::Round($_.FreeSpace/1GB,1)}} |
  Format-Table -AutoSize

Write-Host "=== OS / PYTHON ===" -ForegroundColor Green
Get-CimInstance Win32_OperatingSystem | Format-List Caption, Version, OSArchitecture
py -0p 2>$null
py -c "import sys; print('default py:', sys.version)"
py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>$null
py -c "import torch_directml; print('directml OK', torch_directml.device())" 2>$null

Write-Host "`n=== TRAINING HINT ===" -ForegroundColor Yellow
Write-Host "AMD GPU -> use DirectML or full-CPU (not CUDA)."
Write-Host "Run: powershell -ExecutionPolicy Bypass -File .\train_powerhouse.ps1"
