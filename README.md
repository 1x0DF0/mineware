# mineware

Vision agent for Minecraft: screen capture → HUD parse + YOLOv8 tree detection → walk up and chop.

Target: **Minecraft 26.2** (Java), Windows host.

## Layout

| File | Role |
|------|------|
| `main.py` | Window focus, **DPI-aware client capture**, keyboard/mouse (`pydirectinput`) |
| `hud.py` | Health / hunger / hotbar from pixels (no ML) |
| `collect_data.py` | Background frame dump → `dataset_raw/` |
| `setup_dataset.py` | Bootstrap YOLO labels (local auto-label or Roboflow) |
| `train_yolo.py` | Fine-tune YOLOv8n → `minecraft_yolo/run1/weights/best.pt` |
| `detect_live.py` | Live boxes on the game client area |
| `agent.py` | State machine: `SEARCHING` → `APPROACHING` → `CHOPPING` |
| `bootstrap.ps1` | One-shot dataset + train on Windows |

## Setup (Windows)

```powershell
cd path\to\mineware
py -m pip install -r requirements.txt
```

Minecraft must be **in-world** (cursor locked), not in a menu.

## Quick start

```powershell
# Optional: verify detections
py .\detect_live.py --conf 0.35

# Chop one tree
py .\agent.py --conf 0.35

# More trees
py .\agent.py --chops 3
```

If weights are missing:

```powershell
py .\setup_dataset.py --local-only
py .\train_yolo.py --epochs 50
```

Or with a free [Roboflow API key](https://app.roboflow.com/settings/api):

```powershell
$env:ROBOFLOW_API_KEY="your_key"
py .\setup_dataset.py --roboflow-only
py .\train_yolo.py --epochs 50
```

## Notes

- Capture uses the **client area only** (no title bar) and enables DPI awareness so boxes line up with the framebuffer.
- The checked-in tree model is a **bootstrap** (small auto-labeled set). For solid detections, hand-label more frames or pull a public Roboflow tree dataset and retrain.
- Run scripts with **Windows Python** (`py`), not WSL — `pygetwindow` / `pydirectinput` need Win32.

## License

Personal / research tooling. Minecraft is property of Mojang / Microsoft.
