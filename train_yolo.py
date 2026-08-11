"""
Fine-tune YOLOv8 on minecraft_dataset/ (expects data.yaml).

Prerequisites:
    py -m pip install ultralytics
    py .\\setup_dataset.py     # builds minecraft_dataset/

Usage:
    py .\\train_yolo.py
    py .\\train_yolo.py --epochs 50 --imgsz 640
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML_PATH = ROOT / "minecraft_dataset" / "data.yaml"
BASE_MODEL = "yolov8n.pt"  # nano — real-time friendly
EPOCHS = 40
IMAGE_SIZE = 640
PROJECT = "minecraft_yolo"
RUN_NAME = "run1"


def pick_device() -> str | int:
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[train] CUDA GPU: {torch.cuda.get_device_name(0)}")
            return 0
    except Exception:
        pass
    print("[train] No CUDA — training on CPU (slower)")
    return "cpu"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_YAML_PATH)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default=RUN_NAME)
    args = ap.parse_args(argv)

    if not args.data.is_file():
        print(
            f"Missing {args.data}\n"
            f"Run first:  py .\\setup_dataset.py"
        )
        return 1

    # Ultralytics resolves path: . relative to CWD — pin absolute dataset root
    ds_root = args.data.resolve().parent
    yaml_text = args.data.read_text(encoding="utf-8")
    lines = []
    for line in yaml_text.splitlines():
        if line.startswith("path:"):
            lines.append(f"path: {ds_root.as_posix()}")
        else:
            lines.append(line)
    args.data.write_text("\n".join(lines) + "\n", encoding="utf-8")

    device = pick_device()
    print(f"[train] data={args.data}")
    print(f"[train] base={args.model} epochs={args.epochs} imgsz={args.imgsz} device={device}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=25,
        project=str(ROOT / PROJECT),
        name=args.name,
        exist_ok=True,  # overwrite run1 cleanly
        workers=0 if device == "cpu" else 4,
        verbose=True,
    )

    best = ROOT / PROJECT / args.name / "weights" / "best.pt"
    last = ROOT / PROJECT / args.name / "weights" / "last.pt"
    print(f"\n[train] Done.")
    print(f"  best -> {best}  exists={best.is_file()}")
    print(f"  last -> {last}  exists={last.is_file()}")
    print("Next: py .\\detect_live.py   then   py .\\agent.py")
    return 0 if best.is_file() else 1


if __name__ == "__main__":
    sys.exit(main())
