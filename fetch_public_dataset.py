"""
Download a public Minecraft tree detection dataset (YOLOv8 format) into
minecraft_dataset/ for retraining.

Known sources (need free Roboflow API key from
https://app.roboflow.com/settings/api ):

  minecraft-thing / minecraft-tree-detection
  ananthv / minecraft-tree-wood-identification-dataset

Usage:
    set ROBOFLOW_API_KEY=your_key
    py fetch_public_dataset.py
    py fetch_public_dataset.py --project minecraft-tree-wood-identification-dataset --workspace ananthv
    py train_yolo.py --epochs 50
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "minecraft_dataset"

# (workspace, project, version)
DEFAULTS = ("minecraft-thing", "minecraft-tree-detection", 1)
ALTERNATES = [
    ("minecraft-thing", "minecraft-tree-detection", 1),
    ("ananthv", "minecraft-tree-wood-identification-dataset", 1),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=DEFAULTS[0])
    ap.add_argument("--project", default=DEFAULTS[1])
    ap.add_argument("--version", type=int, default=DEFAULTS[2])
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    args = ap.parse_args()

    key = (args.api_key or "").strip()
    if not key:
        print("Need a free Roboflow API key:")
        print("  1) https://app.roboflow.com/settings/api")
        print("  2) set ROBOFLOW_API_KEY=...")
        print("  3) py fetch_public_dataset.py")
        print("\nAlternate Universe projects to try:")
        for ws, proj, ver in ALTERNATES:
            print(f"  --workspace {ws} --project {proj} --version {ver}")
            print(f"    https://universe.roboflow.com/{ws}/{proj}")
        return 1

    try:
        from roboflow import Roboflow
    except ImportError:
        print("pip install roboflow")
        return 1

    print(f"[fetch] {args.workspace}/{args.project} v{args.version}")
    rf = Roboflow(api_key=key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    dl_root = ROOT / "_rf_dl"
    if dl_root.exists():
        shutil.rmtree(dl_root)
    ds = version.download("yolov8", location=str(dl_root))
    src = Path(ds.location)
    yaml = list(src.rglob("data.yaml"))
    if not yaml:
        print(f"No data.yaml under {src}")
        return 1
    ds_root = yaml[0].parent
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(ds_root, OUT)

    # Normalize class name / paths for ultralytics
    data_yaml = OUT / "data.yaml"
    # Prefer train/valid layout if present
    if (OUT / "train" / "images").exists():
        train_p, val_p = "train/images", (
            "valid/images" if (OUT / "valid" / "images").exists() else "train/images"
        )
    elif (OUT / "images" / "train").exists():
        train_p, val_p = "images/train", "images/val"
    else:
        train_p, val_p = "train/images", "valid/images"

    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                f"train: {train_p}",
                f"val: {val_p}",
                "names:",
                "  0: tree",
                "nc: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[fetch] wrote {OUT} (data.yaml forced class 'tree')")
    print("[fetch] next:  py train_yolo.py --epochs 50")
    return 0


if __name__ == "__main__":
    sys.exit(main())
