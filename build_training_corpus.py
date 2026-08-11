"""
Harvest all available Minecraft screenshots + labels into minecraft_dataset/
for YOLO training.

Sources (any that exist on disk):
  1) sessions/*/frames   — your play (auto-labeled trees via CV)
  2) dataset_raw/        — collect_data dumps (auto-labeled)
  3) root frame_*.png    — early captures
  4) external_datasets/minecraft_mobs_yolo — HF mobs set (real boxes, multi-class)
  5) existing minecraft_dataset labels if present

Output: minecraft_dataset/ with YOLO layout + data.yaml

Usage:
    py build_training_corpus.py
    py train_yolo.py --epochs 50 --name run2
"""

from __future__ import annotations

import random
import re
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from setup_dataset import auto_label_trees, crop_window_chrome

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "minecraft_dataset"
SESSIONS = ROOT / "sessions"
RAW = ROOT / "dataset_raw"
EXTERNAL_MOBS = ROOT / "external_datasets" / "minecraft_mobs_yolo" / "minecraft_mobs_yolo"

# Unified class map for multi-task perception
# 0 = tree (our agent needs this)
# 1+ = common mobs from HF set if present
TREE_CLASS = 0


def _write_yolo_label(path: Path, boxes: List[Tuple[int, float, float, float, float]]) -> None:
    """boxes: (cls, cx, cy, w, h) normalized."""
    lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for c, cx, cy, bw, bh in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def collect_raw_images() -> List[Path]:
    imgs: List[Path] = []
    for p in ROOT.glob("frame_*.png"):
        imgs.append(p)
    if (ROOT / "test_capture.png").exists():
        imgs.append(ROOT / "test_capture.png")
    if RAW.is_dir():
        imgs.extend(sorted(RAW.glob("*.png")))
        imgs.extend(sorted(RAW.glob("*.jpg")))
    if SESSIONS.is_dir():
        for sess in sorted(SESSIONS.iterdir()):
            frames = sess / "frames"
            if frames.is_dir():
                imgs.extend(sorted(frames.glob("*.png")))
                imgs.extend(sorted(frames.glob("*.jpg")))
    # dedupe by resolve
    seen = set()
    out = []
    for p in imgs:
        k = str(p.resolve())
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def auto_label_image(bgr: np.ndarray) -> List[Tuple[int, float, float, float, float]]:
    # crop chrome if full-window capture
    if bgr.shape[0] > 900:
        try:
            bgr2 = crop_window_chrome(bgr)
            if bgr2.size > 0 and bgr2.shape[0] > 100:
                bgr = bgr2
        except Exception:
            pass
    boxes = auto_label_trees(bgr)
    return [(TREE_CLASS, cx, cy, w, h) for cx, cy, w, h in boxes]


def ingest_autolabeled(images: List[Path], train_img: Path, train_lbl: Path, val_img: Path, val_lbl: Path, val_frac: float = 0.15):
    random.seed(42)
    random.shuffle(images)
    n_val = max(1, int(len(images) * val_frac)) if len(images) >= 8 else max(1, len(images) // 5)
    labeled = empty = 0
    for i, src in enumerate(images):
        bgr = cv2.imread(str(src))
        if bgr is None:
            continue
        # store cropped version for consistency with live client capture
        try:
            crop = crop_window_chrome(bgr) if bgr.shape[0] > 900 else bgr
        except Exception:
            crop = bgr
        boxes = auto_label_trees(crop)
        is_val = i < n_val
        img_dir, lbl_dir = (val_img, val_lbl) if is_val else (train_img, train_lbl)
        stem = f"auto_{src.parent.name}_{src.stem}_{i:05d}"
        # sanitize stem
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:120]
        dst_img = img_dir / f"{stem}.jpg"
        dst_lbl = lbl_dir / f"{stem}.txt"
        cv2.imwrite(str(dst_img), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        yolo_boxes = [(TREE_CLASS, *b) for b in boxes]
        _write_yolo_label(dst_lbl, yolo_boxes)
        if boxes:
            labeled += 1
        else:
            empty += 1
        if (i + 1) % 100 == 0:
            print(f"  auto-labeled {i+1}/{len(images)} (with_trees={labeled})")
    print(f"[auto] images={len(images)} with_trees={labeled} empty={empty}")
    return labeled


def ingest_external_mobs(train_img: Path, train_lbl: Path, val_img: Path, val_lbl: Path) -> int:
    """
    Copy HF mobs dataset. Remap classes to start at 1 (0 reserved for tree).
    """
    if not EXTERNAL_MOBS.is_dir():
        print(f"[mobs] not found: {EXTERNAL_MOBS} (download still running or skipped)")
        return 0

    yaml = EXTERNAL_MOBS / "data.yaml"
    # class offset: tree=0, mobs shift +1
    class_offset = 1
    n = 0
    for split_src, img_dst, lbl_dst in (
        ("train", train_img, train_lbl),
        ("val", val_img, val_lbl),
    ):
        si = EXTERNAL_MOBS / split_src / "images"
        sl = EXTERNAL_MOBS / split_src / "labels"
        if not si.is_dir():
            continue
        for img_path in si.glob("*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = sl / f"{img_path.stem}.txt"
            stem = f"mobs_{split_src}_{img_path.stem}"
            shutil.copy2(img_path, img_dst / f"{stem}{img_path.suffix.lower()}")
            if lbl_path.is_file():
                lines_out = []
                for line in lbl_path.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    c = int(float(parts[0])) + class_offset
                    lines_out.append(f"{c} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
                (lbl_dst / f"{stem}.txt").write_text(
                    "\n".join(lines_out) + ("\n" if lines_out else ""),
                    encoding="utf-8",
                )
            else:
                (lbl_dst / f"{stem}.txt").write_text("", encoding="utf-8")
            n += 1
    print(f"[mobs] ingested {n} images (classes shifted +1, tree stays 0)")
    return n


def write_data_yaml(n_mob_classes: int = 0) -> None:
    # If we only have trees, nc=1. If mobs present, read their yaml names.
    names = {0: "tree"}
    mob_yaml = EXTERNAL_MOBS / "data.yaml"
    if mob_yaml.is_file() and n_mob_classes >= 0:
        # parse names from yaml roughly
        text = mob_yaml.read_text(encoding="utf-8")
        # expect names: list or dict
        import re
        # try list form names: ['a','b']
        m = re.search(r"names:\s*\[([^\]]+)\]", text)
        if m:
            raw = [x.strip().strip("'\"") for x in m.group(1).split(",")]
            for i, name in enumerate(raw):
                names[i + 1] = name
        else:
            # dict form
            for line in text.splitlines():
                mm = re.match(r"\s*(\d+):\s*(.+)", line)
                if mm and "names" not in line:
                    pass
            # fallback: count label files max class
            max_c = 0
            for p in (EXTERNAL_MOBS).rglob("*.txt"):
                if "labels" not in str(p):
                    continue
                for line in p.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if parts:
                        max_c = max(max_c, int(float(parts[0])) + 1)
            for i in range(max_c):
                names.setdefault(i + 1, f"mob_{i}")

    # Agent only needs class 0 = tree; extra classes are fine for YOLO multi-task
    lines = ["path: .", "train: train/images", "val: valid/images", "names:"]
    for i in sorted(names):
        lines.append(f"  {i}: {names[i]}")
    lines.append(f"nc: {len(names)}")
    lines.append("")
    (OUT / "data.yaml").write_text("\n".join(lines), encoding="utf-8")
    print(f"[yaml] nc={len(names)} names={names}")


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    train_img = OUT / "train" / "images"
    train_lbl = OUT / "train" / "labels"
    val_img = OUT / "valid" / "images"
    val_lbl = OUT / "valid" / "labels"
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    images = collect_raw_images()
    print(f"[corpus] found {len(images)} local screenshots to auto-label")
    ingest_autolabeled(images, train_img, train_lbl, val_img, val_lbl)

    n_mobs = ingest_external_mobs(train_img, train_lbl, val_img, val_lbl)
    write_data_yaml(n_mobs)

    n_train = len(list(train_img.glob("*")))
    n_val = len(list(val_img.glob("*")))
    print(f"[corpus] ready: train={n_train} val={n_val} → {OUT}")
    print("[corpus] next: py train_yolo.py --epochs 50 --name run2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
