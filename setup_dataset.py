"""
Bootstrap a YOLOv8 Minecraft-tree dataset into ./minecraft_dataset/

Priority:
  1) If ROBOFLOW_API_KEY is set — download the public Roboflow tree set
     (workspace minecraft-thing / project minecraft-tree-detection).
  2) Else — build a local bootstrap set by auto-labeling tree-like regions
     (brown trunk + green canopy) on screenshots in dataset_raw/ and any
     root-level frame_*.png / test_capture.png.

Usage:
    py setup_dataset.py
    set ROBOFLOW_API_KEY=xxxx
    py setup_dataset.py --roboflow-only

After this, run:
    py train_yolo.py
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "minecraft_dataset"
RAW_DIRS = [ROOT / "dataset_raw", ROOT / "images", ROOT]
IMAGE_GLOBS = ("*.png", "*.jpg", "*.jpeg")

# class 0 = tree  (agent.py TREE_CLASS_HINTS includes "tree")
CLASS_NAMES = ["tree"]


# ---------- Roboflow download ----------

def download_roboflow(api_key: str, out_dir: Path) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError as e:
        raise SystemExit(
            "roboflow package missing. Install with: py -m pip install roboflow"
        ) from e

    print(f"[roboflow] Logging in …")
    rf = Roboflow(api_key=api_key)
    # Public Universe project (CC BY 4.0)
    project = rf.workspace("minecraft-thing").project("minecraft-tree-detection")
    version = project.version(1)
    print("[roboflow] Downloading YOLOv8 export …")
    ds = version.download("yolov8", location=str(out_dir.parent / "_rf_dl"))
    src = Path(ds.location)
    # Normalize into minecraft_dataset/
    if out_dir.exists():
        shutil.rmtree(out_dir)
    # Roboflow layout varies: sometimes already has data.yaml + train/valid
    yaml_candidates = list(src.rglob("data.yaml"))
    if not yaml_candidates:
        raise RuntimeError(f"No data.yaml in Roboflow download at {src}")
    ds_root = yaml_candidates[0].parent
    shutil.copytree(ds_root, out_dir)
    # Force class name to 'tree' if export used something else
    _normalize_yaml(out_dir / "data.yaml")
    print(f"[roboflow] Dataset ready at {out_dir}")
    return out_dir


def _normalize_yaml(yaml_path: Path) -> None:
    """Rewrite data.yaml so train/val paths are relative and names=['tree']."""
    text = yaml_path.read_text(encoding="utf-8")
    # Keep structure simple — rewrite fully
    # Discover train/val folders
    root = yaml_path.parent
    train = None
    val = None
    for cand in ("train/images", "images/train", "train"):
        if (root / cand).exists():
            train = cand if cand.endswith("images") or (root / cand / "images").exists() is False else cand
            break
    # Prefer standard ultralytics layout
    if (root / "train" / "images").exists():
        train_path = "train/images"
        val_path = "valid/images" if (root / "valid" / "images").exists() else (
            "val/images" if (root / "val" / "images").exists() else "train/images"
        )
    elif (root / "images" / "train").exists():
        train_path = "images/train"
        val_path = "images/val" if (root / "images" / "val").exists() else "images/train"
    else:
        # leave file, just fix names if possible
        train_path = "train/images"
        val_path = "valid/images"

    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                f"train: {train_path}",
                f"val: {val_path}",
                "names:",
                "  0: tree",
                f"nc: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )


# ---------- Classical auto-label (Minecraft trees) ----------
# NOTE: Parallel leaf/wood heuristics also live in trees.detect_trees_cv
# (live perception fallback). If you retune HSV/NMS here, consider updating
# trees.py the same way (and vice versa). Not shared yet on purpose.

def _wood_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Log browns (not dry dirt: keep S higher) + pale birch bark
    brown = cv2.inRange(hsv, (6, 55, 35), (22, 255, 170))
    birch = cv2.inRange(hsv, (0, 0, 140), (35, 45, 230))
    return brown | birch


def _leaf_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Canopy leaves: green, mid/dark value (bright grass is high-V + often yellower)
    # Oak leaves tend toward H~40-80, S moderate, V not sky-bright
    return cv2.inRange(hsv, (32, 50, 20), (90, 255, 140))


def auto_label_trees(bgr: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """
    Return YOLO-normalized boxes (cx, cy, w, h) for likely trees.

    Strict: canopy blob + vertical trunk evidence underneath. Rejects
    dirt hills / full-frame grass that plagued the first bootstrap pass.
    """
    h, w = bgr.shape[:2]
    roi = bgr.copy()
    y0, y1 = int(h * 0.10), int(h * 0.86)
    roi[:y0, :] = 0
    roi[y1:, :] = 0

    leaves = _leaf_mask(roi)
    wood = _wood_mask(roi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaves = cv2.morphologyEx(leaves, cv2.MORPH_OPEN, k, iterations=1)
    # Heavy erode to split merged canopies, then re-dilate per-component
    k_split = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    seeds = cv2.erode(leaves, k_split, iterations=2)
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, k, iterations=1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    boxes: List[Tuple[float, float, float, float]] = []
    min_area = (h * w) * 0.0015  # seeds are smaller after erode
    max_area = (h * w) * 0.20

    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area or area > max_area:
            continue
        if bw < 12 or bh < 12:
            continue
        aspect = bw / max(bh, 1)
        if aspect > 2.8:
            continue

        # Grow seed back toward full canopy using original leaf mask bbox
        pad = max(bw, bh)
        gx1 = max(0, x - pad)
        gy1 = max(y0, y - pad)
        gx2 = min(w, x + bw + pad)
        gy2 = min(y1, y + bh + pad)
        region = leaves[gy1:gy2, gx1:gx2]
        if region.size == 0 or np.count_nonzero(region) < 50:
            continue
        ys, xs = np.where(region > 0)
        x = gx1 + int(xs.min())
        y = gy1 + int(ys.min())
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        area = bw * bh
        if area > (h * w) * 0.30:
            continue

        cx_px = x + bw // 2
        stem_half = max(10, bw // 4)
        sx1 = max(0, cx_px - stem_half)
        sx2 = min(w, cx_px + stem_half)
        sy1 = min(h - 1, y + int(bh * 0.55))
        sy2 = min(y1, y + bh + int(bh * 0.85) + 25)
        if sy2 <= sy1 + 4:
            continue
        stem = wood[sy1:sy2, sx1:sx2]
        wood_frac = float(np.count_nonzero(stem)) / max(stem.size, 1)

        canopy = leaves[y : y + bh, x : x + bw]
        leaf_frac = float(np.count_nonzero(canopy)) / max(canopy.size, 1)
        if leaf_frac < 0.15:
            continue

        large_close = area > (h * w) * 0.06 and (y + bh) > h * 0.42
        if wood_frac < 0.03 and not large_close:
            continue

        # Tighter box: prefer canopy + short trunk, not a huge blob to the side
        expand_down = int(bh * 0.45) if wood_frac >= 0.03 else int(bh * 0.12)
        expand_x = int(bw * 0.05)
        bx1 = max(0, x - expand_x)
        by1 = max(y0, y - int(bh * 0.03))
        bx2 = min(w - 1, x + bw + expand_x)
        by2 = min(y1, y + bh + expand_down)

        # If we have a clear stem, re-center box horizontally on the stem
        if wood_frac >= 0.05:
            stem_cols = np.count_nonzero(stem, axis=0)
            if stem_cols.size and stem_cols.max() > 0:
                stem_cx = sx1 + int(np.argmax(stem_cols))
                half_w = max(bw // 2, int(0.08 * w))
                bx1 = max(0, stem_cx - half_w)
                bx2 = min(w - 1, stem_cx + half_w)

        nw = (bx2 - bx1) / w
        nh = (by2 - by1) / h
        if nw < 0.04 or nh < 0.06:
            continue
        if nw > 0.48 or nh > 0.62:
            continue

        cx = ((bx1 + bx2) / 2.0) / w
        cy = ((by1 + by2) / 2.0) / h
        boxes.append((cx, cy, nw, nh))

    boxes = _nms_boxes(boxes, iou_thresh=0.4)
    return boxes[:8]


def _nms_boxes(
    boxes: List[Tuple[float, float, float, float]], iou_thresh: float
) -> List[Tuple[float, float, float, float]]:
    if not boxes:
        return []
    # sort by area desc
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    keep: List[Tuple[float, float, float, float]] = []

    def iou(a, b):
        ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
        ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
        bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
        bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = a[2] * a[3] + b[2] * b[3] - inter
        return inter / ua if ua > 0 else 0.0

    for b in boxes:
        if all(iou(b, k) < iou_thresh for k in keep):
            keep.append(b)
    return keep


def crop_window_chrome(bgr: np.ndarray) -> np.ndarray:
    """
    Strip title-bar / thin borders from full-window captures so labels match
    client-area live capture (main.get_region(client_only=True)).
    """
    h, w = bgr.shape[:2]
    # Title bar on these captures is ~32–36px of near-uniform light gray
    top = 0
    for y in range(min(80, h)):
        row = bgr[y]
        # game content has higher spatial variance than the title strip
        if row.std() > 35 and row.mean() < 200:
            top = max(0, y - 1)
            break
    if top < 20:
        top = 32  # fallback for this Minecraft window style
    # thin bottom/side borders ~1–2 px; keep conservative
    bottom = h - 2 if h > 100 else h
    left, right = 1, w - 1
    return bgr[top:bottom, left:right].copy()


def collect_source_images() -> List[Path]:
    found: List[Path] = []
    seen = set()
    skip_dirs = {"_hud_debug", "minecraft_dataset", "minecraft_yolo", "__pycache__", "preview", "images"}
    skip_name_prefixes = ("_", "hud_", "pred_", "debug_")
    for d in RAW_DIRS:
        if not d.is_dir() or d.name in skip_dirs:
            continue
        for g in IMAGE_GLOBS:
            for p in d.glob(g):
                if p.parent.name in skip_dirs:
                    continue
                if p.name.startswith(skip_name_prefixes):
                    continue
                if "hud" in p.name.lower() and "overlay" in p.name.lower():
                    continue
                key = p.name
                if key in seen:
                    continue
                seen.add(key)
                found.append(p)
    return sorted(found)


def build_local_dataset(out_dir: Path, val_frac: float = 0.2) -> Path:
    images = collect_source_images()
    if not images:
        raise SystemExit(
            "No source images found. Run collect_data.py first, or put PNGs in dataset_raw/."
        )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    train_img = out_dir / "train" / "images"
    train_lbl = out_dir / "train" / "labels"
    val_img = out_dir / "valid" / "images"
    val_lbl = out_dir / "valid" / "labels"
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    shuffled = images[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac)) if len(shuffled) >= 5 else max(1, len(shuffled) // 5)
    # ensure at least 1 train
    if n_val >= len(shuffled):
        n_val = max(1, len(shuffled) - 1)

    labeled = 0
    empty = 0
    for i, src in enumerate(shuffled):
        bgr = cv2.imread(str(src))
        if bgr is None:
            print(f"  skip unreadable {src}")
            continue
        bgr = crop_window_chrome(bgr)
        boxes = auto_label_trees(bgr)
        is_val = i < n_val
        img_dir = val_img if is_val else train_img
        lbl_dir = val_lbl if is_val else train_lbl
        stem = f"{src.stem}_{i:04d}"
        dst_img = img_dir / f"{stem}.png"
        dst_lbl = lbl_dir / f"{stem}.txt"
        cv2.imwrite(str(dst_img), bgr)
        lines = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cx, cy, bw, bh in boxes]
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if boxes:
            labeled += 1
        else:
            empty += 1
        print(f"  {'val' if is_val else 'train'} {src.name}: {len(boxes)} tree box(es)")

    # Light augment: horizontal flip of labeled train images to pad small sets
    _flip_augment(train_img, train_lbl)

    yaml_path = out_dir / "data.yaml"
    # path: . so the same yaml works on Windows and WSL
    yaml_path.write_text(
        "\n".join(
            [
                "path: .",
                "train: train/images",
                "val: valid/images",
                "names:",
                "  0: tree",
                "nc: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"[local] Done. images with trees={labeled}, empty={empty}, "
        f"yaml={yaml_path}"
    )
    return out_dir


def _flip_augment(img_dir: Path, lbl_dir: Path) -> None:
    for img_path in list(img_dir.glob("*.png")):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists() or not lbl_path.read_text().strip():
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        flipped = cv2.flip(bgr, 1)
        out_img = img_dir / f"{img_path.stem}_flip.png"
        out_lbl = lbl_dir / f"{img_path.stem}_flip.txt"
        cv2.imwrite(str(out_img), flipped)
        lines = []
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = parts
            cx_f = 1.0 - float(cx)
            lines.append(f"{cls} {cx_f:.6f} {cy} {bw} {bh}")
        out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_preview(out_dir: Path, n: int = 6) -> None:
    """Draw boxes on a few train images for visual QA."""
    prev = out_dir / "preview"
    prev.mkdir(exist_ok=True)
    imgs = list((out_dir / "train" / "images").glob("*.png"))[:n]
    if not imgs:
        imgs = list((out_dir / "valid" / "images").glob("*.png"))[:n]
    for p in imgs:
        bgr = cv2.imread(str(p))
        h, w = bgr.shape[:2]
        lbl = p.parent.parent / "labels" / f"{p.stem}.txt"
        # handle train/images + train/labels
        if not lbl.exists():
            lbl = out_dir / "train" / "labels" / f"{p.stem}.txt"
        if not lbl.exists():
            lbl = out_dir / "valid" / "labels" / f"{p.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    bgr, "tree", (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
        cv2.imwrite(str(prev / p.name), bgr)
    print(f"[preview] wrote {len(imgs)} images to {prev}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roboflow-only", action="store_true")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    args = ap.parse_args()

    key = (args.api_key or "").strip()
    if args.local_only:
        key = ""

    if key and not args.local_only:
        try:
            download_roboflow(key, OUT)
            write_preview(OUT)
            print("\nNext: py .\\train_yolo.py")
            return 0
        except Exception as e:
            print(f"[roboflow] failed: {e}")
            if args.roboflow_only:
                return 1
            print("[roboflow] falling back to local auto-label bootstrap …")

    if args.roboflow_only:
        print("Set ROBOFLOW_API_KEY (free at https://app.roboflow.com/settings/api)")
        return 1

    build_local_dataset(OUT)
    write_preview(OUT)
    print("\nNext: py .\\train_yolo.py")
    print("Tip: open minecraft_dataset/preview/ to sanity-check boxes.")
    print(
        "For the real Roboflow set later:\n"
        "  set ROBOFLOW_API_KEY=your_key\n"
        "  py .\\setup_dataset.py --roboflow-only"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
