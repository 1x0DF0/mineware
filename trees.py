"""
Tree perception: YOLO primary + classical CV fallback.

The bootstrap YOLO model is small and often silent live. Classical canopy+trunk
detection keeps the agent/detect_live useful until a better labeled model exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore


@dataclass
class Detection:
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    source: str = "yolo"  # "yolo" | "cv"

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def height_frac(self, frame_h: int) -> float:
        return self.height / max(frame_h, 1)


TREE_CLASS_HINTS = (
    "tree", "log", "wood", "oak", "birch", "spruce", "jungle", "acacia", "dark_oak",
)


def is_tree_class(name: str) -> bool:
    n = name.lower().replace(" ", "_")
    return any(h in n for h in TREE_CLASS_HINTS)


# ---------- classical CV ----------

def _wood_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    brown = cv2.inRange(hsv, (5, 40, 30), (25, 255, 190))
    birch = cv2.inRange(hsv, (0, 0, 130), (40, 55, 240))
    return brown | birch


def _leaf_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Slightly looser than training auto-label for live recall
    return cv2.inRange(hsv, (28, 35, 18), (95, 255, 165))


def detect_trees_cv(bgr: np.ndarray, max_boxes: int = 8) -> List[Detection]:
    """
    Canopy + trunk heuristic → pixel xyxy boxes.
    Tuned for recall so the agent still sees *something* when YOLO is silent.
    """
    h, w = bgr.shape[:2]
    if h < 64 or w < 64:
        return []

    roi = bgr.copy()
    y0, y1 = int(h * 0.08), int(h * 0.88)
    roi[:y0, :] = 0
    roi[y1:, :] = 0

    leaves = _leaf_mask(roi)
    wood = _wood_mask(roi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaves = cv2.morphologyEx(leaves, cv2.MORPH_OPEN, k, iterations=1)
    k_split = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    seeds = cv2.erode(leaves, k_split, iterations=1)
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, k, iterations=1)

    n, _, stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    out: List[Detection] = []
    min_area = (h * w) * 0.001
    max_area = (h * w) * 0.35

    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area or area > max_area:
            continue
        if bw < 10 or bh < 10:
            continue
        if bw / max(bh, 1) > 3.2:
            continue

        pad = max(bw, bh)
        gx1, gy1 = max(0, x - pad), max(y0, y - pad)
        gx2, gy2 = min(w, x + bw + pad), min(y1, y + bh + pad)
        region = leaves[gy1:gy2, gx1:gx2]
        if region.size == 0 or np.count_nonzero(region) < 40:
            continue
        ys, xs = np.where(region > 0)
        x = gx1 + int(xs.min())
        y = gy1 + int(ys.min())
        bw = int(xs.max() - xs.min() + 1)
        bh = int(ys.max() - ys.min() + 1)
        if bw * bh > (h * w) * 0.40:
            continue

        cx_px = x + bw // 2
        stem_half = max(8, bw // 4)
        sx1, sx2 = max(0, cx_px - stem_half), min(w, cx_px + stem_half)
        sy1 = min(h - 1, y + int(bh * 0.5))
        sy2 = min(y1, y + bh + int(bh * 0.9) + 30)
        if sy2 <= sy1 + 3:
            continue
        stem = wood[sy1:sy2, sx1:sx2]
        wood_frac = float(np.count_nonzero(stem)) / max(stem.size, 1)
        canopy = leaves[y : y + bh, x : x + bw]
        leaf_frac = float(np.count_nonzero(canopy)) / max(canopy.size, 1)
        if leaf_frac < 0.12:
            continue

        large_close = (bw * bh) > (h * w) * 0.05 and (y + bh) > h * 0.4
        if wood_frac < 0.02 and not large_close:
            continue

        expand_down = int(bh * 0.5) if wood_frac >= 0.02 else int(bh * 0.15)
        expand_x = int(bw * 0.06)
        bx1 = max(0, x - expand_x)
        by1 = max(y0, y - int(bh * 0.04))
        bx2 = min(w - 1, x + bw + expand_x)
        by2 = min(y1, y + bh + expand_down)

        if wood_frac >= 0.04:
            stem_cols = np.count_nonzero(stem, axis=0)
            if stem_cols.size and stem_cols.max() > 0:
                stem_cx = sx1 + int(np.argmax(stem_cols))
                half_w = max(bw // 2, int(0.07 * w))
                bx1 = max(0, stem_cx - half_w)
                bx2 = min(w - 1, stem_cx + half_w)

        nw = (bx2 - bx1) / w
        nh = (by2 - by1) / h
        if nw < 0.03 or nh < 0.05 or nw > 0.55 or nh > 0.70:
            continue

        # conf proxy: leaf density + trunk evidence
        conf = float(np.clip(0.35 + 0.4 * leaf_frac + 0.5 * min(wood_frac, 0.3), 0.2, 0.95))
        out.append(
            Detection(
                cls_name="tree",
                conf=conf,
                x1=float(bx1),
                y1=float(by1),
                x2=float(bx2),
                y2=float(by2),
                source="cv",
            )
        )

    out = _nms(out, iou_thresh=0.4)
    out.sort(key=lambda d: d.conf, reverse=True)
    return out[:max_boxes]


def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = a.area + b.area - inter
    return inter / ua if ua > 0 else 0.0


def _nms(dets: List[Detection], iou_thresh: float) -> List[Detection]:
    dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    keep: List[Detection] = []
    for d in dets:
        if all(_iou(d, k) < iou_thresh for k in keep):
            keep.append(d)
    return keep


# ---------- YOLO ----------

def detect_trees_yolo(
    model,
    frame: np.ndarray,
    conf: float = 0.2,
    iou: float = 0.5,
    max_det: int = 15,
) -> List[Detection]:
    if model is None or frame is None or frame.size == 0:
        return []
    results = model.predict(frame, conf=conf, iou=iou, max_det=max_det, verbose=False)
    out: List[Detection] = []
    if not results:
        return out
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return out
    names = model.names
    for box in r0.boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id] if isinstance(names, dict) else names[cls_id]
        if not is_tree_class(str(cls_name)):
            # still keep if single-class model misnamed — only filter multi-class noise
            if len(names) > 1:
                continue
        xyxy = box.xyxy[0].tolist()
        out.append(
            Detection(
                cls_name=str(cls_name),
                conf=float(box.conf[0]),
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
                source="yolo",
            )
        )
    return out


def detect_trees(
    frame: np.ndarray,
    model=None,
    conf: float = 0.2,
    use_cv_fallback: bool = True,
    cv_if_yolo_empty: bool = True,
) -> List[Detection]:
    """
    Primary: YOLO. If empty (or always merge), classical CV.

    Default: YOLO first; if zero boxes, run CV fallback.
    """
    yolo_dets = detect_trees_yolo(model, frame, conf=conf) if model is not None else []
    if yolo_dets and not use_cv_fallback:
        return yolo_dets
    if yolo_dets and cv_if_yolo_empty:
        return yolo_dets
    # empty YOLO or forced hybrid
    cv_dets = detect_trees_cv(frame) if use_cv_fallback else []
    if not yolo_dets:
        return cv_dets
    # merge (rare path)
    return _nms(yolo_dets + cv_dets, iou_thresh=0.45)


def pick_best_tree(dets: Sequence[Detection], frame_w: int) -> Optional[Detection]:
    trees = [d for d in dets if is_tree_class(d.cls_name)]
    if not trees:
        return None
    cx = frame_w * 0.5

    def score(d: Detection) -> Tuple[float, float, float]:
        # prefer higher conf, then larger, then closer to center
        return (d.conf, d.area, -abs(d.cx - cx))

    return max(trees, key=score)


def format_dets(dets: Sequence[Detection]) -> str:
    if not dets:
        return "(none)"
    return ", ".join(f"{d.cls_name}:{d.conf:.2f}[{d.source}]" for d in dets)


def draw_detections(frame: np.ndarray, dets: Sequence[Detection]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.drawMarker(
        out, (w // 2, h // 2), (0, 255, 255),
        markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1,
    )
    for d in dets:
        color = (0, 255, 0) if d.source == "yolo" else (255, 180, 0)
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, 2)
        label = f"{d.cls_name} {d.conf:.2f} ({d.source})"
        cv2.putText(
            out, label, (p1[0], max(16, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return out
