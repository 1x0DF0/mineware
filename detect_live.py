"""
Real-time tree detection on the live Minecraft feed.

YOLO primary + classical CV fallback so you still see boxes when the
bootstrap model is silent.

Usage:
    py detect_live.py
    py detect_live.py --conf 0.2
    py detect_live.py --no-cv
Press q in the display window to quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from main import (
    IMAGES_DIR,
    _ensure_images_dir,
    capture_frame,
    enable_dpi_awareness,
    focus_minecraft,
    get_region,
)
from trees import detect_trees, draw_detections, format_dets, load_yolo

def _default_yolo_weights() -> Path:
    root = Path(__file__).resolve().parent
    for name in ("run2", "run1"):
        p = root / "minecraft_yolo" / name / "weights" / "best.pt"
        if p.is_file():
            return p
    return root / "minecraft_yolo" / "run1" / "weights" / "best.pt"


MODEL_PATH = _default_yolo_weights()
DEFAULT_CONF = 0.25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--no-cv", action="store_true", help="disable classical CV fallback")
    ap.add_argument("--full-window", action="store_true", help="capture full window incl. title bar")
    args = ap.parse_args()

    enable_dpi_awareness()
    try:
        model = load_yolo(args.model)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e
    print(f"conf={args.conf} cv_fallback={not args.no_cv}")

    win = focus_minecraft()
    region = get_region(win, client_only=not args.full_window)
    print(f"Capturing region: {region}")
    print("Green box = YOLO, orange = CV fallback. Yellow cross = frame center.")
    print("Press 'q' to quit. Last empty frame saved to images/last_live.png")

    out_dir = _ensure_images_dir()
    last_region_refresh = 0.0
    empty_streak = 0

    while True:
        now = time.time()
        if now - last_region_refresh > 3.0:
            try:
                import pygetwindow as gw
                wins = gw.getWindowsWithTitle("Minecraft")
                if wins:
                    region = get_region(wins[0], client_only=not args.full_window)
            except Exception as e:
                print(f"[warn] region refresh: {e}")
            last_region_refresh = now

        frame = capture_frame(region)
        mean = float(frame.mean())
        dets = detect_trees(
            frame,
            model=model,
            conf=args.conf,
            use_cv_fallback=not args.no_cv,
        )
        annotated = draw_detections(frame, dets)

        # HUD strip
        status = (
            f"shape={frame.shape[1]}x{frame.shape[0]} mean={mean:.0f} "
            f"dets={len(dets)} {format_dets(dets)}"
        )
        cv2.rectangle(annotated, (0, 0), (min(frame.shape[1], 12 * len(status)), 28), (0, 0, 0), -1)
        cv2.putText(
            annotated, status[:120], (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

        if dets:
            empty_streak = 0
            for d in dets:
                cx, cy = int(d.cx), int(d.cy)
                ox, oy = cx - frame.shape[1] // 2, cy - frame.shape[0] // 2
                print(
                    f"{d.cls_name} ({d.conf:.2f}/{d.source}) "
                    f"box=({int(d.x1)},{int(d.y1)})-({int(d.x2)},{int(d.y2)}) "
                    f"off_center=({ox:+d},{oy:+d})"
                )
        else:
            empty_streak += 1
            print(f"(no detections) frame mean={mean:.1f} shape={frame.shape}")
            if empty_streak in (1, 30, 90):
                path = out_dir / "last_live.png"
                cv2.imwrite(str(path), frame)
                print(f"  saved {path} for debugging")

        cv2.imshow("Minecraft Object Detection", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):
            # toggle full window capture at runtime
            args.full_window = not args.full_window
            print(f"[toggle] full_window={args.full_window}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
