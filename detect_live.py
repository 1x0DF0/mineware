"""
Real-time object detection on the live Minecraft feed using your trained
YOLOv8 model. Captures the game *client* area (no title bar), runs detection,
draws boxes, and prints structured results.

Usage:
    py .\detect_live.py
    py .\detect_live.py --conf 0.45
Press 'q' in the display window to quit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from main import capture_frame, enable_dpi_awareness, focus_minecraft, get_region

MODEL_PATH = Path(__file__).resolve().parent / "minecraft_yolo" / "run1" / "weights" / "best.pt"
DEFAULT_CONF = 0.35
IOU_NMS = 0.50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=MODEL_PATH)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--iou", type=float, default=IOU_NMS)
    args = ap.parse_args()

    enable_dpi_awareness()
    if not args.model.is_file():
        raise SystemExit(f"Missing weights: {args.model}\nRun train_yolo.py first.")

    model = YOLO(str(args.model))
    print(f"Model classes: {model.names}")
    print(f"conf={args.conf} iou={args.iou}")

    win = focus_minecraft()
    region = get_region(win, client_only=True)
    print(f"Capturing CLIENT region: {region}")
    print("Boxes are drawn on this same client image — compare inside the OpenCV window,")
    print("not against a different window size. Press 'q' to quit.")

    last_refresh = 0.0
    while True:
        now = time.time()
        # Refresh region occasionally (move/resize / DPI)
        if now - last_refresh > 2.0:
            try:
                win = focus_minecraft()
                region = get_region(win, client_only=True)
            except Exception as e:
                print(f"[warn] region refresh: {e}")
            last_refresh = now

        frame = capture_frame(region)
        results = model.predict(
            frame,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
            # max_det keeps the OpenCV window readable
            max_det=15,
        )
        r0 = results[0]
        annotated = r0.plot()

        # Crosshair at frame center — agent aims relative to this
        h, w = frame.shape[:2]
        cv2.drawMarker(
            annotated, (w // 2, h // 2), (0, 255, 255),
            markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1,
        )

        if r0.boxes is not None and len(r0.boxes):
            for box in r0.boxes:
                cls_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0].tolist()]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                # offset of box center from frame center (what the agent uses)
                ox, oy = cx - w // 2, cy - h // 2
                print(
                    f"{cls_name} ({conf:.2f}) box=({x1},{y1})-({x2},{y2}) "
                    f"center=({cx},{cy}) off_center=({ox:+d},{oy:+d})"
                )
        else:
            print("(no detections)")

        cv2.imshow("Minecraft Object Detection (client area)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
