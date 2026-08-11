"""
Simple frame dumper for offline labeling / dataset bootstrap.

Writes PNGs to dataset_raw/ at a fixed interval. Capture goes through main.py
(DPI-aware client region) so frames match agent/detect_live geometry.

Usage:
    py collect_data.py
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from main import capture_frame, enable_dpi_awareness, focus_minecraft, get_region

OUTPUT_DIR = Path(__file__).resolve().parent / "dataset_raw"
CAPTURE_INTERVAL_SECONDS = 1.0
MAX_FRAMES = 2000


def main() -> None:
    enable_dpi_awareness()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    win = focus_minecraft()
    region = get_region(win, client_only=True)
    print(f"Capturing region: {region}")
    print(f"Saving one frame every {CAPTURE_INTERVAL_SECONDS}s to {OUTPUT_DIR}/")
    print("Play normally. Ctrl+C to stop early.")

    count = 0
    try:
        while count < MAX_FRAMES:
            # refresh region occasionally (move/resize) without re-focus spam
            if count > 0 and count % 30 == 0:
                try:
                    import pygetwindow as gw

                    wins = gw.getWindowsWithTitle("Minecraft")
                    if wins:
                        region = get_region(wins[0], client_only=True)
                except Exception:
                    pass

            frame = capture_frame(region)
            path = OUTPUT_DIR / f"frame_{count:05d}.png"
            cv2.imwrite(str(path), frame)
            count += 1
            if count % 50 == 0:
                print(f"Captured {count} frames")
            time.sleep(CAPTURE_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print(f"\nStopped early. Total frames captured: {count}")

    print(f"Done. {count} frames saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
