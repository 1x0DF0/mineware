#run this while playing to cpture one frame which will go into dataset_raw/
#usage py.\collect_data.py


import time
import os
import mss
import numpy as np
import pygetwindow as gw
import cv2


OUTPUT_DIR = "dataset_raw"
CAPTURE_INTERVAL_SECONDS = 1.0  # one frame per second is plenty for labeling
MAX_FRAMES = 2000  # stop after this many, to avoid filling disk unattended


def get_minecraft_region():
    windows = gw.getWindowsWithTitle("Minecraft")
    if not windows:
        raise RuntimeError("Minecraft window not found. Is it running?")
    win = windows[0]
    return {"left": win.left, "top": win.top, "width": win.width, "height": win.height}
 
 
def capture_frame(region):
    with mss.mss() as sct:
        shot = sct.grab(region)
        frame = np.array(shot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame
 
 
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    region = get_minecraft_region()
    print(f"Capturing region: {region}")
    print(f"Saving one frame every {CAPTURE_INTERVAL_SECONDS}s to ./{OUTPUT_DIR}/")
    print("Play normally. Ctrl+C to stop early.")
 
    count = 0
    try:
        while count < MAX_FRAMES:
            frame = capture_frame(region)
            path = os.path.join(OUTPUT_DIR, f"frame_{count:05d}.png")
            cv2.imwrite(path, frame)
            count += 1
            if count % 50 == 0:
                print(f"Captured {count} frames")
            time.sleep(CAPTURE_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print(f"\nStopped early. Total frames captured: {count}")
 
    print(f"Done. {count} frames saved to ./{OUTPUT_DIR}/")
 
 
if __name__ == "__main__":
    main()


