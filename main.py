import ctypes
import sys
import time
from pathlib import Path

import mss
import numpy as np
import pygetwindow as gw
import cv2
import pydirectinput

from hud import parse_hud, draw_hud_overlay

pydirectinput.FAILSAFE = False

IMAGES_DIR = Path(__file__).resolve().parent / "images"

_DPI_AWARE_SET = False


def _ensure_images_dir() -> Path:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    return IMAGES_DIR


def enable_dpi_awareness() -> None:
    """
    Match Win32 coordinate space to physical pixels so pygetwindow / ClientToScreen
    line up with mss. Without this, 125%/150% display scale shifts every box.
    """
    global _DPI_AWARE_SET
    if _DPI_AWARE_SET or sys.platform != "win32":
        return
    try:
        # Per-monitor v2 (Win 10+)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    _DPI_AWARE_SET = True


# ---------- Window handling ----------

def focus_minecraft():
    """Find the Minecraft window, bring it to the foreground, and return it."""
    enable_dpi_awareness()
    windows = gw.getWindowsWithTitle("Minecraft")
    if not windows:
        raise RuntimeError("Minecraft window not found. Is it running?")
    win = windows[0]
    try:
        win.activate()
    except Exception:
        pass
    time.sleep(0.5)  # give the OS a moment to actually raise it
    return win


def get_region(win, client_only: bool = True):
    """
    mss capture region for the Minecraft window.

    client_only=True (default): game framebuffer only — no title bar / borders.
    That keeps boxes aligned with what you see in-world and matches HUD geometry.
    """
    enable_dpi_awareness()
    if client_only and sys.platform == "win32":
        region = _client_region_win32(win)
        if region is not None:
            return region
    return {
        "left": int(win.left),
        "top": int(win.top),
        "width": int(win.width),
        "height": int(win.height),
    }


def _client_region_win32(win):
    """Map HWND client rect to screen pixels via Win32 (DPI-aware)."""
    hwnd = getattr(win, "_hWnd", None) or getattr(win, "hWnd", None)
    if not hwnd:
        return None
    try:
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return None
        pt = wintypes.POINT(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
            return None
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width < 32 or height < 32:
            return None
        return {"left": int(pt.x), "top": int(pt.y), "width": width, "height": height}
    except Exception:
        return None


# ---------- Screen capture ----------

def capture_frame(region):
    """Grab a single frame from the given screen region. Returns a BGR numpy array."""
    with mss.mss() as sct:
        shot = sct.grab(region)
        frame = np.array(shot)  # BGRA
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


# ---------- Input injection ----------

def move_forward(duration_seconds):
    pydirectinput.keyDown('w')
    time.sleep(duration_seconds)
    pydirectinput.keyUp('w')


def look(dx, dy):
    """Move the camera by pixel deltas. Requires the game window to have mouse capture
    (i.e. you're in-game with the cursor locked, not in a menu)."""
    pydirectinput.moveRel(dx, dy, relative=True)


def jump():
    pydirectinput.press('space')


def mine_or_attack(duration_seconds=0.5):
    pydirectinput.mouseDown()
    time.sleep(duration_seconds)
    pydirectinput.mouseUp()


# ---------- Combined capture -> act -> capture loop ----------

def run_loop(steps=5, step_duration=0.5, save_frames=True, save_overlays=True):
    out = _ensure_images_dir()
    win = focus_minecraft()
    region = get_region(win)
    print(f"Region: {region}")
    print(f"Saving frames to {out}")

    for i in range(steps):
        frame = capture_frame(region)
        state = parse_hud(frame)
        if state.ok:
            slots = "".join(
                ("[" if s.selected else " ")
                + ("#" if not s.empty else ".")
                + ("]" if s.selected else " ")
                for s in state.hotbar
            )
            print(
                f"Step {i}: HP={state.health:2d}/20 Food={state.hunger:2d}/20 "
                f"sel={state.selected_slot} hotbar={slots}"
            )
        else:
            print(f"Step {i}: HUD parse failed — {state.error}")

        if save_frames:
            cv2.imwrite(str(out / f"frame_{i}.png"), frame)
        if save_overlays and state.ok:
            cv2.imwrite(str(out / f"hud_overlay_{i}.png"), draw_hud_overlay(frame, state))

        move_forward(step_duration)
        time.sleep(0.2)

    print("Loop complete")


if __name__ == "__main__":
    print("Click into the Minecraft game world now — you have 5 seconds")
    time.sleep(5)

    win = focus_minecraft()
    region = get_region(win)
    print(f"Region: {region}")

    out = _ensure_images_dir()
    frame = capture_frame(region)
    print(f"Captured frame shape: {frame.shape}")
    cv2.imwrite(str(out / "test_capture.png"), frame)

    state = parse_hud(frame)
    if state.ok:
        print(
            f"HUD: HP={state.health}/20 Food={state.hunger}/20 "
            f"sel={state.selected_slot} scale={state.gui_scale:.1f}"
        )
        for s in state.hotbar:
            flag = "SEL" if s.selected else ("item" if not s.empty else "empty")
            print(f"  slot {s.index}: {flag} (occ={s.occupancy:.1f})")
        cv2.imwrite(str(out / "hud_overlay.png"), draw_hud_overlay(frame, state))
        print(f"Saved {out / 'test_capture.png'} + {out / 'hud_overlay.png'}")
    else:
        print(f"HUD parse failed: {state.error}")

    print("Moving forward...")
    move_forward(2)

    print("Jumping...")
    jump()

    print("Looking right...")
    look(200, 0)

    print("Done — check if character moved, jumped, and camera turned")

    # Uncomment to run the full capture/act loop instead of the single test above:
    # run_loop(steps=5, step_duration=0.5)
