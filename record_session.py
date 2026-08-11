"""
Record human Minecraft play for imitation learning.

Each tick (~ --fps Hz) writes:
  - frames/NNNNNN.png          (optional)
  - meta.jsonl line with:
      t, frame, hud, detections, keys, mouse, mine hold state

Keyboard/mouse are captured via pynput while YOU play (not via pydirectinput).

Usage (Windows, Minecraft in-world):
    py record_session.py
    py record_session.py --fps 10 --conf 0.2
    py record_session.py --no-frames          # actions+hud only (small)
    py record_session.py --max-minutes 30

Stop with Ctrl+C. Then:
    py mine_stats.py --session sessions/<id>
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np

from main import (
    capture_frame,
    enable_dpi_awareness,
    focus_minecraft,
    get_region,
)
from hud import parse_hud
from trees import detect_trees, pick_best_tree

ROOT = Path(__file__).resolve().parent
SESSIONS_DIR = ROOT / "sessions"
MODEL_PATH = ROOT / "minecraft_yolo" / "run1" / "weights" / "best.pt"

# Keys we care about for a movement/mine policy
TRACKED_KEYS = {
    "w", "a", "s", "d",
    "space", "shift", "ctrl",
    "e", "q", "f", "r",
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "esc", "tab",
}


def _norm_key(key) -> Optional[str]:
    """Map pynput key object → stable string, or None if ignored."""
    try:
        from pynput.keyboard import Key
    except ImportError:
        return None

    if hasattr(key, "char") and key.char:
        c = key.char.lower()
        return c if c in TRACKED_KEYS else None
    mapping = {
        Key.space: "space",
        Key.shift: "shift",
        Key.shift_l: "shift",
        Key.shift_r: "shift",
        Key.ctrl: "ctrl",
        Key.ctrl_l: "ctrl",
        Key.ctrl_r: "ctrl",
        Key.esc: "esc",
        Key.tab: "tab",
    }
    name = mapping.get(key)
    if name and name in TRACKED_KEYS:
        return name
    return None


@dataclass
class InputState:
    """Thread-safe snapshot of keyboard / mouse since last tick."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    keys_down: Set[str] = field(default_factory=set)
    lmb: bool = False
    rmb: bool = False
    mmb: bool = False
    # mouse absolute last sample + accumulated relative delta this tick
    last_x: Optional[int] = None
    last_y: Optional[int] = None
    dx: int = 0
    dy: int = 0
    # mine timing
    lmb_down_t: Optional[float] = None
    # completed holds waiting to be flushed into the next meta line
    completed_mines: List[Dict[str, Any]] = field(default_factory=list)

    def on_key_down(self, name: str) -> None:
        with self.lock:
            self.keys_down.add(name)

    def on_key_up(self, name: str) -> None:
        with self.lock:
            self.keys_down.discard(name)

    def on_click(self, button_name: str, pressed: bool) -> None:
        with self.lock:
            now = time.perf_counter()
            if button_name == "lmb":
                if pressed:
                    self.lmb = True
                    if self.lmb_down_t is None:
                        self.lmb_down_t = now
                else:
                    self.lmb = False
                    if self.lmb_down_t is not None:
                        hold_ms = int((now - self.lmb_down_t) * 1000)
                        self.completed_mines.append(
                            {"hold_ms": hold_ms, "ended_t": time.time()}
                        )
                        self.lmb_down_t = None
            elif button_name == "rmb":
                self.rmb = pressed
            elif button_name == "mmb":
                self.mmb = pressed

    def on_move(self, x: int, y: int) -> None:
        with self.lock:
            if self.last_x is not None and self.last_y is not None:
                self.dx += int(x - self.last_x)
                self.dy += int(y - self.last_y)
            self.last_x = int(x)
            self.last_y = int(y)

    def snapshot_and_reset_deltas(self) -> Dict[str, Any]:
        with self.lock:
            now = time.perf_counter()
            hold_ms = 0
            if self.lmb and self.lmb_down_t is not None:
                hold_ms = int((now - self.lmb_down_t) * 1000)
            mines = list(self.completed_mines)
            self.completed_mines.clear()
            dx, dy = self.dx, self.dy
            self.dx = 0
            self.dy = 0
            keys = sorted(self.keys_down)
            return {
                "keys": {k: (k in self.keys_down) for k in TRACKED_KEYS},
                "keys_down": keys,
                "mouse": {
                    "lmb": self.lmb,
                    "rmb": self.rmb,
                    "mmb": self.mmb,
                    "dx": dx,
                    "dy": dy,
                },
                "mine": {
                    "lmb_held": self.lmb,
                    "hold_ms_so_far": hold_ms,
                    "completed": mines,  # holds that ended this tick
                },
            }


def start_listeners(state: InputState):
    from pynput import keyboard, mouse

    def on_press(key):
        name = _norm_key(key)
        if name and name in TRACKED_KEYS:
            state.on_key_down(name)

    def on_release(key):
        name = _norm_key(key)
        if name and name in TRACKED_KEYS:
            state.on_key_up(name)

    def on_click(x, y, button, pressed):
        btn = str(button).lower()
        if "left" in btn:
            state.on_click("lmb", pressed)
        elif "right" in btn:
            state.on_click("rmb", pressed)
        elif "middle" in btn:
            state.on_click("mmb", pressed)

    def on_move(x, y):
        state.on_move(x, y)

    kb = keyboard.Listener(on_press=on_press, on_release=on_release)
    ms = mouse.Listener(on_click=on_click, on_move=on_move)
    kb.start()
    ms.start()
    return kb, ms


def hud_to_dict(hud) -> Dict[str, Any]:
    if hud is None or not getattr(hud, "ok", False):
        return {"ok": False, "error": getattr(hud, "error", "fail")}
    return {
        "ok": True,
        "health": hud.health,
        "hunger": hud.hunger,
        "selected_slot": hud.selected_slot,
        "hotbar": [
            {
                "index": s.index,
                "empty": s.empty,
                "selected": s.selected,
                "occupancy": round(s.occupancy, 2),
            }
            for s in hud.hotbar
        ],
        "gui_scale": hud.gui_scale,
    }


def dets_to_list(dets, frame_h: int) -> List[Dict[str, Any]]:
    out = []
    for d in dets:
        out.append(
            {
                "cls": d.cls_name,
                "conf": round(d.conf, 3),
                "source": d.source,
                "xyxy": [round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)],
                "cx": round(d.cx, 1),
                "cy": round(d.cy, 1),
                "h_frac": round(d.height_frac(frame_h), 3),
            }
        )
    return out


def new_session_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SESSIONS_DIR / stamp
    (path / "frames").mkdir(parents=True, exist_ok=True)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Record Minecraft play sessions")
    ap.add_argument("--fps", type=float, default=10.0, help="record rate")
    ap.add_argument("--conf", type=float, default=0.2, help="YOLO conf")
    ap.add_argument("--no-frames", action="store_true", help="skip saving PNGs")
    ap.add_argument("--no-yolo", action="store_true", help="skip detection (faster)")
    ap.add_argument("--no-hud", action="store_true", help="skip HUD parse")
    ap.add_argument("--no-cv", action="store_true", help="YOLO only, no CV fallback")
    ap.add_argument("--full-window", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0, help="0 = until Ctrl+C")
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--jpeg", type=int, default=0, help="if >0 save JPEG quality instead of PNG")
    args = ap.parse_args(argv)

    try:
        import pynput  # noqa: F401
    except ImportError:
        print("Missing pynput. Install:  py -m pip install pynput")
        return 1

    enable_dpi_awareness()
    session = new_session_dir()
    meta_path = session / "meta.jsonl"
    frames_dir = session / "frames"

    model = None
    if not args.no_yolo:
        if MODEL_PATH.is_file():
            from ultralytics import YOLO
            print(f"[init] Loading YOLO {MODEL_PATH}")
            model = YOLO(str(MODEL_PATH))
        else:
            print(f"[init] No weights at {MODEL_PATH} — CV-only detections")

    print(f"[init] Click into Minecraft — recording in {args.countdown:.0f}s")
    time.sleep(args.countdown)

    win = focus_minecraft()
    region = get_region(win, client_only=not args.full_window)
    print(f"[init] region={region}")
    print(f"[init] session → {session}")
    print(f"[init] fps={args.fps} frames={not args.no_frames} yolo={model is not None}")
    print("[init] Play normally. Ctrl+C to stop.")

    state = InputState()
    kb_listener, ms_listener = start_listeners(state)

    period = 1.0 / max(args.fps, 0.5)
    t0 = time.time()
    deadline = t0 + args.max_minutes * 60.0 if args.max_minutes > 0 else None
    tick = 0
    bytes_frames = 0

    session_info = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "fps_target": args.fps,
        "region": region,
        "save_frames": not args.no_frames,
        "model": str(MODEL_PATH) if model else None,
        "conf": args.conf,
    }
    (session / "session.json").write_text(json.dumps(session_info, indent=2), encoding="utf-8")

    try:
        with meta_path.open("a", encoding="utf-8") as meta_f:
            while True:
                loop_t0 = time.perf_counter()
                if deadline and time.time() >= deadline:
                    print("[done] max-minutes reached")
                    break

                # refresh region occasionally without stealing focus
                if tick % 50 == 0 and tick > 0:
                    try:
                        import pygetwindow as gw
                        wins = gw.getWindowsWithTitle("Minecraft")
                        if wins:
                            region = get_region(wins[0], client_only=not args.full_window)
                    except Exception:
                        pass

                wall_t = time.time()
                frame = capture_frame(region)
                h, w = frame.shape[:2]

                # inputs for this tick
                inputs = state.snapshot_and_reset_deltas()

                # perception
                hud_dict: Dict[str, Any] = {"ok": False, "skipped": True}
                if not args.no_hud:
                    try:
                        hud = parse_hud(frame, return_crops=False)
                        hud_dict = hud_to_dict(hud)
                    except Exception as e:
                        hud_dict = {"ok": False, "error": str(e)}

                dets_list: List[Dict[str, Any]] = []
                best = None
                if not args.no_yolo:
                    try:
                        dets = detect_trees(
                            frame,
                            model=model,
                            conf=args.conf,
                            use_cv_fallback=not args.no_cv,
                        )
                        dets_list = dets_to_list(dets, h)
                        bt = pick_best_tree(dets, w)
                        if bt is not None:
                            best = {
                                "cls": bt.cls_name,
                                "conf": round(bt.conf, 3),
                                "source": bt.source,
                                "cx": round(bt.cx, 1),
                                "cy": round(bt.cy, 1),
                                "h_frac": round(bt.height_frac(h), 3),
                                "off_x_frac": round((bt.cx - w * 0.5) / max(w, 1), 3),
                            }
                    except Exception as e:
                        dets_list = []
                        best = {"error": str(e)}

                frame_name = None
                if not args.no_frames:
                    frame_name = f"{tick:06d}"
                    if args.jpeg > 0:
                        fpath = frames_dir / f"{frame_name}.jpg"
                        cv2.imwrite(str(fpath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
                    else:
                        fpath = frames_dir / f"{frame_name}.png"
                        cv2.imwrite(str(fpath), frame)
                    try:
                        bytes_frames += fpath.stat().st_size
                    except OSError:
                        pass
                    frame_name = fpath.name

                rec = {
                    "tick": tick,
                    "t": wall_t,
                    "t_rel": round(wall_t - t0, 4),
                    "frame": frame_name,
                    "shape": [int(h), int(w)],
                    "hud": hud_dict,
                    "detections": dets_list,
                    "best_tree": best,
                    "actions": {
                        "keys_down": inputs["keys_down"],
                        "keys": inputs["keys"],
                        "mouse": inputs["mouse"],
                    },
                    "mine": inputs["mine"],
                }
                meta_f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                meta_f.flush()

                tick += 1
                if tick % max(1, int(args.fps * 5)) == 0:
                    held = "LMB" if inputs["mine"]["lmb_held"] else ""
                    keys = ",".join(inputs["keys_down"]) or "-"
                    tree = (
                        f"{best['cls']}:{best['conf']:.2f}"
                        if best and "conf" in best
                        else "none"
                    )
                    n_done = len(inputs["mine"]["completed"])
                    print(
                        f"[{tick:05d}] t={wall_t - t0:6.1f}s keys={keys:12s} "
                        f"{held:3s} tree={tree:16s} "
                        f"mine_done={n_done} disk~{bytes_frames / 1e6:.1f}MB"
                    )

                elapsed = time.perf_counter() - loop_t0
                sleep_for = period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

    except KeyboardInterrupt:
        print("\n[stop] Ctrl+C")
    finally:
        try:
            kb_listener.stop()
            ms_listener.stop()
        except Exception:
            pass

        # finalize session.json
        session_info["ended_utc"] = datetime.now(timezone.utc).isoformat()
        session_info["ticks"] = tick
        session_info["duration_s"] = round(time.time() - t0, 2)
        session_info["frames_bytes"] = bytes_frames
        (session / "session.json").write_text(
            json.dumps(session_info, indent=2), encoding="utf-8"
        )
        print(f"[done] ticks={tick} duration={session_info['duration_s']}s")
        print(f"[done] {meta_path}")
        print(f"[done] next:  py mine_stats.py --session {session}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
