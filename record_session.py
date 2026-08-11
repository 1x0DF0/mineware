"""
Record human Minecraft play for imitation learning.

Each tick (~ --fps Hz) writes:
  - frames/NNNNNN.jpg|png   (optional)
  - meta.jsonl line with:
      t, frame, hud, detections, keys, mouse, mine hold state

Input is polled via Win32 GetAsyncKeyState (works with Minecraft raw input).
pynput is not required on Windows.

Usage:
    py record_session.py --fps 10 --jpeg 85
    py record_session.py --no-frames
    py record_session.py --max-minutes 30

Ctrl+C to stop. Then:
    py mine_stats.py --session sessions/<id>
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from main import (
    capture_frame,
    enable_dpi_awareness,
    focus_minecraft,
    get_region,
)
from hud import parse_hud
from trees import detect_trees, load_yolo, pick_best_tree

ROOT = Path(__file__).resolve().parent
SESSIONS_DIR = ROOT / "sessions"
MODEL_PATH = ROOT / "minecraft_yolo" / "run1" / "weights" / "best.pt"

# Virtual-key codes (Win32)
VK = {
    "w": 0x57,
    "a": 0x41,
    "s": 0x53,
    "d": 0x44,
    "space": 0x20,
    "shift": 0x10,  # either shift
    "ctrl": 0x11,
    "e": 0x45,
    "q": 0x51,
    "f": 0x46,
    "r": 0x52,
    "1": 0x31,
    "2": 0x32,
    "3": 0x33,
    "4": 0x34,
    "5": 0x35,
    "6": 0x36,
    "7": 0x37,
    "8": 0x38,
    "9": 0x39,
    "esc": 0x1B,
    "tab": 0x09,
}
VK_LBUTTON = 0x01
VK_RBUTTON = 0x02
VK_MBUTTON = 0x04

TRACKED_KEYS = list(VK.keys())


# ---------- Win32 input poller ----------

class WinInputPoller:
    """
    Poll key/mouse state each tick with GetAsyncKeyState.
    Minecraft's raw input often never reaches pynput hooks; this does.
    """

    def __init__(self) -> None:
        self.user32 = ctypes.windll.user32
        self._lmb_prev = False
        self._lmb_down_t: Optional[float] = None
        self._last_pos: Optional[Tuple[int, int]] = None
        self.total_mines = 0

    def _down(self, vk: int) -> bool:
        # high bit set ⇒ currently down
        return bool(self.user32.GetAsyncKeyState(vk) & 0x8000)

    def _cursor(self) -> Tuple[int, int]:
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        pt = POINT()
        self.user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)

    def snapshot(self) -> Dict[str, Any]:
        now = time.perf_counter()
        keys = {name: self._down(code) for name, code in VK.items()}
        keys_down = sorted(k for k, v in keys.items() if v)

        lmb = self._down(VK_LBUTTON)
        rmb = self._down(VK_RBUTTON)
        mmb = self._down(VK_MBUTTON)

        completed: List[Dict[str, Any]] = []
        if lmb and not self._lmb_prev:
            self._lmb_down_t = now
        elif not lmb and self._lmb_prev and self._lmb_down_t is not None:
            hold_ms = int((now - self._lmb_down_t) * 1000)
            completed.append({"hold_ms": hold_ms, "ended_t": time.time()})
            self.total_mines += 1
            self._lmb_down_t = None
        self._lmb_prev = lmb

        hold_ms_so_far = 0
        if lmb and self._lmb_down_t is not None:
            hold_ms_so_far = int((now - self._lmb_down_t) * 1000)

        x, y = self._cursor()
        dx = dy = 0
        if self._last_pos is not None:
            dx = x - self._last_pos[0]
            dy = y - self._last_pos[1]
        self._last_pos = (x, y)

        return {
            "keys": keys,
            "keys_down": keys_down,
            "mouse": {
                "lmb": lmb,
                "rmb": rmb,
                "mmb": mmb,
                "dx": dx,
                "dy": dy,
                "x": x,
                "y": y,
            },
            "mine": {
                "lmb_held": lmb,
                "hold_ms_so_far": hold_ms_so_far,
                "completed": completed,
            },
        }


class FallbackPynputPoller:
    """Non-Windows fallback using pynput listeners."""

    def __init__(self) -> None:
        from pynput import keyboard, mouse

        self.keys_down: set = set()
        self.lmb = self.rmb = self.mmb = False
        self.dx = self.dy = 0
        self._last = None
        self._lmb_down_t = None
        self._lmb_prev = False
        self.total_mines = 0
        self._completed: List[Dict[str, Any]] = []

        def on_press(key):
            name = _norm_key_pynput(key)
            if name:
                self.keys_down.add(name)

        def on_release(key):
            name = _norm_key_pynput(key)
            if name:
                self.keys_down.discard(name)

        def on_click(x, y, button, pressed):
            btn = str(button).lower()
            if "left" in btn:
                self.lmb = pressed
            elif "right" in btn:
                self.rmb = pressed
            elif "middle" in btn:
                self.mmb = pressed

        def on_move(x, y):
            if self._last is not None:
                self.dx += int(x - self._last[0])
                self.dy += int(y - self._last[1])
            self._last = (x, y)

        self._kb = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._ms = mouse.Listener(on_click=on_click, on_move=on_move)
        self._kb.start()
        self._ms.start()

    def snapshot(self) -> Dict[str, Any]:
        now = time.perf_counter()
        lmb = self.lmb
        completed: List[Dict[str, Any]] = []
        if lmb and not self._lmb_prev:
            self._lmb_down_t = now
        elif not lmb and self._lmb_prev and self._lmb_down_t is not None:
            hold_ms = int((now - self._lmb_down_t) * 1000)
            completed.append({"hold_ms": hold_ms, "ended_t": time.time()})
            self.total_mines += 1
            self._lmb_down_t = None
        self._lmb_prev = lmb
        hold_ms_so_far = (
            int((now - self._lmb_down_t) * 1000)
            if lmb and self._lmb_down_t is not None
            else 0
        )
        dx, dy = self.dx, self.dy
        self.dx = self.dy = 0
        keys = {k: (k in self.keys_down) for k in TRACKED_KEYS}
        return {
            "keys": keys,
            "keys_down": sorted(self.keys_down),
            "mouse": {"lmb": lmb, "rmb": self.rmb, "mmb": self.mmb, "dx": dx, "dy": dy},
            "mine": {
                "lmb_held": lmb,
                "hold_ms_so_far": hold_ms_so_far,
                "completed": completed,
            },
        }

    def stop(self) -> None:
        try:
            self._kb.stop()
            self._ms.stop()
        except Exception:
            pass


def _norm_key_pynput(key) -> Optional[str]:
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
    return name if name in TRACKED_KEYS else None


def make_poller():
    if sys.platform == "win32":
        print("[init] input: Win32 GetAsyncKeyState poller (game-safe)")
        return WinInputPoller()
    print("[init] input: pynput fallback")
    return FallbackPynputPoller()


# ---------- helpers ----------

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


# ---------- main ----------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Record Minecraft play sessions")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--full-window", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument(
        "--jpeg",
        type=int,
        default=85,
        help="JPEG quality 1-100 (default 85). Use 0 for PNG.",
    )
    args = ap.parse_args(argv)

    enable_dpi_awareness()
    session = new_session_dir()
    meta_path = session / "meta.jsonl"
    frames_dir = session / "frames"

    model = None
    if not args.no_yolo:
        try:
            model = load_yolo(MODEL_PATH)
        except FileNotFoundError:
            print(f"[init] No weights at {MODEL_PATH} — CV-only detections")
            model = None

    print(f"[init] Click into Minecraft — recording in {args.countdown:.0f}s")
    time.sleep(args.countdown)

    win = focus_minecraft()
    region = get_region(win, client_only=not args.full_window)
    print(f"[init] region={region}")
    print(f"[init] session → {session}")
    print(
        f"[init] fps={args.fps} frames={not args.no_frames} "
        f"jpeg={args.jpeg} yolo={model is not None}"
    )
    print("[init] Play normally (WASD + look + hold LMB to mine). Ctrl+C to stop.")

    poller = make_poller()
    period = 1.0 / max(args.fps, 0.5)
    t0 = time.time()
    deadline = t0 + args.max_minutes * 60.0 if args.max_minutes > 0 else None
    tick = 0
    bytes_frames = 0
    ticks_with_keys = 0

    session_info = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "fps_target": args.fps,
        "region": region,
        "save_frames": not args.no_frames,
        "jpeg": args.jpeg,
        "model": str(MODEL_PATH) if model else None,
        "conf": args.conf,
        "input": "win32_poll" if sys.platform == "win32" else "pynput",
    }
    (session / "session.json").write_text(
        json.dumps(session_info, indent=2), encoding="utf-8"
    )

    try:
        with meta_path.open("a", encoding="utf-8") as meta_f:
            while True:
                loop_t0 = time.perf_counter()
                if deadline and time.time() >= deadline:
                    print("[done] max-minutes reached")
                    break

                if tick % 50 == 0 and tick > 0:
                    try:
                        import pygetwindow as gw

                        wins = gw.getWindowsWithTitle("Minecraft")
                        if wins:
                            region = get_region(
                                wins[0], client_only=not args.full_window
                            )
                    except Exception:
                        pass

                wall_t = time.time()
                frame = capture_frame(region)
                h, w = frame.shape[:2]
                inputs = poller.snapshot()
                if inputs["keys_down"]:
                    ticks_with_keys += 1

                hud_dict: Dict[str, Any] = {"ok": False, "skipped": True}
                if not args.no_hud:
                    try:
                        hud_dict = hud_to_dict(parse_hud(frame, return_crops=False))
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
                                "off_x_frac": round(
                                    (bt.cx - w * 0.5) / max(w, 1), 3
                                ),
                            }
                    except Exception as e:
                        best = {"error": str(e)}

                frame_name = None
                if not args.no_frames:
                    stem = f"{tick:06d}"
                    if args.jpeg and args.jpeg > 0:
                        fpath = frames_dir / f"{stem}.jpg"
                        cv2.imwrite(
                            str(fpath),
                            frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg)],
                        )
                    else:
                        fpath = frames_dir / f"{stem}.png"
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
                    print(
                        f"[{tick:05d}] t={wall_t - t0:6.1f}s keys={keys:12s} "
                        f"{held:3s} tree={tree:16s} "
                        f"mines_total={poller.total_mines} "
                        f"key_ticks={ticks_with_keys} "
                        f"disk~{bytes_frames / 1e6:.1f}MB"
                    )

                sleep_for = period - (time.perf_counter() - loop_t0)
                if sleep_for > 0:
                    time.sleep(sleep_for)

    except KeyboardInterrupt:
        print("\n[stop] Ctrl+C")
    finally:
        if hasattr(poller, "stop"):
            poller.stop()

        session_info["ended_utc"] = datetime.now(timezone.utc).isoformat()
        session_info["ticks"] = tick
        session_info["duration_s"] = round(time.time() - t0, 2)
        session_info["frames_bytes"] = bytes_frames
        session_info["mines_total"] = getattr(poller, "total_mines", 0)
        session_info["ticks_with_keys"] = ticks_with_keys
        (session / "session.json").write_text(
            json.dumps(session_info, indent=2), encoding="utf-8"
        )
        print(
            f"[done] ticks={tick} duration={session_info['duration_s']}s "
            f"mines={session_info['mines_total']} key_ticks={ticks_with_keys}"
        )
        print(f"[done] {meta_path}")
        print(f"[done] next:  py mine_stats.py --session {session}")

        # quick integrity hint
        if ticks_with_keys == 0 and tick > 20:
            print(
                "[warn] No keys recorded. Run a terminal *as Admin* if still empty, "
                "or check you're not on a different keyboard layout."
            )
        if getattr(poller, "total_mines", 0) == 0 and tick > 20:
            print("[warn] No completed LMB holds — fully release left mouse between chops.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
