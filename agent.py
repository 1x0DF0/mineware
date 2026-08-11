"""
Closed-loop Minecraft agent: find a tree, walk to it, chop it down.

Reuses capture/input from main.py, HUD parsing from hud.py, and the trained
YOLOv8 weights at minecraft_yolo/run1/weights/best.pt.

Usage (Windows Python, Minecraft in-world with cursor locked):
    py agent.py
    py agent.py --chops 3
    py agent.py --conf 0.2

Ctrl+C exits cleanly and releases held keys/mouse.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pydirectinput
from ultralytics import YOLO

from main import (
    capture_frame,
    enable_dpi_awareness,
    focus_minecraft,
    get_region,
    look,
    mine_or_attack,
    move_forward,
)
from hud import parse_hud
from trees import (
    Detection,
    detect_trees,
    format_dets,
    pick_best_tree,
)

pydirectinput.FAILSAFE = False

# =====================================================================
# Config constants
# =====================================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "minecraft_yolo" / "run1" / "weights" / "best.pt"

# Perception — low YOLO conf; CV fallback fills in when model is silent
CONF_THRESHOLD = 0.20
USE_CV_FALLBACK = True

# Approach / distance (bbox height / frame height)
CLOSE_ENOUGH_FRAC = 0.42       # box taller than this → CHOPPING
LOST_TREE_GRACE = 4            # consecutive frames without a tree before SEARCHING

# Camera / movement
SEARCH_LOOK_DX = 45            # px per SEARCHING tick (slow scan right)
CENTER_DEADZONE_FRAC = 0.08    # |offset|/width below this → "centered"
CENTER_LOOK_GAIN = 0.35        # look_dx = gain * pixel_offset (clamped)
CENTER_LOOK_MAX = 80           # max |look| px per tick while centering
APPROACH_BURST_S = 0.25        # short forward bursts
APPROACH_LOOK_DY = 0           # keep pitch level while approaching (tune if needed)

# Chopping
CHOP_HOLD_S = 0.55             # each mine_or_attack hold duration
MAX_CHOP_ATTEMPTS = 25         # timeout per tree
CHOP_AIM_LOOK_GAIN = 0.25      # micro-adjust aim while chopping
CHOP_AIM_LOOK_MAX = 40

# Loop
TARGET_FPS = 6                 # decision rate cap
STARTUP_COUNTDOWN_S = 5
DEFAULT_MAX_CHOPS = 1          # exit after this many successful chops


# =====================================================================
# Types
# =====================================================================

class State(Enum):
    SEARCHING = auto()
    APPROACHING = auto()
    CHOPPING = auto()
    DONE = auto()


# =====================================================================
# Perception helpers
# =====================================================================

def load_model(path: Path) -> YOLO:
    if not path.is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: {path}\n"
            f"Train first (train_yolo.py) or pass --model <path>"
        )
    print(f"[init] Loading YOLO model from {path} ...")
    model = YOLO(str(path))
    names = model.names if isinstance(model.names, dict) else {
        i: n for i, n in enumerate(model.names)
    }
    print(f"[init] Classes: {names}")
    return model


# =====================================================================
# Action helpers (safe release + thin wrappers)
# =====================================================================

def release_controls() -> None:
    """Always safe to call — drop held W and mouse button."""
    try:
        pydirectinput.keyUp("w")
    except Exception:
        pass
    try:
        pydirectinput.mouseUp()
    except Exception:
        pass


def safe_look(dx: int, dy: int = 0) -> None:
    if dx == 0 and dy == 0:
        return
    look(int(dx), int(dy))


def safe_forward(duration_s: float) -> None:
    if duration_s <= 0:
        return
    move_forward(duration_s)


def safe_chop(hold_s: float) -> None:
    mine_or_attack(hold_s)


def center_look_dx(tree: Detection, frame_w: int) -> int:
    """Pixel look delta to center the tree horizontally. 0 if in deadzone."""
    offset = tree.cx - (frame_w * 0.5)
    if abs(offset) < CENTER_DEADZONE_FRAC * frame_w:
        return 0
    dx = int(round(offset * CENTER_LOOK_GAIN))
    dx = max(-CENTER_LOOK_MAX, min(CENTER_LOOK_MAX, dx))
    # Deadzone escape: never return 0 when outside deadzone
    if dx == 0:
        dx = 1 if offset > 0 else -1
    return dx


# =====================================================================
# State machine
# =====================================================================

@dataclass
class AgentContext:
    state: State = State.SEARCHING
    chops_done: int = 0
    max_chops: int = DEFAULT_MAX_CHOPS
    lost_frames: int = 0
    chop_attempts: int = 0
    last_tree: Optional[Detection] = None
    tick: int = 0


def transition(ctx: AgentContext, new_state: State, reason: str) -> None:
    if new_state is ctx.state:
        return
    print(f"  >> STATE {ctx.state.name} -> {new_state.name}  ({reason})")
    ctx.state = new_state
    if new_state is State.CHOPPING:
        ctx.chop_attempts = 0
    if new_state is State.SEARCHING:
        ctx.lost_frames = 0
        ctx.last_tree = None


def step_searching(ctx: AgentContext, tree: Optional[Detection]) -> str:
    if tree is not None:
        transition(ctx, State.APPROACHING, f"tree {tree.cls_name}:{tree.conf:.2f}")
        ctx.last_tree = tree
        ctx.lost_frames = 0
        return f"action=lock_on conf={tree.conf:.2f}"
    safe_look(SEARCH_LOOK_DX, 0)
    return f"action=look(+{SEARCH_LOOK_DX},0) scan"


def step_approaching(
    ctx: AgentContext,
    tree: Optional[Detection],
    frame_w: int,
    frame_h: int,
) -> str:
    if tree is None:
        ctx.lost_frames += 1
        if ctx.lost_frames >= LOST_TREE_GRACE:
            transition(ctx, State.SEARCHING, f"lost tree for {ctx.lost_frames} frames")
            return "action=abort_approach"
        # keep creeping forward on brief dropouts using last box
        safe_forward(APPROACH_BURST_S * 0.5)
        return f"action=forward(brief) lost={ctx.lost_frames}/{LOST_TREE_GRACE}"

    ctx.lost_frames = 0
    ctx.last_tree = tree
    h_frac = tree.height_frac(frame_h)

    if h_frac >= CLOSE_ENOUGH_FRAC:
        transition(ctx, State.CHOPPING, f"close enough h_frac={h_frac:.2f}")
        return f"action=begin_chop h_frac={h_frac:.2f}"

    dx = center_look_dx(tree, frame_w)
    if dx != 0:
        safe_look(dx, APPROACH_LOOK_DY)
        # if badly off-center, re-aim before walking
        if abs(tree.cx - frame_w * 0.5) > 0.18 * frame_w:
            return f"action=center dx={dx} h_frac={h_frac:.2f}"

    safe_forward(APPROACH_BURST_S)
    return f"action=forward({APPROACH_BURST_S}s) dx={dx} h_frac={h_frac:.2f}"


def step_chopping(
    ctx: AgentContext,
    tree: Optional[Detection],
    frame_w: int,
    frame_h: int,
) -> str:
    ctx.chop_attempts += 1

    if tree is None:
        # Tree gone from view — treat as success if we were actively chopping
        if ctx.chop_attempts >= 3:
            ctx.chops_done += 1
            print(f"  ** chop success? tree vanished after {ctx.chop_attempts} swings "
                  f"(chops_done={ctx.chops_done}/{ctx.max_chops})")
            if ctx.chops_done >= ctx.max_chops:
                transition(ctx, State.DONE, "target chop count reached")
            else:
                transition(ctx, State.SEARCHING, "find next tree")
            return "action=confirm_gone"
        ctx.lost_frames += 1
        if ctx.lost_frames >= LOST_TREE_GRACE:
            transition(ctx, State.SEARCHING, "lost tree while chopping")
            return "action=abort_chop"
        # keep swinging at last aim
        safe_chop(CHOP_HOLD_S)
        return f"action=chop(blind) attempt={ctx.chop_attempts}/{MAX_CHOP_ATTEMPTS}"

    ctx.lost_frames = 0
    ctx.last_tree = tree
    h_frac = tree.height_frac(frame_h)

    # If we somehow drifted far away again, re-approach
    if h_frac < CLOSE_ENOUGH_FRAC * 0.55:
        transition(ctx, State.APPROACHING, f"drifted away h_frac={h_frac:.2f}")
        return f"action=reapproach h_frac={h_frac:.2f}"

    # Micro-center then swing
    dx = center_look_dx(tree, frame_w)
    if dx != 0:
        # softer gain while chopping
        soft = max(-CHOP_AIM_LOOK_MAX, min(CHOP_AIM_LOOK_MAX, int(dx * CHOP_AIM_LOOK_GAIN / max(CENTER_LOOK_GAIN, 1e-3))))
        if soft == 0:
            soft = 1 if dx > 0 else -1
        safe_look(soft, 0)

    if ctx.chop_attempts > MAX_CHOP_ATTEMPTS:
        print(f"  ** chop timeout after {MAX_CHOP_ATTEMPTS} attempts — rescan")
        transition(ctx, State.SEARCHING, "chop timeout")
        return "action=timeout_rescan"

    safe_chop(CHOP_HOLD_S)
    return (
        f"action=chop({CHOP_HOLD_S}s) attempt={ctx.chop_attempts}/{MAX_CHOP_ATTEMPTS} "
        f"h_frac={h_frac:.2f} conf={tree.conf:.2f}"
    )


def tick_once(
    ctx: AgentContext,
    model: YOLO,
    region: dict,
    conf: float,
) -> None:
    ctx.tick += 1
    frame = capture_frame(region)
    h, w = frame.shape[:2]
    # Black / frozen capture → useless
    if frame.mean() < 5:
        print(f"[{ctx.tick:04d}] WARN capture nearly black mean={frame.mean():.1f} region={region}")
    dets = detect_trees(
        frame, model=model, conf=conf, use_cv_fallback=USE_CV_FALLBACK
    )
    tree = pick_best_tree(dets, w)

    # Optional HUD (non-fatal if it fails — menus, darkness, etc.)
    hud_str = ""
    try:
        hud = parse_hud(frame, return_crops=False)
        if hud.ok:
            hud_str = f" HP={hud.health}/20 Food={hud.hunger}/20"
    except Exception:
        pass

    if ctx.state is State.SEARCHING:
        action = step_searching(ctx, tree)
    elif ctx.state is State.APPROACHING:
        action = step_approaching(ctx, tree, w, h)
    elif ctx.state is State.CHOPPING:
        action = step_chopping(ctx, tree, w, h)
    else:
        action = "action=idle"

    tree_str = (
        f"tree={tree.cls_name}:{tree.conf:.2f} "
        f"cx={tree.cx:.0f}/{w} h_frac={tree.height_frac(h):.2f}"
        if tree is not None
        else "tree=None"
    )
    print(
        f"[{ctx.tick:04d}] {ctx.state.name:12s} | dets={format_dets(dets)} | "
        f"{tree_str} | {action}{hud_str}"
    )


# =====================================================================
# Main loop
# =====================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minecraft tree-chop agent")
    p.add_argument("--model", type=Path, default=MODEL_PATH, help="path to best.pt")
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="YOLO confidence")
    p.add_argument("--chops", type=int, default=DEFAULT_MAX_CHOPS, help="successful chops then exit")
    p.add_argument("--fps", type=float, default=TARGET_FPS, help="decision rate cap")
    p.add_argument("--countdown", type=float, default=STARTUP_COUNTDOWN_S)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    period = 1.0 / max(args.fps, 0.5)

    enable_dpi_awareness()
    model = load_model(args.model)

    print(f"Click into the Minecraft game world now — {args.countdown:.0f}s")
    time.sleep(args.countdown)

    win = focus_minecraft()
    region = get_region(win, client_only=True)
    print(f"[init] Capture region: {region}")
    print(
        f"[init] conf={args.conf} cv_fallback={USE_CV_FALLBACK} "
        f"close_frac={CLOSE_ENOUGH_FRAC} fps={args.fps} max_chops={args.chops}"
    )
    print("[init] Goal: SEARCH -> APPROACH -> CHOP tree. Ctrl+C to stop.")

    ctx = AgentContext(max_chops=max(1, args.chops))

    try:
        while ctx.state is not State.DONE:
            t0 = time.perf_counter()
            try:
                # Refresh region only (no re-activate — that steals focus mid-game)
                if ctx.tick % 30 == 0 and ctx.tick > 0:
                    try:
                        wins = __import__("pygetwindow").getWindowsWithTitle("Minecraft")
                        if wins:
                            region = get_region(wins[0], client_only=True)
                    except Exception as e:
                        print(f"[warn] window refresh failed: {e}")

                tick_once(ctx, model, region, args.conf)
            except Exception as e:
                # Per-tick errors shouldn't leave keys stuck; log and continue
                release_controls()
                print(f"[error] tick failed: {type(e).__name__}: {e}")
                time.sleep(0.5)

            elapsed = time.perf_counter() - t0
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

        print(f"[done] Finished. chops_done={ctx.chops_done}")
        return 0
    except KeyboardInterrupt:
        print("\n[exit] Ctrl+C — shutting down")
        return 130
    finally:
        release_controls()
        print("[exit] Controls released (W + mouse)")


if __name__ == "__main__":
    sys.exit(main())
