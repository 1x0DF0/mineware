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
    load_yolo,
)

pydirectinput.FAILSAFE = False

# =====================================================================
# Config constants
# =====================================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "minecraft_yolo" / "run1" / "weights" / "best.pt"

# Perception — low YOLO conf; CV fallback fills in when model is silent
CONF_THRESHOLD = 0.25
USE_CV_FALLBACK = True
# While chopping, ignore CV (it invents many small distant "trees" and thrashs FSM)
CV_FALLBACK_WHILE_CHOPPING = False
MIN_ACTION_CONF = 0.35         # ignore ultra-weak YOLO boxes for control

# Approach / distance (bbox height / frame height)
# Log showed h_frac~0.8 on huge off-center foliage → instant CHOPPING into air.
# Require BOTH size and centering before swinging.
CLOSE_ENOUGH_FRAC = 0.38
CENTER_CHOP_FRAC = 0.12        # |cx - mid| / width must be ≤ this to start/keep chop
LOST_TREE_GRACE = 5            # consecutive frames without a tree before SEARCHING
STICKY_MAX_JUMP_FRAC = 0.22    # reject new pick farther than this from last_cx (fraction of W)

# Camera / movement
SEARCH_LOOK_DX = 45            # px per SEARCHING tick (slow scan right)
CENTER_DEADZONE_FRAC = 0.08    # |offset|/width below this → "centered" for look
CENTER_LOOK_GAIN = 0.40        # look_dx = gain * pixel_offset (clamped)
CENTER_LOOK_MAX = 100          # max |look| px per tick while centering
APPROACH_BURST_S = 0.28        # short forward bursts
APPROACH_LOOK_DY = 0           # keep pitch level while approaching (tune if needed)

# Chopping
CHOP_HOLD_S = 0.55             # each mine_or_attack hold duration (overridden by BC hold table)
MAX_CHOP_ATTEMPTS = 25         # timeout per tree
CHOP_AIM_LOOK_GAIN = 0.25      # micro-adjust aim while chopping
CHOP_AIM_LOOK_MAX = 40
BC_LOOK_MAG = 45               # look pixels when policy says look_left/right
BC_FORWARD_S = 0.20            # forward burst when policy says forward

# Loop
TARGET_FPS = 6                 # decision rate cap
STARTUP_COUNTDOWN_S = 5
DEFAULT_MAX_CHOPS = 1          # exit after this many successful chops
POLICY_PATH = ROOT / "policy" / "bc_mlp.pt"


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


def is_centered(tree: Detection, frame_w: int, frac: float = CENTER_CHOP_FRAC) -> bool:
    return abs(tree.cx - frame_w * 0.5) <= frac * frame_w


def pick_control_tree(
    dets: list,
    frame_w: int,
    frame_h: int,
    last: Optional[Detection],
    prefer_yolo: bool,
) -> Optional[Detection]:
    """
    Sticky target selection for the FSM.

    Prefer: YOLO over CV when available, near last_cx (don't hop forests),
    reasonable conf, not a random edge blob.
    """
    trees = [d for d in dets if d.cls_name and "tree" in d.cls_name.lower()]
    if not trees:
        return None

    if prefer_yolo:
        yolo = [d for d in trees if d.source == "yolo" and d.conf >= MIN_ACTION_CONF]
        if yolo:
            trees = yolo
        else:
            # weak YOLO only — still allow but filter conf
            yolo_any = [d for d in trees if d.source == "yolo"]
            if yolo_any:
                trees = [d for d in yolo_any if d.conf >= CONF_THRESHOLD]
    else:
        trees = [d for d in trees if d.conf >= CONF_THRESHOLD or d.source == "cv"]

    if not trees:
        return None

    mid = frame_w * 0.5

    def score(d: Detection) -> float:
        # higher is better
        s = float(d.conf)
        if d.source == "yolo":
            s += 0.25
        # soft preference for nearer-center when acquiring
        s -= 0.9 * (abs(d.cx - mid) / max(frame_w, 1))
        # stickiness: heavily prefer continuity with last lock
        if last is not None:
            jump = abs(d.cx - last.cx) / max(frame_w, 1)
            s -= 2.5 * jump
            # mild vertical continuity
            s -= 0.5 * (abs(d.cy - last.cy) / max(frame_h, 1))
        # reject absurd full-frame wallpaper boxes slightly by penalizing width
        s -= 0.3 * (d.width / max(frame_w, 1))
        return s

    best = max(trees, key=score)

    # Hard sticky gate: if we have a lock, don't accept a jump across the screen
    if last is not None:
        jump = abs(best.cx - last.cx) / max(frame_w, 1)
        if jump > STICKY_MAX_JUMP_FRAC:
            # try nearest to last instead of highest score far away
            near = [
                d for d in trees
                if abs(d.cx - last.cx) / max(frame_w, 1) <= STICKY_MAX_JUMP_FRAC
            ]
            if near:
                best = max(near, key=score)
            else:
                # keep last geometry as ghost (caller may treat as brief dropout)
                return None
    return best


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
        transition(ctx, State.APPROACHING, f"tree {tree.cls_name}:{tree.conf:.2f}/{tree.source}")
        ctx.last_tree = tree
        ctx.lost_frames = 0
        return f"action=lock_on conf={tree.conf:.2f} src={tree.source}"
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
        # hold position — don't walk blind toward a lost lock
        return f"action=wait_lock lost={ctx.lost_frames}/{LOST_TREE_GRACE}"

    ctx.lost_frames = 0
    ctx.last_tree = tree
    h_frac = tree.height_frac(frame_h)
    off = (tree.cx - frame_w * 0.5) / max(frame_w, 1)

    # 1) Always center first if outside chop deadzone
    dx = center_look_dx(tree, frame_w)
    if not is_centered(tree, frame_w, CENTER_CHOP_FRAC):
        if dx != 0:
            safe_look(dx, APPROACH_LOOK_DY)
        return (
            f"action=center dx={dx} off={off:+.2f} h_frac={h_frac:.2f} "
            f"src={tree.source}"
        )

    # 2) Centered + big enough → chop
    if h_frac >= CLOSE_ENOUGH_FRAC and is_centered(tree, frame_w, CENTER_CHOP_FRAC):
        transition(
            ctx,
            State.CHOPPING,
            f"close+centered h_frac={h_frac:.2f} off={off:+.2f}",
        )
        return f"action=begin_chop h_frac={h_frac:.2f} off={off:+.2f}"

    # 3) Centered but far → walk in
    safe_forward(APPROACH_BURST_S)
    return (
        f"action=forward({APPROACH_BURST_S}s) h_frac={h_frac:.2f} "
        f"off={off:+.2f} src={tree.source}"
    )


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
        # keep swinging at last aim (do not re-pick random forest blobs)
        safe_chop(CHOP_HOLD_S)
        return f"action=chop(blind) attempt={ctx.chop_attempts}/{MAX_CHOP_ATTEMPTS}"

    ctx.lost_frames = 0
    ctx.last_tree = tree
    h_frac = tree.height_frac(frame_h)
    off = (tree.cx - frame_w * 0.5) / max(frame_w, 1)

    # Only re-approach if *this sticky target* got small AND we are still
    # roughly on it. Do not chase a different far blob (old bug from CV).
    if h_frac < CLOSE_ENOUGH_FRAC * 0.45 and is_centered(tree, frame_w, CENTER_CHOP_FRAC * 1.5):
        transition(ctx, State.APPROACHING, f"target shrunk h_frac={h_frac:.2f}")
        return f"action=reapproach h_frac={h_frac:.2f}"

    # If lock jumped off-center a lot, re-center before swinging again
    if not is_centered(tree, frame_w, CENTER_CHOP_FRAC * 1.4):
        dx = center_look_dx(tree, frame_w)
        if dx != 0:
            soft = max(
                -CHOP_AIM_LOOK_MAX,
                min(
                    CHOP_AIM_LOOK_MAX,
                    int(dx * CHOP_AIM_LOOK_GAIN / max(CENTER_LOOK_GAIN, 1e-3)),
                ),
            )
            if soft == 0:
                soft = 1 if dx > 0 else -1
            safe_look(soft, 0)
        return (
            f"action=reaim dx off={off:+.2f} attempt={ctx.chop_attempts}/"
            f"{MAX_CHOP_ATTEMPTS}"
        )

    if ctx.chop_attempts > MAX_CHOP_ATTEMPTS:
        print(f"  ** chop timeout after {MAX_CHOP_ATTEMPTS} attempts — rescan")
        transition(ctx, State.SEARCHING, "chop timeout")
        return "action=timeout_rescan"

    safe_chop(CHOP_HOLD_S)
    return (
        f"action=chop({CHOP_HOLD_S}s) attempt={ctx.chop_attempts}/{MAX_CHOP_ATTEMPTS} "
        f"h_frac={h_frac:.2f} off={off:+.2f} conf={tree.conf:.2f} src={tree.source}"
    )


def step_bc(ctx: AgentContext, tree, hud, frame_w: int, frame_h: int, bc) -> str:
    """
    Behavioral cloning tick: execute policy multi-label actions.
    Still uses tree disappearance to count successful chops.
    """
    act = bc.predict_live(tree, hud, frame_w, frame_h)
    parts = []

    # look first so forward goes toward target
    ldx = act.look_dx(BC_LOOK_MAG)
    if ldx != 0:
        safe_look(ldx, 0)
        parts.append(f"look={ldx}")

    if act.jump:
        from main import jump
        jump()
        parts.append("jump")

    if act.forward and not act.back:
        safe_forward(BC_FORWARD_S)
        parts.append(f"fwd({BC_FORWARD_S})")
    elif act.back:
        # no move_back helper — small look turn instead of walking into danger
        parts.append("back(ignored)")

    # strafe via brief look+forward not implemented; record only
    if act.left:
        parts.append("left")
    if act.right:
        parts.append("right")

    slot = hud.selected_slot if hud is not None and getattr(hud, "ok", False) else 0
    hold_s = bc.hold_s_for_slot(slot if slot is not None and slot >= 0 else 0)
    # clamp insane p90 from accidental long holds in data
    hold_s = float(min(max(hold_s, 0.2), 3.0))

    if act.mine:
        safe_chop(hold_s)
        parts.append(f"mine({hold_s:.2f}s)")
        ctx.chop_attempts += 1

    # success heuristic: were mining and tree vanished
    if tree is None and ctx.chop_attempts >= 2:
        ctx.chops_done += 1
        ctx.chop_attempts = 0
        parts.append(f"chop_ok? done={ctx.chops_done}/{ctx.max_chops}")
        if ctx.chops_done >= ctx.max_chops:
            transition(ctx, State.DONE, "bc chop count")
    elif tree is not None:
        # reset streak if tree still there after long mine
        if not act.mine:
            pass

    if not parts:
        # idle — slow scan so we don't freeze
        if tree is None:
            safe_look(SEARCH_LOOK_DX, 0)
            parts.append(f"idle_scan(+{SEARCH_LOOK_DX})")
        else:
            parts.append("idle")

    # probs for debug
    if act.probs:
        top = sorted(act.probs.items(), key=lambda kv: -kv[1])[:4]
        pr = " ".join(f"{k[:4]}={v:.2f}" for k, v in top)
        parts.append(f"p[{pr}]")

    return "action=bc " + " ".join(parts)


def tick_once(
    ctx: AgentContext,
    model,
    region: dict,
    conf: float,
    bc=None,
) -> None:
    ctx.tick += 1
    frame = capture_frame(region)
    h, w = frame.shape[:2]
    # Black / frozen capture → useless
    if frame.mean() < 5:
        print(f"[{ctx.tick:04d}] WARN capture nearly black mean={frame.mean():.1f} region={region}")

    use_cv = USE_CV_FALLBACK
    if ctx.state is State.CHOPPING and not CV_FALLBACK_WHILE_CHOPPING:
        use_cv = False

    dets = detect_trees(
        frame, model=model, conf=conf, use_cv_fallback=use_cv
    )
    prefer_yolo = ctx.state in (State.APPROACHING, State.CHOPPING)
    tree = pick_control_tree(
        dets, w, h, last=ctx.last_tree, prefer_yolo=prefer_yolo
    )

    # Optional HUD (non-fatal if it fails — menus, darkness, etc.)
    hud = None
    hud_str = ""
    try:
        hud = parse_hud(frame, return_crops=False)
        if hud.ok:
            hud_str = f" HP={hud.health}/20 Food={hud.hunger}/20 slot={hud.selected_slot}"
    except Exception:
        pass

    if bc is not None and ctx.state is not State.DONE:
        action = step_bc(ctx, tree, hud, w, h, bc)
        # keep a coarse state label for logs
        if tree is None:
            ctx.state = State.SEARCHING
        elif tree.height_frac(h) >= CLOSE_ENOUGH_FRAC and is_centered(tree, w):
            ctx.state = State.CHOPPING
        else:
            ctx.state = State.APPROACHING
    elif ctx.state is State.SEARCHING:
        action = step_searching(ctx, tree)
    elif ctx.state is State.APPROACHING:
        action = step_approaching(ctx, tree, w, h)
    elif ctx.state is State.CHOPPING:
        action = step_chopping(ctx, tree, w, h)
    else:
        action = "action=idle"

    if tree is not None:
        off = (tree.cx - w * 0.5) / max(w, 1)
        tree_str = (
            f"tree={tree.cls_name}:{tree.conf:.2f}[{tree.source}] "
            f"cx={tree.cx:.0f}/{w} off={off:+.2f} h_frac={tree.height_frac(h):.2f}"
        )
    else:
        tree_str = "tree=None"
    print(
        f"[{ctx.tick:04d}] {ctx.state.name:12s} | dets={format_dets(dets)} | "
        f"{tree_str} | {action}{hud_str}"
    )


# =====================================================================
# Main loop
# =====================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minecraft tree-chop agent")
    p.add_argument("--model", type=Path, default=MODEL_PATH, help="path to YOLO best.pt")
    p.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="YOLO confidence")
    p.add_argument("--chops", type=int, default=DEFAULT_MAX_CHOPS, help="successful chops then exit")
    p.add_argument("--fps", type=float, default=TARGET_FPS, help="decision rate cap")
    p.add_argument("--countdown", type=float, default=STARTUP_COUNTDOWN_S)
    p.add_argument(
        "--bc",
        action="store_true",
        help="use behavioral-cloning policy (policy/bc_mlp.pt) instead of pure state machine",
    )
    p.add_argument("--policy", type=Path, default=POLICY_PATH, help="BC weights path")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    period = 1.0 / max(args.fps, 0.5)

    enable_dpi_awareness()
    model = load_yolo(args.model)

    bc = None
    if args.bc:
        from policy import BCAgent
        bc = BCAgent(weights=args.policy)
        print(f"[init] BC policy loaded from {args.policy}")
        print(f"[init] hold_table={bc.hold_table}")

    print(f"Click into the Minecraft game world now — {args.countdown:.0f}s")
    time.sleep(args.countdown)

    win = focus_minecraft()
    region = get_region(win, client_only=True)
    print(f"[init] Capture region: {region}")
    print(
        f"[init] conf={args.conf} cv_fallback={USE_CV_FALLBACK} "
        f"close_frac={CLOSE_ENOUGH_FRAC} fps={args.fps} max_chops={args.chops} "
        f"mode={'BC' if bc else 'FSM'}"
    )
    print("[init] Goal: SEARCH -> APPROACH -> CHOP tree. Ctrl+C to stop.")

    ctx = AgentContext(max_chops=max(1, args.chops))

    # apply learned default hold into FSM chop path too
    global CHOP_HOLD_S
    if bc is not None:
        CHOP_HOLD_S = float(min(max(bc.hold_table.get("default_hold_s", CHOP_HOLD_S), 0.2), 3.0))

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

                tick_once(ctx, model, region, args.conf, bc=bc)
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
