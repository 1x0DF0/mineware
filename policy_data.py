"""
Build behavioral-cloning datasets from sessions/*/meta.jsonl.

Features are structured (tree + HUD) — no image encoder required for v1.
Labels are multi-hot actions from the human's keys/mouse that tick.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
SESSIONS_DIR = ROOT / "sessions"

# Action heads (multi-label binary)
ACTION_NAMES = [
    "forward",  # w
    "back",     # s
    "left",     # a
    "right",    # d
    "jump",     # space
    "sprint",   # shift
    "mine",     # lmb
    "look_left",
    "look_right",
]

LOOK_DX_THRESH = 12  # |mouse dx| above this ⇒ look left/right label


def iter_session_dirs(sessions_root: Path = SESSIONS_DIR) -> List[Path]:
    if not sessions_root.is_dir():
        return []
    return sorted(
        p for p in sessions_root.iterdir()
        if p.is_dir() and (p / "meta.jsonl").is_file()
    )


def load_ticks(session: Path) -> List[Dict[str, Any]]:
    ticks = []
    with (session / "meta.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ticks.append(json.loads(line))
    return ticks


def tick_features(rec: Dict[str, Any]) -> np.ndarray:
    """
    Fixed-length float32 vector from one meta.jsonl record.

    Layout:
      [0] tree_present
      [1] tree_conf
      [2] tree_h_frac
      [3] tree_off_x_frac
      [4] health_n
      [5] hunger_n
      [6] selected_slot_n
      [7:16] hotbar empty flags (1=empty)
    """
    best = rec.get("best_tree") or {}
    if not isinstance(best, dict) or "conf" not in best:
        tree = [0.0, 0.0, 0.0, 0.0]
    else:
        tree = [
            1.0,
            float(best.get("conf") or 0.0),
            float(best.get("h_frac") or 0.0),
            float(best.get("off_x_frac") or 0.0),
        ]

    hud = rec.get("hud") or {}
    if hud.get("ok"):
        health = float(hud.get("health") or 0) / 20.0
        hunger = float(hud.get("hunger") or 0) / 20.0
        slot = float(hud.get("selected_slot") if hud.get("selected_slot") is not None else 0)
        if slot < 0:
            slot = 0.0
        slot_n = slot / 8.0
        empties = []
        for i in range(9):
            hb = (hud.get("hotbar") or [])
            empty = 1.0
            if i < len(hb):
                empty = 1.0 if hb[i].get("empty", True) else 0.0
            empties.append(empty)
    else:
        health = hunger = slot_n = 0.0
        empties = [1.0] * 9

    return np.asarray(tree + [health, hunger, slot_n] + empties, dtype=np.float32)


FEATURE_DIM = 16  # keep in sync with tick_features
assert FEATURE_DIM == 4 + 3 + 9


def tick_labels(rec: Dict[str, Any]) -> np.ndarray:
    keys = set((rec.get("actions") or {}).get("keys_down") or [])
    mouse = (rec.get("actions") or {}).get("mouse") or {}
    mine = rec.get("mine") or {}
    dx = int(mouse.get("dx") or 0)

    y = np.zeros(len(ACTION_NAMES), dtype=np.float32)
    y[0] = 1.0 if "w" in keys else 0.0
    y[1] = 1.0 if "s" in keys else 0.0
    y[2] = 1.0 if "a" in keys else 0.0
    y[3] = 1.0 if "d" in keys else 0.0
    y[4] = 1.0 if "space" in keys else 0.0
    y[5] = 1.0 if "shift" in keys else 0.0
    y[6] = 1.0 if mine.get("lmb_held") or mouse.get("lmb") else 0.0
    y[7] = 1.0 if dx < -LOOK_DX_THRESH else 0.0
    y[8] = 1.0 if dx > LOOK_DX_THRESH else 0.0
    return y


def build_dataset(
    sessions: Optional[Sequence[Path]] = None,
    min_ticks: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns X [N,F], Y [N,A], session_ids per row (for split).
    """
    if sessions is None:
        sessions = iter_session_dirs()
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ids: List[str] = []
    for sess in sessions:
        ticks = load_ticks(sess)
        if len(ticks) < min_ticks:
            print(f"[data] skip {sess.name}: only {len(ticks)} ticks")
            continue
        n = 0
        for rec in ticks:
            xs.append(tick_features(rec))
            ys.append(tick_labels(rec))
            ids.append(sess.name)
            n += 1
        print(f"[data] {sess.name}: {n} ticks")
    if not xs:
        raise SystemExit(
            f"No session data under {SESSIONS_DIR}. Run record_session.py first."
        )
    X = np.stack(xs, axis=0)
    Y = np.stack(ys, axis=0)
    print(f"[data] total N={len(X)} F={X.shape[1]} A={Y.shape[1]}")
    print("[data] label rates:", {ACTION_NAMES[i]: float(Y[:, i].mean()) for i in range(Y.shape[1])})
    return X, Y, ids


def train_val_split(
    X: np.ndarray,
    Y: np.ndarray,
    ids: List[str],
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = max(1, int(len(X) * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]


def extract_hold_table(
    sessions: Optional[Sequence[Path]] = None,
    min_hold_ms: int = 200,
    max_hold_ms: int = 8000,
) -> Dict[str, Any]:
    """
    Median hold_ms by selected_slot for real chops.
    Drops micro-clicks (<200ms) and AFK/menu holds (>8s).
    Floor default at 0.55s so fist chops still work with sparse data.
    """
    if sessions is None:
        sessions = iter_session_dirs()
    by_slot: Dict[str, List[int]] = {}
    all_ms: List[int] = []
    for sess in sessions:
        for rec in load_ticks(sess):
            hud = rec.get("hud") or {}
            slot = hud.get("selected_slot", -1) if hud.get("ok") else -1
            for ev in (rec.get("mine") or {}).get("completed") or []:
                ms = int(ev.get("hold_ms") or 0)
                if ms < min_hold_ms or ms > max_hold_ms:
                    continue
                all_ms.append(ms)
                key = str(slot)
                by_slot.setdefault(key, []).append(ms)

    def med(vals: List[int]) -> float:
        if not vals:
            return 800.0
        s = sorted(vals)
        mid = len(s) // 2
        if len(s) % 2:
            return float(s[mid])
        return 0.5 * (s[mid - 1] + s[mid])

    def clamp_s(ms: float) -> float:
        return round(min(max(ms / 1000.0, 0.4), 2.5), 3)

    default = clamp_s(med(all_ms)) if all_ms else 0.8
    table = {
        "default_hold_s": default,
        "by_slot_s": {
            k: clamp_s(med(v)) for k, v in sorted(by_slot.items())
        },
        "n_holds": len(all_ms),
        "min_hold_ms_filter": min_hold_ms,
        "max_hold_ms_filter": max_hold_ms,
    }
    return table
