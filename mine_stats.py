"""
Mine-hold statistics from a recorded session (meta.jsonl).

Groups completed LMB holds by context available at release time:
  - best tree h_frac bucket
  - selected hotbar slot (proxy for tool until item ID exists)

Usage:
    py mine_stats.py --session sessions/20260811_143022
    py mine_stats.py --session sessions/20260811_143022 --csv holds.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_ticks(meta_path: Path) -> List[Dict[str, Any]]:
    ticks = []
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ticks.append(json.loads(line))
    return ticks


def h_frac_bucket(h: Optional[float]) -> str:
    if h is None:
        return "no_target"
    if h < 0.15:
        return "far"
    if h < 0.30:
        return "mid"
    if h < 0.45:
        return "near"
    return "close"


def extract_holds(ticks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Each meta line may contain mine.completed[] holds that ended that tick.
    Attach context from the same tick (slot, best_tree).
    """
    holds: List[Dict[str, Any]] = []
    for rec in ticks:
        mine = rec.get("mine") or {}
        completed = mine.get("completed") or []
        if not completed:
            continue
        hud = rec.get("hud") or {}
        slot = hud.get("selected_slot", -1) if hud.get("ok") else -1
        best = rec.get("best_tree") or {}
        h_frac = best.get("h_frac") if isinstance(best, dict) else None
        tree_cls = best.get("cls") if isinstance(best, dict) else None
        for ev in completed:
            hold_ms = int(ev.get("hold_ms", 0))
            if hold_ms < 30:
                # ignore micro-clicks
                continue
            holds.append(
                {
                    "tick": rec.get("tick"),
                    "t_rel": rec.get("t_rel"),
                    "hold_ms": hold_ms,
                    "selected_slot": slot,
                    "tree_cls": tree_cls,
                    "h_frac": h_frac,
                    "h_bucket": h_frac_bucket(h_frac),
                    "keys_down": list((rec.get("actions") or {}).get("keys_down") or []),
                }
            )
    return holds


def summarize(holds: List[Dict[str, Any]]) -> None:
    if not holds:
        print("No completed mine holds found (hold LMB while recording).")
        return

    ms = [h["hold_ms"] for h in holds]
    print(f"holds: {len(holds)}")
    print(
        f"hold_ms: min={min(ms)}  p50={statistics.median(ms):.0f}  "
        f"mean={statistics.mean(ms):.0f}  p90={_pct(ms, 90):.0f}  max={max(ms)}"
    )

    # by hotbar slot (tool proxy)
    by_slot: Dict[Any, List[int]] = defaultdict(list)
    for h in holds:
        by_slot[h["selected_slot"]].append(h["hold_ms"])
    print("\nBy selected_slot (use as tool proxy until item ID exists):")
    for slot in sorted(by_slot.keys(), key=lambda s: (s is None, s)):
        vals = by_slot[slot]
        print(
            f"  slot {slot}: n={len(vals)}  "
            f"p50={statistics.median(vals):.0f}ms  p90={_pct(vals, 90):.0f}ms"
        )

    # by distance bucket
    by_h: Dict[str, List[int]] = defaultdict(list)
    for h in holds:
        by_h[h["h_bucket"]].append(h["hold_ms"])
    print("\nBy best_tree h_frac bucket:")
    for bucket in ("no_target", "far", "mid", "near", "close"):
        if bucket not in by_h:
            continue
        vals = by_h[bucket]
        print(
            f"  {bucket:9s}: n={len(vals)}  "
            f"p50={statistics.median(vals):.0f}ms  p90={_pct(vals, 90):.0f}ms"
        )

    # suggested agent table
    print("\nSuggested agent hold defaults (p90, ms → s):")
    for slot in sorted(by_slot.keys(), key=lambda s: (s is None, s)):
        vals = by_slot[slot]
        p90 = _pct(vals, 90)
        print(f"  slot {slot}: hold_s ≈ {p90 / 1000.0:.2f}")


def _pct(vals: List[int], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return s[f] + (s[c] - s[f]) * (k - f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--session",
        type=Path,
        required=True,
        help="path to sessions/<id> directory",
    )
    ap.add_argument("--csv", type=Path, default=None, help="optional CSV dump of holds")
    args = ap.parse_args(argv)

    session = args.session
    if not session.is_dir():
        # allow bare id under sessions/
        alt = Path("sessions") / session
        if alt.is_dir():
            session = alt
        else:
            print(f"Session not found: {args.session}")
            return 1

    meta = session / "meta.jsonl"
    if not meta.is_file():
        print(f"Missing {meta}")
        return 1

    ticks = load_ticks(meta)
    print(f"session: {session.name}  ticks={len(ticks)}")
    info = session / "session.json"
    if info.is_file():
        sj = json.loads(info.read_text(encoding="utf-8"))
        print(f"duration_s={sj.get('duration_s')}  fps_target={sj.get('fps_target')}")

    holds = extract_holds(ticks)
    summarize(holds)

    if args.csv and holds:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(holds[0].keys()))
            w.writeheader()
            for h in holds:
                row = dict(h)
                row["keys_down"] = " ".join(row["keys_down"])
                w.writerow(row)
        print(f"\nwrote {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
