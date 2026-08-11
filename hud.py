"""
Minecraft HUD parser — structured state from pixels.

No ML. Fixed-layout UI via color thresholding + geometry:
  health  : 0..20 (half-heart units)
  hunger  : 0..20 (half-shank units)
  hotbar  : 9 slots {empty, selected, occupied}
  selected: 0..8 index of highlighted hotbar slot

Auto-locates hearts/hunger each frame so window resizes / GUI scale changes
are tolerated without hard-coded absolute coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------- public types ----------

@dataclass
class HotbarSlot:
    index: int
    empty: bool
    selected: bool
    # occupancy score: higher => more likely an item icon is present
    occupancy: float
    # BGR crop of the slot interior (16x16 GUI units, screen-scaled)
    crop: Optional[np.ndarray] = None

    def to_dict(self, include_crop: bool = False) -> dict:
        d = {
            "index": self.index,
            "empty": self.empty,
            "selected": self.selected,
            "occupancy": round(self.occupancy, 2),
        }
        if include_crop and self.crop is not None:
            d["crop_shape"] = list(self.crop.shape)
        return d


@dataclass
class HudState:
    health: int              # 0..20
    hunger: int              # 0..20
    health_hearts: float     # 0..10 (e.g. 9.5)
    hunger_shanks: float     # 0..10
    hotbar: List[HotbarSlot]
    selected_slot: int       # 0..8, or -1 if unknown
    # geometry found this frame (screen px) — useful for debug overlays
    hearts_origin: Optional[Tuple[int, int]] = None
    hunger_origin: Optional[Tuple[int, int]] = None
    hotbar_origin: Optional[Tuple[int, int]] = None
    gui_scale: float = 0.0
    ok: bool = True
    error: str = ""

    def to_dict(self, include_crops: bool = False) -> dict:
        return {
            "health": self.health,
            "hunger": self.hunger,
            "health_hearts": self.health_hearts,
            "hunger_shanks": self.hunger_shanks,
            "selected_slot": self.selected_slot,
            "hotbar": [s.to_dict(include_crop=include_crops) for s in self.hotbar],
            "gui_scale": self.gui_scale,
            "hearts_origin": self.hearts_origin,
            "hunger_origin": self.hunger_origin,
            "hotbar_origin": self.hotbar_origin,
            "ok": self.ok,
            "error": self.error,
        }


# ---------- color masks ----------

def _red_heart_mask(bgr: np.ndarray, dark: bool = False) -> np.ndarray:
    """Pixels that look like filled heart (normal red health).

    Vanilla full-heart fill is ~RGB(255, 19, 19). Pass dark=True for a
    looser V floor (damage vignette / darkened frames).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v0 = 60 if dark else 140
    a = cv2.inRange(hsv, (0, 150, v0), (8, 255, 255))
    b = cv2.inRange(hsv, (172, 150, v0), (180, 255, 255))
    return a | b


def _hunger_mask(bgr: np.ndarray) -> np.ndarray:
    """Pixels that look like filled hunger shank (brown/orange flesh)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Shank meat is orange-brown. Exclude near-red (hue < 6) so heart pixels
    # do not register as hunger.
    return cv2.inRange(hsv, (6, 90, 70), (28, 255, 255))


def _restrict_to_hud_band(mask: np.ndarray) -> np.ndarray:
    """Keep only the lower-middle band where the survival HUD lives.

    Excludes the very bottom (held-item viewmodel) and side margins.
    """
    h, w = mask.shape[:2]
    out = mask.copy()
    out[: int(h * 0.78), :] = 0
    out[int(h * 0.96) :, :] = 0  # drop viewmodel / window border
    out[:, : int(w * 0.22)] = 0
    out[:, int(w * 0.78) :] = 0
    return out


# ---------- geometry helpers ----------

def _segments_1d(active: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous True runs → list of (start, end) inclusive."""
    segs: List[Tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, a in enumerate(active):
        if a and not in_seg:
            start = i
            in_seg = True
        elif not a and in_seg:
            segs.append((start, i - 1))
            in_seg = False
    if in_seg:
        segs.append((start, len(active) - 1))
    return segs


def _locate_icon_row(
    mask: np.ndarray,
    expected_count: int = 10,
    min_seg_width: int = 4,
) -> Optional[Tuple[int, int, int, List[Tuple[int, int]]]]:
    """
    Find a horizontal row of ~expected_count icon blobs.

    Strategy: peak row-sum in the mask → thin y-band around that peak →
    column segments. This ignores large red regions (viewmodel, sky tint)
    that would dominate a global bbox.

    Returns (y0, y1, pitch, segments_abs_x) or None.
    """
    if mask is None or mask.size == 0 or int(mask.sum()) < 20 * 255:
        return None

    row_sum = mask.sum(axis=1).astype(np.float64)
    peak_y = int(row_sum.argmax())
    if row_sum[peak_y] < 20 * 255:
        return None

    # Icon height is small; take the contiguous band around the peak that
    # stays above 15% of peak energy.
    thresh = 0.15 * row_sum[peak_y]
    y0 = peak_y
    while y0 > 0 and row_sum[y0 - 1] >= thresh:
        y0 -= 1
    y1 = peak_y
    h = mask.shape[0]
    while y1 + 1 < h and row_sum[y1 + 1] >= thresh:
        y1 += 1

    # Clamp ridiculous bands (shouldn't span more than ~30 px at normal scales)
    if y1 - y0 > 40:
        y0 = max(0, peak_y - 10)
        y1 = min(h - 1, peak_y + 10)

    band = mask[y0 : y1 + 1, :]
    col_sum = band.sum(axis=0)
    segs = _segments_1d(col_sum > 0)
    segs = [(a, b) for a, b in segs if (b - a + 1) >= min_seg_width]
    if not segs:
        return None

    # Drop thin outliers relative to the wider population (hearts ~14px,
    # hunger-bleed false positives ~6px). Skip this when a single fat blob
    # dominates — that case is handled by the splitter below.
    if len(segs) >= 3:
        widths = np.array([b - a + 1 for a, b in segs], dtype=np.float64)
        med_w = float(np.median(widths))
        p75_w = float(np.percentile(widths, 75))
        ref_w = max(med_w, p75_w)
        segs = [(a, b) for a, b in segs if (b - a + 1) >= 0.55 * ref_w]
        if not segs:
            return None

    # Darkened frames sometimes merge all 10 hearts into one fat blob
    # (gaps no longer drop to zero under a looser red threshold). Split any
    # segment that is clearly a multi-icon run at the vanilla 8-GUI pitch.
    split_segs: List[Tuple[int, int]] = []
    band_h = max(1, y1 - y0 + 1)
    approx_pitch = max(8, int(round(band_h * 8 / 9)))
    for a, b in segs:
        width = b - a + 1
        if width >= int(1.6 * approx_pitch):
            # Prefer splitting into expected_count when the span matches a
            # full icon row (dark frames merge all 10 hearts into one run).
            n_est = max(2, int(round(width / approx_pitch)))
            p_est = width / n_est
            p_exp = width / expected_count
            if (
                expected_count >= 3
                and abs(p_exp - approx_pitch) <= abs(p_est - approx_pitch) + 2
                and width >= expected_count * (approx_pitch * 0.7)
            ):
                n = expected_count
            else:
                n = n_est
            p = width / n
            for k in range(n):
                sa = int(round(a + k * p))
                sb = int(round(a + (k + 1) * p)) - 1
                sb = min(sb, b)
                if sb - sa + 1 >= min_seg_width:
                    split_segs.append((sa, sb))
        else:
            split_segs.append((a, b))
    segs = split_segs
    if len(segs) < 3:
        return None

    def _mass(seg_list):
        return sum(int(col_sum[a : b + 1].sum()) for a, b in seg_list)

    if len(segs) > expected_count:
        best_i, best_score = 0, -1.0
        for i in range(len(segs) - expected_count + 1):
            chunk = segs[i : i + expected_count]
            span = chunk[-1][1] - chunk[0][0] + 1
            score = _mass(chunk) / max(span, 1)
            if score > best_score:
                best_score = score
                best_i = i
        segs = segs[best_i : best_i + expected_count]

    abs_segs = [(a, b) for a, b in segs]

    starts = [a for a, _ in abs_segs]
    if len(starts) >= 2:
        pitches = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
        pitch = int(round(float(np.median(pitches))))
    else:
        pitch = abs_segs[0][1] - abs_segs[0][0] + 2

    if pitch < 4:
        return None

    return y0, y1, pitch, abs_segs


def _classify_fill(red_px: int, full_ref: float) -> float:
    """
    Map filled-pixel count → 0 / 0.5 / 1.0 icon units.
    full_ref is the expected pixel count for a completely full icon.
    """
    if full_ref <= 0:
        return 0.0
    ratio = red_px / full_ref
    if ratio >= 0.70:
        return 1.0
    if ratio >= 0.20:
        return 0.5
    return 0.0


# ---------- hotbar ----------

def _parse_hotbar(
    bgr: np.ndarray,
    hearts_x: int,
    hearts_y1: int,
    scale: float,
) -> Tuple[List[HotbarSlot], int, Tuple[int, int]]:
    """
    Vanilla layout (screen px, scale s = GUI scale):
      slot pitch     = 20 * s
      hotbar width   = 182 * s
      item icon      = 16 * s, inset ~2*s inside each slot
      selection box  = 24 * s (overflows slot by 2*s each side)

    Measured on 1509x1039 @ s=2: hearts red-fill x=574, hotbar slot
    dividers at 571+i*40, so hotbar_left ≈ hearts_x - 1.5*s.
    """
    h, w = bgr.shape[:2]
    s = float(scale)
    pitch = 20.0 * s

    # Slot-0 left edge sits just left of the first heart's red fill.
    hotbar_left = int(round(hearts_x - 1.5 * s))

    # Hotbar top: search below hearts for the gray frame row. Fallback ~10 GUI
    # below heart bottom (XP bar + padding lives in the gap).
    search_y0 = hearts_y1 + int(2 * s)
    search_y1 = min(h - int(6 * s), hearts_y1 + int(28 * s))
    hotbar_top = hearts_y1 + int(round(10 * s))
    best_gray, best_y = 0, hotbar_top
    hb_right = min(w, hotbar_left + int(182 * s))
    for y in range(search_y0, search_y1):
        row = bgr[y, hotbar_left:hb_right]
        if row.size == 0:
            break
        hsv = cv2.cvtColor(row.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        grayish = int(np.sum((hsv[:, 1] < 50) & (hsv[:, 2] > 40) & (hsv[:, 2] < 130)))
        if grayish > best_gray:
            best_gray = grayish
            best_y = y
    if best_gray > 80 * s:
        hotbar_top = best_y

    # Content crop: 16 GUI icon with 3 GUI inset from slot edges keeps the
    # 24-GUI selection highlight out of the occupancy sample.
    inset = 3.0 * s
    content = 14.0 * s  # slightly tighter than full 16 GUI icon

    slots: List[HotbarSlot] = []
    selected = -1
    selected_score = 0.0
    raw_occ: List[float] = []
    crops: List[np.ndarray] = []

    for i in range(9):
        slot_x0 = hotbar_left + i * pitch
        # Selection hit-box: 24 GUI, centered on the 20-GUI slot → -2 GUI origin
        ox0 = int(round(slot_x0 - 2 * s))
        oy0 = int(round(hotbar_top - 2 * s))
        ox1 = int(round(slot_x0 + 22 * s))
        oy1 = int(round(hotbar_top + 22 * s))
        ox0, oy0 = max(0, ox0), max(0, oy0)
        ox1, oy1 = min(w, ox1), min(h, oy1)

        ix0 = int(round(slot_x0 + inset))
        iy0 = int(round(hotbar_top + inset))
        ix1 = int(round(ix0 + content))
        iy1 = int(round(iy0 + content))
        ix0, iy0 = max(0, ix0), max(0, iy0)
        ix1, iy1 = min(w, max(ix0 + 1, ix1)), min(h, max(iy0 + 1, iy1))

        crop = bgr[iy0:iy1, ix0:ix1].copy()
        crops.append(crop)

        # Occupancy via std-dev of interior. World bleed is smooth at this
        # scale; item sprites add high local contrast.
        occ = float(crop.std()) if crop.size >= 10 else 0.0
        raw_occ.append(occ)

        # Selection: count near-white pixels on the outer ring only
        ring = bgr[oy0:oy1, ox0:ox1]
        white = 0
        if ring.size > 0:
            white_mask = (
                (ring[:, :, 0] > 230) & (ring[:, :, 1] > 230) & (ring[:, :, 2] > 230)
            )
            pad = max(1, int(round(3 * s)))
            if ring.shape[0] > 2 * pad and ring.shape[1] > 2 * pad:
                white_mask[pad:-pad, pad:-pad] = False
            white = int(np.sum(white_mask))

        if white > selected_score:
            selected_score = white
            selected = i

    # Adaptive empty threshold: empty slots cluster together (world bleed is
    # similar); items sit well above that cluster. Fallback absolute floor.
    sorted_occ = sorted(raw_occ)
    baseline = float(np.median(sorted_occ[:5]))  # majority-empty assumption
    thresh = max(40.0, baseline + 15.0)

    if selected_score < 8 * s:
        selected = -1

    for i in range(9):
        occ = raw_occ[i]
        # Selected empty slots can pick up white-border bleed → require more
        # evidence of an item when this slot is the highlight.
        local_thresh = thresh + (8.0 if i == selected else 0.0)
        empty = occ < local_thresh
        slots.append(
            HotbarSlot(
                index=i,
                empty=empty,
                selected=(i == selected),
                occupancy=occ,
                crop=crops[i],
            )
        )

    return slots, selected, (hotbar_left, hotbar_top)


# ---------- main entry ----------

def parse_hud(bgr: np.ndarray, return_crops: bool = True) -> HudState:
    """
    Parse survival HUD from a BGR frame (full Minecraft window capture is fine).

    Returns HudState. On failure, ok=False and error explains why; numeric
    fields default to 0 / empty hotbar.
    """
    empty_hotbar = [
        HotbarSlot(index=i, empty=True, selected=False, occupancy=0.0) for i in range(9)
    ]

    if bgr is None or bgr.size == 0:
        return HudState(
            health=0, hunger=0, health_hearts=0.0, hunger_shanks=0.0,
            hotbar=empty_hotbar, selected_slot=-1, ok=False, error="empty frame",
        )

    # --- health ---
    red = _restrict_to_hud_band(_red_heart_mask(bgr, dark=False))
    located = _locate_icon_row(red, expected_count=10, min_seg_width=4)
    if located is None:
        # Damage vignette / darkened frame: retry with looser V floor
        red = _restrict_to_hud_band(_red_heart_mask(bgr, dark=True))
        located = _locate_icon_row(red, expected_count=10, min_seg_width=4)
    if located is None:
        return HudState(
            health=0, hunger=0, health_hearts=0.0, hunger_shanks=0.0,
            hotbar=empty_hotbar, selected_slot=-1, ok=False,
            error="could not locate health hearts",
        )

    hy0, hy1, heart_pitch, heart_segs = located
    hearts_x = heart_segs[0][0]
    # GUI scale: vanilla heart placement pitch is 8 GUI pixels
    scale = heart_pitch / 8.0
    if scale < 0.9:
        return HudState(
            health=0, hunger=0, health_hearts=0.0, hunger_shanks=0.0,
            hotbar=empty_hotbar, selected_slot=-1, ok=False,
            error=f"implausible gui scale {scale:.2f} from heart pitch {heart_pitch}",
        )

    # Per-heart fill. Uniform grid from first-heart origin; more stable than
    # blob segs once a half-heart leaves a gap.
    icon_w = max(heart_pitch - 2, int(7 * scale))
    icon_h = max(hy1 - hy0 + 1, int(7 * scale))
    red_counts: List[int] = []
    for i in range(10):
        x0 = hearts_x + i * heart_pitch
        patch = bgr[hy0 : hy0 + icon_h, x0 : x0 + icon_w]
        if patch.size == 0:
            red_counts.append(0)
            continue
        # dark=True still matches bright hearts; covers vignette frames too
        red_counts.append(int(np.sum(_red_heart_mask(patch, dark=True) > 0)))

    full_ref = float(max(red_counts)) if max(red_counts) > 0 else 1.0
    # If badly hurt, max may be a half-heart — floor with scale-based estimate
    # (~132 px at s=2 → ~33 * s^2).
    est_full = 33.0 * scale * scale
    if full_ref < 0.6 * est_full:
        full_ref = est_full

    heart_vals = [_classify_fill(c, full_ref) for c in red_counts]
    health_hearts = float(sum(heart_vals))
    health = int(round(health_hearts * 2))

    # --- hunger ---
    # Same y-band as hearts, right half of hotbar. Vanilla: hunger row is
    # right-aligned to hotbar; left edge ≈ hearts_x + 80*scale + gap.
    hung = _hunger_mask(bgr)
    hung[: max(0, hy0 - 2), :] = 0
    hung[hy1 + 3 :, :] = 0
    hung[:, : hearts_x + 5 * heart_pitch] = 0
    # Also drop far-right (viewmodel / window chrome)
    hung[:, int(bgr.shape[1] * 0.82) :] = 0

    hung_located = _locate_icon_row(hung, expected_count=10, min_seg_width=3)
    # Reject hunger rows that sit on top of the hearts (dirt/world false positives
    # under a loose brown threshold). Real hunger starts ~right of XP bar.
    min_hunger_x = hearts_x + int(round(90 * scale))
    if hung_located is not None and hung_located[3][0][0] >= min_hunger_x:
        uy0, uy1, hung_pitch, hung_segs = hung_located
        hunger_x = hung_segs[0][0]
        if abs(hung_pitch - heart_pitch) <= 2:
            hung_pitch = heart_pitch
    else:
        # Synthetic grid: hearts left-aligned on hotbar, hunger right-aligned.
        # Measured: hunger_x ≈ hearts_x + 100*scale (80 hearts + ~20 XP gap).
        uy0, uy1 = hy0, hy1
        hung_pitch = heart_pitch
        hunger_x = hearts_x + int(round(100 * scale))

    hunger_origin = (hunger_x, uy0)
    hw = max(hung_pitch - 2, int(6 * scale))
    hh = max(uy1 - uy0 + 1, int(6 * scale))
    hung_counts: List[int] = []
    for i in range(10):
        x0 = hunger_x + i * hung_pitch
        patch = bgr[uy0 : uy0 + hh, x0 : x0 + hw]
        hung_counts.append(
            int(np.sum(_hunger_mask(patch) > 0)) if patch.size else 0
        )
    h_full = float(max(hung_counts)) if max(hung_counts) > 0 else 1.0
    est_h = 10.0 * scale * scale  # ~40 at s=2
    if h_full < 0.6 * est_h:
        h_full = est_h
    shank_vals = [_classify_fill(c, h_full) for c in hung_counts]
    hunger_shanks = float(sum(shank_vals))
    hunger = int(round(hunger_shanks * 2))

    # --- hotbar ---
    slots, selected, hotbar_origin = _parse_hotbar(bgr, hearts_x, hy1, scale)
    if not return_crops:
        for s in slots:
            s.crop = None

    return HudState(
        health=health,
        hunger=hunger,
        health_hearts=health_hearts,
        hunger_shanks=hunger_shanks,
        hotbar=slots,
        selected_slot=selected,
        hearts_origin=(hearts_x, hy0),
        hunger_origin=hunger_origin,
        hotbar_origin=hotbar_origin,
        gui_scale=scale,
        ok=True,
        error="",
    )


def draw_hud_overlay(bgr: np.ndarray, state: HudState) -> np.ndarray:
    """Annotate a copy of the frame with detected HUD geometry + values."""
    out = bgr.copy()
    if not state.ok:
        cv2.putText(
            out, f"HUD FAIL: {state.error}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
        )
        return out

    s = state.gui_scale
    if state.hearts_origin:
        x, y = state.hearts_origin
        for i in range(10):
            x0 = int(x + i * 8 * s)
            cv2.rectangle(out, (x0, y), (x0 + int(8 * s), y + int(9 * s)), (0, 255, 255), 1)
        cv2.putText(
            out, f"HP {state.health}/20", (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
        )

    if state.hunger_origin:
        x, y = state.hunger_origin
        for i in range(10):
            x0 = int(x + i * 8 * s)
            cv2.rectangle(out, (x0, y), (x0 + int(8 * s), y + int(9 * s)), (0, 165, 255), 1)
        cv2.putText(
            out, f"Food {state.hunger}/20", (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA,
        )

    if state.hotbar_origin:
        hx, hy = state.hotbar_origin
        for slot in state.hotbar:
            ix0 = int(round(hx + (3 + slot.index * 20) * s))
            iy0 = int(round(hy + 3 * s))
            ix1 = int(round(ix0 + 16 * s))
            iy1 = int(round(iy0 + 16 * s))
            color = (0, 255, 0) if slot.selected else ((255, 200, 0) if not slot.empty else (120, 120, 120))
            thickness = 2 if slot.selected else 1
            cv2.rectangle(out, (ix0, iy0), (ix1, iy1), color, thickness)
            label = f"{slot.index}"
            if not slot.empty:
                label += "*"
            cv2.putText(
                out, label, (ix0, iy0 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

    # Summary strip top-left of frame
    summary = (
        f"HP={state.health}/20  Food={state.hunger}/20  "
        f"Slot={state.selected_slot}  scale={state.gui_scale:.1f}"
    )
    cv2.rectangle(out, (8, 8), (8 + 9 * len(summary), 36), (0, 0, 0), -1)
    cv2.putText(
        out, summary, (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


# ---------- CLI ----------

def _cli():
    import argparse
    import json
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Parse Minecraft HUD from a screenshot")
    ap.add_argument("images", nargs="+", help="PNG/JPG frames to parse")
    ap.add_argument("--overlay-dir", default=None, help="If set, write annotated overlays here")
    ap.add_argument("--json", action="store_true", help="Print full JSON per frame")
    args = ap.parse_args()

    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    for path in args.images:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"{path}: FAILED to read", file=sys.stderr)
            continue
        state = parse_hud(bgr)
        if args.json:
            print(json.dumps({"file": path, **state.to_dict()}, indent=2))
        else:
            slots = "".join(
                ("[" if s.selected else " ")
                + ("#" if not s.empty else ".")
                + ("]" if s.selected else " ")
                for s in state.hotbar
            )
            status = "OK" if state.ok else f"FAIL({state.error})"
            print(
                f"{path}: {status}  HP={state.health:2d}/20  "
                f"Food={state.hunger:2d}/20  sel={state.selected_slot}  "
                f"hotbar={slots}  scale={state.gui_scale:.2f}"
            )
        if overlay_dir and state.ok:
            ov = draw_hud_overlay(bgr, state)
            out_path = overlay_dir / f"hud_{Path(path).stem}.png"
            cv2.imwrite(str(out_path), ov)


if __name__ == "__main__":
    _cli()
