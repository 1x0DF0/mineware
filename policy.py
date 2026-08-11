"""
Load and run the behavioral-cloning policy trained by train_policy.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from policy_data import ACTION_NAMES, FEATURE_DIM, tick_features

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "policy" / "bc_mlp.pt"
DEFAULT_META = ROOT / "policy" / "bc_meta.json"


class BCPolicy(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, n_actions: int = len(ACTION_NAMES), hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PolicyAction:
    forward: bool = False
    back: bool = False
    left: bool = False
    right: bool = False
    jump: bool = False
    sprint: bool = False
    mine: bool = False
    look_left: bool = False
    look_right: bool = False
    probs: Optional[Dict[str, float]] = None

    def look_dx(self, magnitude: int = 40) -> int:
        if self.look_left and not self.look_right:
            return -magnitude
        if self.look_right and not self.look_left:
            return magnitude
        return 0


class BCAgent:
    def __init__(
        self,
        weights: Path = DEFAULT_WEIGHTS,
        meta_path: Path = DEFAULT_META,
        device: Optional[str] = None,
    ):
        if not weights.is_file():
            raise FileNotFoundError(
                f"No policy weights at {weights}. Run: py train_policy.py"
            )
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        hidden = int(ckpt.get("hidden", 64))
        self.action_names = list(ckpt.get("action_names", ACTION_NAMES))
        self.model = BCPolicy(
            in_dim=int(ckpt.get("feature_dim", FEATURE_DIM)),
            n_actions=len(self.action_names),
            hidden=hidden,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.meta: Dict[str, Any] = {}
        if meta_path.is_file():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.thresholds = {
            name: float(self.meta.get("thresholds", {}).get(name, 0.5))
            for name in self.action_names
        }
        self.hold_table = self.meta.get("hold_table") or {"default_hold_s": 0.55, "by_slot_s": {}}

    def hold_s_for_slot(self, slot: int) -> float:
        by = self.hold_table.get("by_slot_s") or {}
        if str(slot) in by:
            return float(by[str(slot)])
        return float(self.hold_table.get("default_hold_s", 0.55))

    @torch.no_grad()
    def predict_from_features(self, feat: np.ndarray) -> PolicyAction:
        x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(self.device)
        prob = torch.sigmoid(self.model(x)).cpu().numpy()[0]
        probs = {self.action_names[i]: float(prob[i]) for i in range(len(self.action_names))}
        flags = {
            self.action_names[i]: bool(prob[i] >= self.thresholds[self.action_names[i]])
            for i in range(len(self.action_names))
        }
        return PolicyAction(
            forward=flags.get("forward", False),
            back=flags.get("back", False),
            left=flags.get("left", False),
            right=flags.get("right", False),
            jump=flags.get("jump", False),
            sprint=flags.get("sprint", False),
            mine=flags.get("mine", False),
            look_left=flags.get("look_left", False),
            look_right=flags.get("look_right", False),
            probs=probs,
        )

    def predict_from_record(self, rec: dict) -> PolicyAction:
        return self.predict_from_features(tick_features(rec))

    def predict_live(
        self,
        tree,
        hud,
        frame_w: int,
        frame_h: int,
    ) -> PolicyAction:
        """Build a synthetic meta-like dict from live perception."""
        best = None
        if tree is not None:
            best = {
                "conf": float(tree.conf),
                "h_frac": float(tree.height_frac(frame_h)),
                "off_x_frac": float((tree.cx - frame_w * 0.5) / max(frame_w, 1)),
            }
        hud_dict: Dict[str, Any] = {"ok": False}
        if hud is not None and getattr(hud, "ok", False):
            hud_dict = {
                "ok": True,
                "health": hud.health,
                "hunger": hud.hunger,
                "selected_slot": hud.selected_slot,
                "hotbar": [
                    {"empty": s.empty, "selected": s.selected, "index": s.index}
                    for s in hud.hotbar
                ],
            }
        rec = {"best_tree": best, "hud": hud_dict}
        return self.predict_from_features(tick_features(rec))
