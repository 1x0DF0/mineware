"""
Train a small multi-label MLP policy on recorded sessions (behavioral cloning).

Usage:
    py train_policy.py
    py train_policy.py --epochs 80 --sessions sessions/20260810_174618

Writes:
    policy/bc_mlp.pt
    policy/bc_meta.json   (action names, thresholds, hold table)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from policy import BCPolicy  # canonical definition — do not redefine here
from policy_data import (
    ACTION_NAMES,
    FEATURE_DIM,
    build_dataset,
    extract_hold_table,
    iter_session_dirs,
    train_val_split,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "policy"


def pos_weight(Y: np.ndarray) -> torch.Tensor:
    """Handle class imbalance (mine is frequent, jump rare)."""
    n = len(Y)
    pos = Y.sum(axis=0).clip(min=1.0)
    neg = (n - pos).clip(min=1.0)
    w = neg / pos
    # cap so rare labels don't dominate
    w = np.clip(w, 0.25, 20.0)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def eval_metrics(model: BCPolicy, loader: DataLoader, device: str) -> dict:
    model.eval()
    all_p = []
    all_y = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits).cpu().numpy()
        all_p.append(prob)
        all_y.append(yb.numpy())
    P = np.concatenate(all_p, axis=0)
    Y = np.concatenate(all_y, axis=0)
    pred = (P >= 0.5).astype(np.float32)
    # per-action F1
    metrics = {}
    for i, name in enumerate(ACTION_NAMES):
        tp = ((pred[:, i] == 1) & (Y[:, i] == 1)).sum()
        fp = ((pred[:, i] == 1) & (Y[:, i] == 0)).sum()
        fn = ((pred[:, i] == 0) & (Y[:, i] == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        metrics[name] = {
            "f1": float(f1),
            "prec": float(prec),
            "rec": float(rec),
            "rate": float(Y[:, i].mean()),
        }
    acc = float((pred == Y).mean())
    metrics["hamming_acc"] = acc
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None, help="session dirs (default: all)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args(argv)

    if args.sessions:
        sessions = [Path(s) for s in args.sessions]
    else:
        sessions = iter_session_dirs()

    if not sessions:
        print("No sessions found. Record with: py record_session.py")
        return 1

    print(f"[train] sessions: {[s.name if s.is_dir() else s for s in sessions]}")
    X, Y, ids = build_dataset(sessions)
    Xtr, Ytr, Xva, Yva = train_val_split(X, Y, ids, val_frac=args.val_frac)
    print(f"[train] train={len(Xtr)} val={len(Xva)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    model = BCPolicy(FEATURE_DIM, len(ACTION_NAMES), hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pw = pos_weight(Ytr).to(device)
    print("[train] pos_weight:", {ACTION_NAMES[i]: round(float(pw[i]), 2) for i in range(len(ACTION_NAMES))})
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
        batch_size=args.batch,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
        batch_size=args.batch,
        shuffle=False,
    )

    best_f1 = -1.0
    best_state = None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        metrics = eval_metrics(model, val_loader, device)
        # mean F1 over actions that appear in val
        f1s = [m["f1"] for name, m in metrics.items() if name in ACTION_NAMES and m["rate"] > 0]
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            active = {
                k: round(v["f1"], 2)
                for k, v in metrics.items()
                if k in ACTION_NAMES and v["rate"] > 0.01
            }
            print(
                f"epoch {epoch:3d}/{args.epochs} loss={total/max(n,1):.4f} "
                f"val_hamming={metrics['hamming_acc']:.3f} mean_f1={mean_f1:.3f} {active}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    weights_path = OUT_DIR / "bc_mlp.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": FEATURE_DIM,
            "action_names": ACTION_NAMES,
            "hidden": args.hidden,
        },
        weights_path,
    )

    hold_table = extract_hold_table(sessions)
    # per-action thresholds: default 0.5, lower for rare but important if recall low
    thresholds = {name: 0.5 for name in ACTION_NAMES}
    # mine is common — keep 0.5; forward may need lower if sparse
    final_metrics = eval_metrics(model, val_loader, device)
    for name in ACTION_NAMES:
        m = final_metrics[name]
        if m["rate"] > 0 and m["rec"] < 0.3:
            thresholds[name] = 0.35

    meta = {
        "action_names": ACTION_NAMES,
        "feature_dim": FEATURE_DIM,
        "hidden": args.hidden,
        "thresholds": thresholds,
        "hold_table": hold_table,
        "val_metrics": final_metrics,
        "sessions": [str(s) for s in sessions],
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "best_mean_f1": best_f1,
    }
    meta_path = OUT_DIR / "bc_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[train] wrote {weights_path}")
    print(f"[train] wrote {meta_path}")
    print(f"[train] hold_table: {hold_table}")
    print("[train] val F1:", {k: round(v["f1"], 2) for k, v in final_metrics.items() if k in ACTION_NAMES})
    print("\nNext:  py agent.py --bc")
    return 0


if __name__ == "__main__":
    sys.exit(main())
