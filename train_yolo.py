"""
Fine-tune YOLOv8 on minecraft_dataset/.

Backends (auto-picked unless --device set):
  - cuda          NVIDIA only
  - directml / dml  AMD (and Intel) on Windows via torch-directml
  - cpu           all Ryzen cores (batch/cache tuned for big RAM)

Usage:
    py train_yolo.py --epochs 40 --name run2
    py train_yolo.py --device dml --batch 16 --cache
    py train_yolo.py --device cpu --batch 32 --workers 20 --cache
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML_PATH = ROOT / "minecraft_dataset" / "data.yaml"
BASE_MODEL = "yolov8n.pt"
EPOCHS = 40
IMAGE_SIZE = 640
PROJECT = "minecraft_yolo"
RUN_NAME = "run2"


def _cuda_ok() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _directml_device():
    """Return a torch device for AMD/Intel via DirectML, or None."""
    try:
        import torch_directml
        import torch

        d = torch_directml.device()
        # smoke: allocate a tiny tensor
        x = torch.zeros(1, device=d)
        _ = x + 1
        return d
    except Exception as e:
        print(f"[train] DirectML not usable: {e}")
        return None


def pick_device(requested: str | None):
    """
    Returns ultralytics-compatible device arg:
      int 0 for CUDA, 'cpu', or a torch.device for DirectML (str 'cpu' fallback if YOLO rejects DML).
    Also returns backend name for logging: 'cuda'|'directml'|'cpu'
    """
    req = (requested or "auto").lower().strip()

    if req in ("cpu",):
        return "cpu", "cpu"

    if req in ("0", "cuda", "gpu") or req.startswith("cuda"):
        if _cuda_ok():
            import torch
            print(f"[train] CUDA: {torch.cuda.get_device_name(0)}")
            return 0, "cuda"
        print("[train] CUDA requested but not available (normal on AMD).")
        if req != "auto":
            # fall through to try DML then CPU
            pass

    if req in ("dml", "directml", "amd") or req == "auto":
        dml = _directml_device()
        if dml is not None:
            print(f"[train] DirectML device: {dml}")
            return dml, "directml"
        if req in ("dml", "directml", "amd"):
            print("[train] DirectML requested but failed — falling back to CPU.")

    if req == "auto" and _cuda_ok():
        import torch
        print(f"[train] CUDA: {torch.cuda.get_device_name(0)}")
        return 0, "cuda"

    return "cpu", "cpu"


def cpu_thread_setup() -> int:
    """Use all logical cores for torch/OMP."""
    n = os.cpu_count() or 8
    # leave 2 cores free for OS/desktop
    threads = max(4, n - 2)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    try:
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(2, threads // 4))
    except Exception:
        pass
    print(f"[train] CPU threads={threads} (logical_cpus={n})")
    return threads


def suggest_batch(backend: str) -> int:
    # 128GB RAM machine: large CPU batches OK; DML/CUDA limited by VRAM
    if backend == "cpu":
        return 32
    if backend == "directml":
        # RX 9060 XT — WMI often under-reports VRAM; start moderate
        return 16
    # CUDA
    try:
        import torch
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if mem_gb >= 16:
            return 64
        if mem_gb >= 10:
            return 48
        if mem_gb >= 8:
            return 32
        return 16
    except Exception:
        return 16


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_YAML_PATH)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    ap.add_argument("--batch", type=int, default=0, help="0 = auto")
    ap.add_argument("--name", default=RUN_NAME)
    ap.add_argument(
        "--device",
        default="auto",
        help="auto | cpu | dml | directml | amd | 0 | cuda",
    )
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--cache", action="store_true", default=True)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)
    if args.no_cache:
        args.cache = False

    if not args.data.is_file():
        print(f"Missing {args.data}\nRun: py build_training_corpus.py")
        return 1

    ds_root = args.data.resolve().parent
    yaml_text = args.data.read_text(encoding="utf-8")
    lines = []
    for line in yaml_text.splitlines():
        if line.startswith("path:"):
            lines.append(f"path: {ds_root.as_posix()}")
        else:
            lines.append(line)
    args.data.write_text("\n".join(lines) + "\n", encoding="utf-8")

    device, backend = pick_device(args.device)
    if backend == "cpu":
        cpu_thread_setup()

    batch = args.batch if args.batch > 0 else suggest_batch(backend)
    if args.workers >= 0:
        workers = args.workers
    elif backend == "cpu":
        workers = min(16, max(4, (os.cpu_count() or 8) - 4))
    else:
        workers = 8

    print(f"[train] data={args.data}")
    print(
        f"[train] base={args.model} epochs={args.epochs} imgsz={args.imgsz} "
        f"batch={batch} backend={backend} device={device!r} workers={workers} cache={args.cache}"
    )

    model = YOLO(args.model)

    # Ultralytics: DirectML is experimental. Try DML device; on failure fall back to CPU.
    train_kwargs = dict(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device if backend != "directml" else "cpu",  # set below
        patience=25,
        project=str(ROOT / PROJECT),
        name=args.name,
        exist_ok=True,
        workers=workers,
        cache=args.cache,
        verbose=True,
        amp=(backend == "cuda"),
    )

    if backend == "directml":
        # Prefer letting ultralytics use CPU for the trainer loop is wrong —
        # try passing the DML torch device if supported; else maxed CPU.
        try:
            import torch_directml
            dml = torch_directml.device()
            print(f"[train] Attempting Ultralytics on DirectML ({dml})...")
            train_kwargs["device"] = dml
            train_kwargs["amp"] = False
            model.train(**train_kwargs)
        except Exception as e:
            print(f"[train] DirectML path failed ({e})")
            print("[train] Falling back to FULL CPU (Ryzen 24 threads + big batch + RAM cache)")
            cpu_thread_setup()
            train_kwargs["device"] = "cpu"
            train_kwargs["batch"] = max(batch, 32)
            train_kwargs["workers"] = min(16, max(4, (os.cpu_count() or 8) - 4))
            train_kwargs["amp"] = False
            train_kwargs["cache"] = True
            model.train(**train_kwargs)
    else:
        model.train(**train_kwargs)

    best = ROOT / PROJECT / args.name / "weights" / "best.pt"
    last = ROOT / PROJECT / args.name / "weights" / "last.pt"
    print(f"\n[train] Done.")
    print(f"  best -> {best}  exists={best.is_file()}")
    print(f"  last -> {last}  exists={last.is_file()}")
    if not best.is_file():
        return 1
    print("Next: py detect_live.py --conf 0.35")
    print("      py agent.py --conf 0.35 --chops 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
