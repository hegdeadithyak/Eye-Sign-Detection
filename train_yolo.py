"""
YOLOv11 Training Script for EyeWave dataset_subset.

Prepares the dataset into YOLO-expected train/val splits, then trains
a YOLOv11 model (Ultralytics) on the eye-signal detection task.

Usage:
    python train_yolo.py                          # defaults: yolo11n, 100 epochs
    python train_yolo.py --model yolo11s.pt       # use small model
    python train_yolo.py --epochs 50 --batch 16   # custom training
    python train_yolo.py --resume                 # resume from last checkpoint
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(
    data_yaml: str = "data.yaml",
    model_name: str = "yolo11n.pt",
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "",
    resume: bool = False,
    workers: int = 4,
    patience: int = 20,
    name: str = "eyewave_yolo11",
    project: str = "runs/detect",
):
    """Train YOLOv11 on the eye-signal detection dataset."""

    # Auto-detect device
    if not device:
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
        print(f"[Train] Using device: {'GPU' if device == '0' else 'CPU'}")

    # Load model
    if resume:
        # Resume from last checkpoint
        last_ckpt = Path(project) / name / "weights" / "last.pt"
        if last_ckpt.exists():
            model = YOLO(str(last_ckpt))
            print(f"[Train] Resuming from {last_ckpt}")
        else:
            print(f"[Train] No checkpoint found at {last_ckpt}, starting fresh")
            model = YOLO(model_name)
    else:
        model = YOLO(model_name)

    print(f"[Train] Model: {model_name}")
    print(f"[Train] Epochs: {epochs} | Batch: {batch} | ImgSz: {imgsz}")

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        patience=patience,
        project=project,
        name=name,
        exist_ok=True,  # overwrite existing run

        # ── Optimiser ────────────────────────────────────────
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,            # final LR = lr0 * lrf
        weight_decay=1e-4,
        warmup_epochs=3,
        warmup_bias_lr=0.1,

        # ── Augmentation ─────────────────────────────────────
        hsv_h=0.0,            # hue shift
        hsv_s=0.0,            # saturation shift
        hsv_v=0.0,            # value shift
        degrees=0.0,          # rotation
        translate=0.0,        # translation
        scale=0.0,            # scale
        flipud=0.0,           # no vertical flip
        fliplr=0.0,           # horizontal flip
        mosaic=0.0,           # mosaic augmentation
        mixup=0.0,            # mixup probability
        copy_paste=0.0,       # copy-paste augmentation

        # ── Loss weights ─────────────────────────────────────
        box=7.5,              # box loss weight
        cls=0.5,              # classification loss weight

        # ── Misc ─────────────────────────────────────────────
        save=True,
        save_period=10,       # save checkpoint every N epochs
        plots=True,           # generate training plots
        verbose=True,
    )

    # Validate best model
    print("\n" + "=" * 60)
    print("[Train] Validating best model...")
    print("=" * 60)

    best_ckpt = Path(project) / name / "weights" / "best.pt"
    if best_ckpt.exists():
        best_model = YOLO(str(best_ckpt))
        metrics = best_model.val(data=str(data_yaml), device=device)
        print(f"\n[Results] mAP@0.5:    {metrics.box.map50:.4f}")
        print(f"[Results] mAP@0.5:95: {metrics.box.map:.4f}")
        print(f"[Results] Precision:  {metrics.box.mp:.4f}")
        print(f"[Results] Recall:     {metrics.box.mr:.4f}")
    else:
        print("[Warning] No best.pt found — check training logs")

    return results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 on EyeWave dataset_subset"
    )
    parser.add_argument("--data", default="data.yaml",
                        help="Path to dataset YAML config")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="YOLO model variant (yolo11n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="",
                        help="Device: '', '0', 'cpu', '0,1', etc.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (0=disabled)")
    parser.add_argument("--name", default="eyewave_yolo11",
                        help="Run name")
    parser.add_argument("--project", default="runs/detect",
                        help="Project directory")

    args = parser.parse_args()

    train(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume,
        workers=args.workers,
        patience=args.patience,
        name=args.name,
        project=args.project,
    )
