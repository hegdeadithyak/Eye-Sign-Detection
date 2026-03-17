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
import random
import shutil
from pathlib import Path

from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────────────────────

def prepare_dataset(data_dir: str, val_split: float = 0.2, seed: int = 42):
    """
    Reorganise flat images/ + labels/ into train/val splits.

    YOLO expects:
        data_dir/
            images/train/  images/val/
            labels/train/  labels/val/

    This function is idempotent — skips if the split already exists.
    """
    data_path = Path(data_dir)
    img_dir = data_path / "images"
    lbl_dir = data_path / "labels"

    train_img = img_dir / "train"
    val_img = img_dir / "val"
    train_lbl = lbl_dir / "train"
    val_lbl = lbl_dir / "val"

    # Skip if already split
    if train_img.exists() and val_img.exists():
        n_train = len(list(train_img.glob("*")))
        n_val = len(list(val_img.glob("*")))
        if n_train > 0 and n_val > 0:
            print(f"[Dataset] Split already exists: {n_train} train, {n_val} val")
            return

    # Collect all images from the flat directory
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = sorted([
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ])

    if not all_images:
        raise FileNotFoundError(f"No images found in {img_dir}")

    print(f"[Dataset] Found {len(all_images)} images, splitting {1-val_split:.0%} / {val_split:.0%}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_images)

    n_val = max(1, int(len(all_images) * val_split))
    val_imgs = all_images[:n_val]
    train_imgs = all_images[n_val:]

    # Create directories
    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # Move files into splits
    def move_files(img_list, dst_img, dst_lbl):
        for img_path in img_list:
            shutil.move(str(img_path), str(dst_img / img_path.name))
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(dst_lbl / lbl_path.name))

    move_files(train_imgs, train_img, train_lbl)
    move_files(val_imgs, val_img, val_lbl)

    print(f"[Dataset] Split complete: {len(train_imgs)} train, {len(val_imgs)} val")


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(
    data_yaml: str = "dataset_subset.yaml",
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

    # Prepare train/val split (idempotent)
    data_path = Path(data_yaml)
    if data_path.exists():
        # Read the 'path' field from the YAML to find the data root
        import yaml
        with open(data_path) as f:
            cfg = yaml.safe_load(f)
        data_root = cfg.get("path", "dataset_subset")
        prepare_dataset(data_root)

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
        hsv_h=0.015,          # hue shift
        hsv_s=0.4,            # saturation shift
        hsv_v=0.4,            # value shift
        degrees=10.0,         # rotation ±10°
        translate=0.1,        # translation ±10%
        scale=0.5,            # scale ±50%
        flipud=0.0,           # no vertical flip (faces aren't upside down)
        fliplr=0.5,           # horizontal flip (left/right eye swap OK)
        mosaic=1.0,           # mosaic augmentation
        mixup=0.1,            # mixup probability
        copy_paste=0.1,       # copy-paste augmentation

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
    parser.add_argument("--data", default="dataset_subset.yaml",
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
