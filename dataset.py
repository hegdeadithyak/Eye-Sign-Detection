import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps

from config import ModelConfig, TrainConfig


# ─────────────────────────────────────────────────────────────
# Label loader
# ─────────────────────────────────────────────────────────────

def load_labels(label_path: str, class_offset: int = 0) -> Tensor:
    if not os.path.exists(label_path):
        return torch.zeros(0, 5)

    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, cx, cy, w, h = map(float, parts)
            cls = int(cls) - class_offset

            if cls < 0:
                continue

            boxes.append([cls, cx, cy, w, h])

    if not boxes:
        return torch.zeros(0, 5)

    return torch.tensor(boxes, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# Partial face crop
# ─────────────────────────────────────────────────────────────

def partial_face_crop(
    image: Image.Image,
    labels: Tensor,
    crop_scale=(0.4, 0.7),
):
    W, H = image.size

    for _ in range(10):

        scale_h = random.uniform(*crop_scale)
        scale_w = random.uniform(0.6, 1.0)

        crop_h = int(H * scale_h)
        crop_w = int(W * scale_w)

        y1 = random.randint(0, max(0, int(H * 0.15)))
        x1 = random.randint(0, max(0, W - crop_w))

        x2 = x1 + crop_w
        y2 = y1 + crop_h

        if labels.numel() == 0:
            break

        cx = labels[:, 1] * W
        cy = labels[:, 2] * H
        w  = labels[:, 3] * W
        h  = labels[:, 4] * H

        bx1 = cx - w/2
        by1 = cy - h/2
        bx2 = cx + w/2
        by2 = cy + h/2

        inside = (bx1 >= x1) & (by1 >= y1) & (bx2 <= x2) & (by2 <= y2)

        if not inside.any():
            continue

        cropped = image.crop((x1, y1, x2, y2))

        new_labels = []
        for i, ok in enumerate(inside):
            if not ok:
                continue

            new_cx = (cx[i] - x1) / crop_w
            new_cy = (cy[i] - y1) / crop_h
            new_w  = w[i] / crop_w
            new_h  = h[i] / crop_h

            new_labels.append([
                labels[i,0].item(),
                new_cx.item(),
                new_cy.item(),
                new_w.item(),
                new_h.item()
            ])

        if new_labels:
            return cropped, torch.tensor(new_labels)

    return image, labels


# ─────────────────────────────────────────────────────────────
# Eye-region crop (crop to GT bounding box)
# ─────────────────────────────────────────────────────────────

def eye_bbox_crop(
    image: Image.Image,
    labels: Tensor,
    padding: float = 0.3,
) -> Tuple[Image.Image, Tensor]:
    """
    Crop image to the union of all GT bounding boxes with padding.
    Forces the model to see ONLY the eye region — no nose, cheeks, etc.

    Args:
        image:   PIL image
        labels:  (N, 5) tensor [cls, cx, cy, w, h] normalised 0..1
        padding: fractional expansion around the union bbox (0.3 = 30%)

    Returns:
        cropped_image, remapped_labels
    """
    if labels.numel() == 0:
        return image, labels

    W, H = image.size

    # ── Compute union of all GT bboxes in pixel coords ─────────────────
    cx = labels[:, 1] * W
    cy = labels[:, 2] * H
    bw = labels[:, 3] * W
    bh = labels[:, 4] * H

    bx1 = (cx - bw / 2).min().item()
    by1 = (cy - bh / 2).min().item()
    bx2 = (cx + bw / 2).max().item()
    by2 = (cy + bh / 2).max().item()

    union_w = bx2 - bx1
    union_h = by2 - by1

    # ── Expand by padding fraction ─────────────────────────────────────
    pad_x = union_w * padding
    pad_y = union_h * padding

    crop_x1 = max(0, int(bx1 - pad_x))
    crop_y1 = max(0, int(by1 - pad_y))
    crop_x2 = min(W, int(bx2 + pad_x))
    crop_y2 = min(H, int(by2 + pad_y))

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    if crop_w < 1 or crop_h < 1:
        return image, labels

    # ── Crop ───────────────────────────────────────────────────────────
    cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # ── Remap labels to cropped coordinate space ───────────────────────
    new_labels = []
    for i in range(labels.shape[0]):
        new_cx = (cx[i].item() - crop_x1) / crop_w
        new_cy = (cy[i].item() - crop_y1) / crop_h
        new_w  = bw[i].item() / crop_w
        new_h  = bh[i].item() / crop_h

        # Clamp to [0, 1]
        new_cx = max(0.0, min(1.0, new_cx))
        new_cy = max(0.0, min(1.0, new_cy))
        new_w  = max(0.001, min(1.0, new_w))
        new_h  = max(0.001, min(1.0, new_h))

        new_labels.append([
            labels[i, 0].item(),
            new_cx, new_cy, new_w, new_h
        ])

    return cropped, torch.tensor(new_labels, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# Top-60% crop (eye-region focus)
# ─────────────────────────────────────────────────────────────

def top_crop(
    image: Image.Image,
    labels: Tensor,
    keep_ratio: float = 0.6,
) -> Tuple[Image.Image, Tensor]:
    """
    Keep only the top `keep_ratio` fraction of image rows.
    Designed for face images where eyes are in the upper portion.

    Guarantees:
        - Zero null bboxes: only fully-contained boxes are kept.
        - Zero false bboxes: coordinates are strictly validated after remap.
        - If no valid boxes survive the crop, returns original image unchanged.

    Args:
        image:      PIL image
        labels:     (N, 5) tensor [cls, cx, cy, w, h] normalised 0..1
        keep_ratio: fraction of rows to keep from the top (default 0.6)

    Returns:
        cropped_image, remapped_labels
    """
    W, H = image.size
    crop_h = int(H * keep_ratio)

    if crop_h < 1 or labels.numel() == 0:
        return image, labels

    # ── Convert normalised coords to pixel coords ──────────────────────
    cx = labels[:, 1] * W
    cy = labels[:, 2] * H
    bw = labels[:, 3] * W
    bh = labels[:, 4] * H

    # Bbox edges in pixel space
    by1 = cy - bh / 2   # top edge
    by2 = cy + bh / 2   # bottom edge

    # ── Strict filtering: keep ONLY boxes fully inside the crop ────────
    # A box is valid iff its ENTIRE vertical extent is within [0, crop_h]
    fully_inside = (by1 >= 0) & (by2 <= crop_h)

    if not fully_inside.any():
        # No valid boxes survive → return original image unchanged
        return image, labels

    # ── Crop the image (top crop_h rows, full width) ───────────────────
    cropped = image.crop((0, 0, W, crop_h))

    # ── Remap surviving boxes to the cropped coordinate space ──────────
    new_labels = []
    for i in range(labels.shape[0]):
        if not fully_inside[i]:
            continue

        # Renormalise to cropped dimensions (width unchanged, height = crop_h)
        new_cx = cx[i].item() / W            # x unchanged (full width kept)
        new_cy = cy[i].item() / crop_h       # y remapped to crop height
        new_w  = bw[i].item() / W
        new_h  = bh[i].item() / crop_h

        # ── Final validation: reject any degenerate box ────────────────
        if (new_w <= 0 or new_h <= 0 or
            new_cx - new_w / 2 < -0.01 or new_cx + new_w / 2 > 1.01 or
            new_cy - new_h / 2 < -0.01 or new_cy + new_h / 2 > 1.01):
            continue

        # Clamp to strict [0, 1] to eliminate floating point drift
        new_cx = max(0.0, min(1.0, new_cx))
        new_cy = max(0.0, min(1.0, new_cy))
        new_w  = max(0.001, min(1.0, new_w))
        new_h  = max(0.001, min(1.0, new_h))

        new_labels.append([
            labels[i, 0].item(),  # class unchanged
            new_cx, new_cy, new_w, new_h
        ])

    if not new_labels:
        # Safety: if all boxes were somehow rejected, return original
        return image, labels

    return cropped, torch.tensor(new_labels, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class EyeDataset(Dataset):

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        img_paths: List[Path],
        label_dir: str,
        model_cfg: ModelConfig,
        augment=False,
        partial_crop_prob=0.0,
        class_offset=0,
        top_crop_ratio=0.0,
        use_eye_crop=False,
        eye_crop_padding=0.3,
    ):

        self.img_paths = img_paths
        self.label_dir = Path(label_dir)
        self.model_cfg = model_cfg

        self.augment = augment
        self.partial_crop_prob = partial_crop_prob
        self.class_offset = class_offset
        self.top_crop_ratio = top_crop_ratio
        self.use_eye_crop = use_eye_crop
        self.eye_crop_padding = eye_crop_padding

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        img_path = self.img_paths[idx]
        
        # Search for the label file (handling YOLO train/val splits)
        lbl_stem = img_path.stem + ".txt"
        label_path = self.label_dir / lbl_stem
        if not label_path.exists():
            for candidate in [self.label_dir / "train" / lbl_stem, 
                              self.label_dir / "val" / lbl_stem]:
                if candidate.exists():
                    label_path = candidate
                    break

        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        labels = load_labels(str(label_path), self.class_offset)

        # Eye-region crop: crop to GT bbox so model only sees eyes
        if self.use_eye_crop:
            image, labels = eye_bbox_crop(image, labels, self.eye_crop_padding)
        else:
            # Legacy crops (only used when eye crop is disabled)
            if self.top_crop_ratio > 0:
                image, labels = top_crop(image, labels, self.top_crop_ratio)
            if self.augment and random.random() < self.partial_crop_prob:
                image, labels = partial_face_crop(image, labels)

        target_size = self.model_cfg.train_input_size
        image = image.resize((target_size[1], target_size[0]))

        if self.model_cfg.input_channels == 1:
            image = TF.to_grayscale(image, num_output_channels=1)

        img_tensor = TF.to_tensor(image)

        img_tensor = TF.normalize(
            img_tensor,
            mean=[0.5]*self.model_cfg.input_channels,
            std=[0.5]*self.model_cfg.input_channels
        )

        return img_tensor, labels


# ─────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────

def collate_fn(batch):

    images, labels = zip(*batch)
    images = torch.stack(images)

    targets = []

    for i, lbl in enumerate(labels):

        if lbl.numel() == 0:
            continue

        batch_idx = torch.full((lbl.shape[0],1), i)

        entry = torch.cat([
            batch_idx,
            lbl[:,1:5],
            lbl[:,0:1]
        ], dim=1)

        targets.append(entry)

    if targets:
        targets = torch.cat(targets)
    else:
        targets = torch.zeros(0,6)

    return images, targets


# ─────────────────────────────────────────────────────────────
# Loader factory
# ─────────────────────────────────────────────────────────────

def get_loaders(
    data_dir: str,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_fraction=1.0,
    val_split=0.2,
    class_offset=0,  
    ):
    img_dir = Path(data_dir) / "images"
    label_dir = Path(data_dir) / "labels"

    all_images = sorted([
        p for p in img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in EyeDataset.IMG_EXTS
    ])

    total = len(all_images)

    # apply fraction BEFORE split
    if train_fraction < 1.0:
        keep = max(1, int(total * train_fraction))
        all_images = all_images[:keep]
        print(f"[Dataset] using {keep}/{total} images")

    random.seed(42)
    random.shuffle(all_images)

    n = len(all_images)
    n_val = max(1, int(n * val_split))

    val_imgs = all_images[:n_val]
    train_imgs = all_images[n_val:]

    train_ds = EyeDataset(
        train_imgs,
        label_dir,
        model_cfg,
        augment=True,
        partial_crop_prob=train_cfg.partial_crop_prob,
        class_offset=class_offset,
        top_crop_ratio=train_cfg.top_crop_ratio,
        use_eye_crop=train_cfg.use_eye_crop,
        eye_crop_padding=train_cfg.eye_crop_padding,
    )

    val_ds = EyeDataset(
        val_imgs,
        label_dir,
        model_cfg,
        augment=False,
        class_offset=class_offset,
        top_crop_ratio=train_cfg.top_crop_ratio,
        use_eye_crop=train_cfg.use_eye_crop,
        eye_crop_padding=train_cfg.eye_crop_padding,
    )

    print(f"[Dataset] train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size*2,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader
