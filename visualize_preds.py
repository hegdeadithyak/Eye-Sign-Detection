"""
    python visualize_preds.py --data_dir dataset_subset
    python visualize_preds.py --data_dir dataset_subset --n 10 --conf 0.05
    python visualize_preds.py --data_dir dataset_subset --class_names Normal Abnormal
"""

import argparse
import sys
import random
import math
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CFG, EyeWaveConfig
from model import build_model
from detection import decode_grid, nms, box_cxcywh_to_xyxy
from dataset import get_loaders



PALETTE = [
    (  0, 200,  80),   # green
    ( 30, 144, 255),   # dodger blue
    (255, 165,   0),   # orange
    (220,  20,  60),   # crimson
    (148,   0, 211),   # violet
    (  0, 206, 209),   # dark turquoise
    (255, 215,   0),   # gold
    (255,  20, 147),   # deep pink
]
GT_COLOR   = (255, 60,  60)   # always red for GT
TEXT_BG    = (0,   0,   0)    # black label background


def colour_for(cls_idx: int) -> tuple:
    return PALETTE[cls_idx % len(PALETTE)]


def draw_box(draw: ImageDraw.ImageDraw,
             xyxy: tuple,
             label: str,
             color: tuple,
             width: int = 2,
             font=None):
    x1, y1, x2, y2 = xyxy
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    # Label background
    tw, th = (len(label) * 6 + 4, 13)
    try:
        if font:
            bb = font.getbbox(label)
            tw, th = bb[2] - bb[0] + 4, bb[3] - bb[1] + 4
    except Exception:
        pass

    lx1, ly1 = x1, max(0, y1 - th - 1)
    lx2, ly2 = min(x1 + tw, draw.im.size[0]), y1
    draw.rectangle([lx1, ly1, lx2, ly2], fill=color)
    draw.text((lx1 + 2, ly1 + 1), label, fill=(255, 255, 255), font=font)


def tensor_to_pil(img_tensor: torch.Tensor, input_hw) -> Image.Image:
    """Convert a normalised CHW tensor back to a displayable PIL image."""
    t = img_tensor.cpu().clone()
    # Undo imagenet-style normalisation if values look normalised
    if t.min() < 0 or t.max() <= 1.5:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = t * std + mean
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((input_hw[1], input_hw[0]), Image.BILINEAR)


def make_grid(images: list, cols: int = 5, pad: int = 6) -> Image.Image:
    rows = math.ceil(len(images) / cols)
    w, h = images[0].size
    grid = Image.new("RGB",
                     (cols * (w + pad) + pad, rows * (h + pad) + pad),
                     color=(30, 30, 30))
    for idx, im in enumerate(images):
        r, c = divmod(idx, cols)
        grid.paste(im, (c * (w + pad) + pad, r * (h + pad) + pad))
    return grid


@torch.no_grad()
def visualize(data_dir: str,
              ckpt_path: str,
              n_images: int = 8,
              conf_override: float = None,
              class_names: list = None,
              seed: int = 42,
              output_path: str = "pred_viz.png",
              cols: int = 5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Viz] Device: {device}")

    #Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: EyeWaveConfig = ckpt.get("cfg", CFG)

    if conf_override is not None:
        cfg.infer.conf_thresh_full = conf_override
        print(f"[Viz] conf_thresh overridden → {conf_override}")

    print(f"[Viz] conf_thresh = {cfg.infer.conf_thresh_full}")
    print(f"[Viz] Checkpoint  : epoch={ckpt.get('epoch')}, "
          f"mAP={ckpt.get('map50', 0):.4f}")

    model = build_model(cfg.model, cfg.infer).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Viz] Model loaded — {model.count_parameters():,} params")

    input_hw = cfg.model.train_input_size   # (H, W)

    # Data
    _, val_loader = get_loaders(
        data_dir, cfg.model, cfg.train,
        class_offset=0, train_fraction=1.0,
    )

    # Collect enough samples
    all_images, all_targets = [], []
    for imgs, tgts in val_loader:
        all_images.append(imgs)
        all_targets.append(tgts)
        if sum(x.shape[0] for x in all_images) >= n_images * 3:
            break

    images  = torch.cat(all_images,  dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Random pick
    random.seed(seed)
    total = images.shape[0]
    idxs  = random.sample(range(total), min(n_images, total))
    print(f"[Viz] Sampling {len(idxs)} images from {total} val images")


    font = None
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            pass

    if class_names is None:
        class_names = [f"cls_{i}" for i in range(cfg.model.num_classes)]


    rendered = []
    stats    = {"total_gt": 0, "total_pred": 0, "images_with_pred": 0}

    for sample_idx, i in enumerate(idxs):
        img_tensor = images[i]
        pil_img    = tensor_to_pil(img_tensor, input_hw)

        # Ground truth boxes
        gt_mask = (targets[:, 0] == i)
        gt      = targets[gt_mask]          # (N, 6)  [batch_idx, cx, cy, w, h, cls]

        # Inference
        raw_grid = model(img_tensor.unsqueeze(0).to(device))
        decoded  = decode_grid(raw_grid, input_hw, cfg.model.stride)
        dets     = nms(
            decoded[0],
            conf_thresh=cfg.infer.conf_thresh_full,
            iou_thresh=cfg.infer.nms_iou_thresh,
            max_detections=cfg.model.max_detections,
        )                                   # (D, 6+)  [cx, cy, w, h, conf, cls]

        stats["total_gt"]   += gt.shape[0]
        stats["total_pred"] += dets.shape[0]
        if dets.shape[0] > 0:
            stats["images_with_pred"] += 1

        draw = ImageDraw.Draw(pil_img)
        W, H = pil_img.size

        for g in range(gt.shape[0]):
            _, cx, cy, w, h, cls = gt[g].tolist()
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W
            y2 = (cy + h / 2) * H
            cls_name = class_names[int(cls)] if int(cls) < len(class_names) else f"cls_{int(cls)}"
            draw_box(draw, (x1, y1, x2, y2),
                     f"GT:{cls_name}", GT_COLOR, width=2, font=font)

        for d in range(dets.shape[0]):
            cx, cy, w, h, conf, cls = dets[d, :6].tolist()
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W
            y2 = (cy + h / 2) * H
            cls_int  = int(cls)
            cls_name = class_names[cls_int] if cls_int < len(class_names) else f"cls_{cls_int}"
            color    = colour_for(cls_int)
            draw_box(draw, (x1, y1, x2, y2),
                     f"{cls_name} {conf:.2f}", color, width=2, font=font)

        # Caption bar at bottom
        caption = f"#{sample_idx+1}  GT:{gt.shape[0]}  Pred:{dets.shape[0]}"
        cap_h   = 16
        cap_img = Image.new("RGB", (W, cap_h), (20, 20, 20))
        cap_draw = ImageDraw.Draw(cap_img)
        cap_draw.text((4, 2), caption, fill=(200, 200, 200), font=font)
        combined = Image.new("RGB", (W, H + cap_h))
        combined.paste(pil_img, (0, 0))
        combined.paste(cap_img, (0, H))

        rendered.append(combined)

    #Grid & legend
    actual_cols = min(cols, len(rendered))
    grid = make_grid(rendered, cols=actual_cols)

    # Legend strip
    leg_h  = 28
    legend = Image.new("RGB", (grid.width, leg_h), (15, 15, 15))
    leg_d  = ImageDraw.Draw(legend)
    leg_d.rectangle([4, 6, 24, 22], outline=GT_COLOR, width=2)
    leg_d.text((28, 8), "Ground Truth", fill=GT_COLOR, font=font)
    leg_d.rectangle([140, 6, 160, 22], outline=PALETTE[0], width=2)
    leg_d.text((164, 8), "Prediction", fill=PALETTE[0], font=font)

    final = Image.new("RGB", (grid.width, grid.height + leg_h))
    final.paste(legend, (0, 0))
    final.paste(grid,   (0, leg_h))

    final.save(output_path)
    print(f"\n[Viz] Saved → {output_path}")

    #Console summary
    print("\n" + "=" * 50)
    print("  VISUALISATION SUMMARY")
    print("=" * 50)
    print(f"  Images shown      : {len(rendered)}")
    print(f"  Total GT boxes    : {stats['total_gt']}")
    print(f"  Total predictions : {stats['total_pred']}")
    print(f"  Images with preds : {stats['images_with_pred']} / {len(rendered)}")
    avg_pred = stats["total_pred"] / max(len(rendered), 1)
    print(f"  Avg preds/image   : {avg_pred:.1f}")
    print(f"  conf_thresh used  : {cfg.infer.conf_thresh_full}")
    print("=" * 50)

    if stats["total_pred"] == 0:
        print("\n  ⚠️  No predictions visible at this threshold.")
        print("  Try:  --conf 0.01   (or even 0.001)")

    return final


#CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     required=True)
    parser.add_argument("--ckpt",         default="checkpoints/best_gpu.pth")
    parser.add_argument("--n",            type=int,   default=8,
                        help="Number of images to visualise (default 8)")
    parser.add_argument("--cols",         type=int,   default=4,
                        help="Grid columns (default 4)")
    parser.add_argument("--conf",         type=float, default=None,
                        help="Override NMS conf threshold (e.g. 0.05)")
    parser.add_argument("--class_names",  nargs="+",  default=None,
                        help="Class names in index order")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--out",          default="pred_viz.png",
                        help="Output PNG path (default pred_viz.png)")
    args = parser.parse_args()

    visualize(
        data_dir=args.data_dir,
        ckpt_path=args.ckpt,
        n_images=args.n,
        conf_override=args.conf,
        class_names=args.class_names,
        seed=args.seed,
        output_path=args.out,
        cols=args.cols,
    )
