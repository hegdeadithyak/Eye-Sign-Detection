"""
EyeWave — Explainable AI Visualisation (Grad-CAM)
---------------------------------------------------
Supports TWO model backends:
  • eyewave  — custom EyeWave model (default)
  • yolo     — YOLOv11 (Ultralytics)

For every detected bbox, computes a Grad-CAM heatmap showing
exactly which image regions drove the model's prediction.

Each output tile shows:
  • Original image (greyscale, as seen by the model)
  • GT boxes in red
  • Predicted boxes with confidence
  • Grad-CAM heatmap blended over the image per detection
  • A per-detection strip showing class probability bars

Usage (from ~/eyewave):
    # EyeWave backend (default)
    python xai.py --data_dir dataset_subset --backend eyewave
    python xai.py --data_dir dataset_subset --conf 0.3 --n 6

    # YOLO backend
    python xai.py --data_dir dataset_subset --backend yolo --ckpt runs/detect/eyewave_yolo11/weights/best.pt
    python xai.py --data_dir dataset_subset --backend yolo --ckpt yolo11n.pt --data_yaml dataset_subset.yaml
"""

import argparse
import sys
import random
import math
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CFG, EyeWaveConfig
from model import build_model
from detection import decode_grid, nms, box_cxcywh_to_xyxy
from dataset import get_loaders


# ── Colour helpers ─────────────────────────────────────────────────────────────

PALETTE = [
    (  0, 220,  90),
    ( 30, 144, 255),
    (255, 165,   0),
    (220,  20,  60),
    (148,   0, 211),
    (  0, 206, 209),
    (255, 215,   0),
    (255,  20, 147),
]
GT_COLOR = (255, 60, 60)

def colour_for(cls_idx: int) -> tuple:
    return PALETTE[cls_idx % len(PALETTE)]

def load_font(size: int = 11):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Hooks into the target conv layer and computes a spatial heatmap
    showing which regions contributed most to a chosen output score.

    Works with any model that has named modules.
    Automatically finds the last Conv2d if no layer name is given.
    """

    def __init__(self, model: nn.Module, layer_name: Optional[str] = None):
        self.model      = model
        self.gradients  = None
        self.activations = None
        self._hooks     = []

        # Auto-find last Conv2d layer
        if layer_name is None:
            layer_name = self._find_last_conv(model)
        self.layer_name = layer_name
        print(f"[GradCAM] Hook layer: {layer_name}")

        # Register forward + backward hooks
        target = dict(model.named_modules())[layer_name]

        self._hooks.append(
            target.register_forward_hook(self._save_activations)
        )
        self._hooks.append(
            target.register_full_backward_hook(self._save_gradients)
        )

    @staticmethod
    def _find_last_conv(model: nn.Module) -> str:
        """
        Finds the last Conv2d layer that has spatial extent (kernel > 1).
        If none found, falls back to the absolute last Conv2d.
        """
        spatial_name = None
        last_name = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_name = name
                if module.kernel_size[0] > 1 or (len(module.kernel_size) > 1 and module.kernel_size[1] > 1):
                    spatial_name = name
        
        target = spatial_name if spatial_name else last_name
        if target is None:
            raise RuntimeError("No Conv2d layer found in model.")
        return target

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self,
                tensor: torch.Tensor,
                score: torch.Tensor) -> np.ndarray:
        """
        Args:
            tensor : (1, C, H, W) input with requires_grad=True
            score  : scalar tensor with grad_fn (objectness × cls_prob)
        Returns:
            cam : (H, W) float32 heatmap in [0, 1]
        """
        # Clear any stale gradients
        if tensor.grad is not None:
            tensor.grad.zero_()
        self.model.zero_grad()

        # Backward through score → triggers both hooks
        score.backward(retain_graph=False)

        grads = self.gradients          # (1, C, h, w)
        acts  = self.activations        # (1, C, h, w)

        # Global average pooling of gradients → channel weights
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        H, W = tensor.shape[2], tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


# ── Heatmap → coloured overlay ────────────────────────────────────────────────

def cam_to_heatmap_pil(cam: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    """Convert a [0,1] CAM array to a RGBA PIL heatmap (JET colourmap)."""
    cam_u8 = (cam * 255).astype(np.uint8)
    # Resize to display size
    cam_img = Image.fromarray(cam_u8).resize(size, Image.BILINEAR)
    cam_arr = np.array(cam_img, dtype=np.float32) / 255.0

    # JET colourmap (manual, no matplotlib dependency)
    r = np.clip(1.5 - np.abs(4 * cam_arr - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam_arr - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam_arr - 1), 0, 1)
    a = np.ones_like(r) * 0.55   # 55 % opacity

    rgba = (np.stack([r, g, b, a], axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def blend_cam(base_pil: Image.Image, cam: np.ndarray) -> Image.Image:
    """Blend a Grad-CAM heatmap over a greyscale PIL image."""
    base_rgba = base_pil.convert("RGBA")
    heat_rgba = cam_to_heatmap_pil(cam, base_pil.size)
    blended   = Image.alpha_composite(base_rgba, heat_rgba)
    return blended.convert("RGB")


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_box_pil(draw: ImageDraw.ImageDraw,
                 xyxy: tuple, label: str, color: tuple,
                 width: int = 2, font=None):
    x1, y1, x2, y2 = xyxy
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    th = 13
    try:
        if font:
            bb = font.getbbox(label)
            tw, th = bb[2] - bb[0] + 4, bb[3] - bb[1] + 4
    except Exception:
        tw = len(label) * 6 + 4

    ly1 = max(0, y1 - th - 1)
    draw.rectangle([x1, ly1, x1 + tw, y1], fill=color)
    draw.text((x1 + 2, ly1 + 1), label, fill=(255, 255, 255), font=font)


def tensor_to_pil_gray(img_tensor: torch.Tensor, hw: Tuple[int, int]) -> Image.Image:
    """
    Converts a (1, H, W) or (H, W) greyscale tensor to RGB PIL image.
    Handles both 1-channel (grayscale model) and 3-channel tensors.
    """
    t = img_tensor.cpu().clone().float()
    if t.dim() == 3 and t.shape[0] == 3:
        # 3-channel: convert to gray by averaging
        t = t.mean(dim=0)
    elif t.dim() == 3:
        t = t.squeeze(0)

    # Undo normalisation
    if t.min() < 0:
        t = t * 0.5 + 0.5      # was normalised with mean=0.5, std=0.5
    t = t.clamp(0, 1)

    arr = (t.numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="L").convert("RGB")
    return pil.resize((hw[1], hw[0]), Image.BILINEAR)


def make_cls_bar(class_names: list, cls_probs: np.ndarray,
                 pred_cls: int, width: int = 200, bar_h: int = 18) -> Image.Image:
    """
    Draws a horizontal bar chart of class probabilities.
    Highlights the predicted class.
    """
    n   = len(class_names)
    h   = n * bar_h + 8
    img = Image.new("RGB", (width, h), (18, 18, 18))
    drw = ImageDraw.Draw(img)
    font = load_font(9)

    for i, (name, prob) in enumerate(zip(class_names, cls_probs)):
        y   = 4 + i * bar_h
        bw  = int(prob * (width - 70))
        col = colour_for(i) if i == pred_cls else (80, 80, 80)
        drw.rectangle([60, y + 2, 60 + bw, y + bar_h - 2], fill=col)
        drw.text((2,  y + 2), f"{name[:8]:<8}", fill=(200, 200, 200), font=font)
        drw.text((60 + bw + 4, y + 2), f"{prob:.2f}", fill=(220, 220, 220), font=font)

    return img


# ── Core per-image XAI tile builder ───────────────────────────────────────────

def build_xai_tile(img_tensor: torch.Tensor,
                   gt: torch.Tensor,
                   dets: torch.Tensor,
                   cams: List[np.ndarray],
                   cls_probs_list: List[np.ndarray],
                   input_hw: Tuple[int, int],
                   class_names: List[str],
                   sample_idx: int,
                   font) -> Image.Image:
    """
    Builds a self-contained tile for one image:
        [base + boxes]  |  [heatmap blend]
        [per-det class bars …]
    """
    W_img, H_img = input_hw[1], input_hw[0]

    # ── 1. Base image ─────────────────────────────────────────────────────────
    base = tensor_to_pil_gray(img_tensor, input_hw)
    base_draw = ImageDraw.Draw(base)

    # GT boxes
    for g in range(gt.shape[0]):
        _, cx, cy, bw, bh, cls = gt[g].tolist()
        x1 = (cx - bw / 2) * W_img;  y1 = (cy - bh / 2) * H_img
        x2 = (cx + bw / 2) * W_img;  y2 = (cy + bh / 2) * H_img
        cn = class_names[int(cls)] if int(cls) < len(class_names) else f"c{int(cls)}"
        draw_box_pil(base_draw, (x1, y1, x2, y2),
                     f"GT:{cn}", GT_COLOR, width=2, font=font)

    # Pred boxes
    for d in range(dets.shape[0]):
        cx, cy, bw, bh, conf, cls = dets[d, :6].tolist()
        x1 = (cx - bw / 2) * W_img;  y1 = (cy - bh / 2) * H_img
        x2 = (cx + bw / 2) * W_img;  y2 = (cy + bh / 2) * H_img
        ci = int(cls)
        cn = class_names[ci] if ci < len(class_names) else f"c{ci}"
        draw_box_pil(base_draw, (x1, y1, x2, y2),
                     f"#{d+1} {cn} {conf:.2f}",
                     colour_for(ci), width=2, font=font)

    # ── 2. Heatmap blend (combined all detections) ────────────────────────────
    if cams:
        combined_cam = np.max(np.stack(cams, axis=0), axis=0)  # max over all dets
    else:
        combined_cam = np.zeros((H_img, W_img), dtype=np.float32)

    heat_img = blend_cam(base.copy(), combined_cam)

    # Label heatmap
    heat_draw = ImageDraw.Draw(heat_img)
    heat_draw.text((4, 4), "Grad-CAM", fill=(255, 255, 80), font=font)

    # ── 3. Side-by-side: [base | heatmap] ────────────────────────────────────
    top = Image.new("RGB", (W_img * 2 + 4, H_img), (30, 30, 30))
    top.paste(base,     (0,          0))
    top.paste(heat_img, (W_img + 4,  0))

    # ── 4. Per-detection class bar strips ─────────────────────────────────────
    BAR_W = W_img * 2 + 4
    BAR_H = 18 * len(class_names) + 10
    det_strips = []

    for d in range(min(dets.shape[0], 4)):   # max 4 strips
        cx, cy, bw, bh, conf, cls = dets[d, :6].tolist()
        ci  = int(cls)
        probs = cls_probs_list[d] if d < len(cls_probs_list) else np.zeros(len(class_names))

        bar = make_cls_bar(class_names, probs, ci, width=BAR_W, bar_h=18)

        # Header
        hdr = Image.new("RGB", (BAR_W, 16), (40, 40, 60))
        hdr_draw = ImageDraw.Draw(hdr)
        cn = class_names[ci] if ci < len(class_names) else f"c{ci}"
        hdr_draw.text((4, 2),
                      f"Det #{d+1}  →  {cn}  (conf {conf:.2f})  "
                      f"| Why? — regions shown in heatmap",
                      fill=colour_for(ci), font=font)
        strip = Image.new("RGB", (BAR_W, 16 + bar.height))
        strip.paste(hdr, (0, 0))
        strip.paste(bar, (0, 16))
        det_strips.append(strip)

    # ── 5. Assemble tile ──────────────────────────────────────────────────────
    title_h  = 20
    strip_h  = sum(s.height for s in det_strips)
    tile_h   = title_h + H_img + strip_h + 4

    tile = Image.new("RGB", (BAR_W, tile_h), (20, 20, 20))
    tile_draw = ImageDraw.Draw(tile)

    # Title bar
    tile_draw.rectangle([0, 0, BAR_W, title_h], fill=(35, 35, 55))
    tile_draw.text((6, 3),
                   f"Image #{sample_idx+1}   GT:{gt.shape[0]}   "
                   f"Pred:{dets.shape[0]}   "
                   f"[Left: predictions   Right: Grad-CAM attention]",
                   fill=(200, 200, 220), font=font)

    tile.paste(top, (0, title_h))

    y_off = title_h + H_img + 4
    for strip in det_strips:
        tile.paste(strip, (0, y_off))
        y_off += strip.height

    # Separator line
    tile_draw.line([(0, title_h + H_img + 2),
                    (BAR_W, title_h + H_img + 2)], fill=(60, 60, 80), width=2)

    return tile


# ── Main visualise function ────────────────────────────────────────────────────

def visualize_xai(data_dir: str,
                  ckpt_path: str,
                  n_images: int = 6,
                  conf_override: float = None,
                  class_names: list = None,
                  seed: int = 42,
                  output_path: str = "xai_viz.png",
                  layer_name: str = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[XAI] Device: {device}")

    # ── Load ──────────────────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: EyeWaveConfig = ckpt.get("cfg", CFG)

    if conf_override is not None:
        cfg.infer.conf_thresh_full = conf_override

    print(f"[XAI] Checkpoint: epoch={ckpt.get('epoch')}  "
          f"mAP={ckpt.get('map50', 0):.4f}  "
          f"conf={cfg.infer.conf_thresh_full:.3f}")

    model = build_model(cfg.model, cfg.infer).to(device)
    model.load_state_dict(ckpt["model_state"])
    # Keep in eval but allow grad for Grad-CAM
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    print(f"[XAI] Model: {model.count_parameters():,} params")

    if class_names is None:
        class_names = [f"cls_{i}" for i in range(cfg.model.num_classes)]

    input_hw = cfg.model.train_input_size

    # ── GradCAM hook ──────────────────────────────────────────────────────────
    gcam = GradCAM(model, layer_name)

    # ── Data ──────────────────────────────────────────────────────────────────
    _, val_loader = get_loaders(
        data_dir, cfg.model, cfg.train,
        class_offset=0, train_fraction=1.0,
    )

    all_imgs, all_tgts = [], []
    for imgs, tgts in val_loader:
        all_imgs.append(imgs)
        all_tgts.append(tgts)
        if sum(x.shape[0] for x in all_imgs) >= n_images * 4:
            break

    images  = torch.cat(all_imgs,  dim=0)
    targets = torch.cat(all_tgts,  dim=0)

    random.seed(seed)
    idxs = random.sample(range(images.shape[0]), min(n_images, images.shape[0]))
    print(f"[XAI] Processing {len(idxs)} images …")

    font   = load_font(10)
    tiles  = []
    n_cls  = cfg.model.num_classes

    for sample_idx, i in enumerate(idxs):
        img_tensor = images[i].to(device)
        inp        = img_tensor.unsqueeze(0)           # (1, C, H, W)
        inp.requires_grad_(False)

        # ── Forward (no_grad) to get detections ───────────────────────────────
        with torch.no_grad():
            raw_grid, _ = model(inp)
            decoded  = decode_grid(raw_grid, input_hw, cfg.model.stride)
            dets     = nms(
                decoded[0],
                conf_thresh=cfg.infer.conf_thresh_full,
                iou_thresh=cfg.infer.nms_iou_thresh,
                max_detections=cfg.model.max_detections,
            )

        gt_mask = (targets[:, 0] == i)
        gt      = targets[gt_mask].cpu()

        n_dets  = dets.shape[0]
        print(f"  Image {sample_idx+1}: GT={gt.shape[0]}  Pred={n_dets}")

        # ── Grad-CAM per detection ─────────────────────────────────────────────
        cams          = []
        cls_probs_list = []

        if n_dets > 0:
            for d in range(min(n_dets, 4)):      # compute CAM for up to 4 dets
                cls  = int(dets[d, 5].item())

                # Fresh input tensor WITH grad — must not clone from no-grad tensor
                inp_grad = img_tensor.unsqueeze(0).detach().clone()
                inp_grad.requires_grad_(True)

                # Forward WITH grad (outside no_grad context)
                raw_g, _ = model(inp_grad)             # (1, H*W, 5+cls)

                # Grid cell closest to this detection
                cx_n, cy_n = dets[d, 0].item(), dets[d, 1].item()
                H_g = int(math.sqrt(raw_g.shape[1]))
                W_g = H_g
                gx  = min(int(cx_n * W_g), W_g - 1)
                gy  = min(int(cy_n * H_g), H_g - 1)
                cell_idx = gy * W_g + gx

                # objectness score of that cell
                obj_score  = torch.sigmoid(raw_g[0, cell_idx, 4])

                # class probabilities (softmax over cls logits)
                cls_logits = raw_g[0, cell_idx, 5: 5 + n_cls]
                cls_probs  = torch.softmax(cls_logits, dim=0).detach().cpu().numpy()
                cls_probs_list.append(cls_probs)

                # Combined score: objectness × class confidence (has grad_fn)
                score = obj_score * torch.softmax(cls_logits, dim=0)[cls]

                cam = gcam.compute(inp_grad, score)
                cams.append(cam)
        else:
            # No detections — show raw objectness attention map
            inp_grad = img_tensor.unsqueeze(0).detach().clone()
            inp_grad.requires_grad_(True)
            raw_g, _ = model(inp_grad)
            obj_all  = torch.sigmoid(raw_g[0, :, 4]).mean()
            cam      = gcam.compute(inp_grad, obj_all)
            cams.append(cam)
            cls_probs_list = [np.ones(n_cls) / n_cls]

        # ── Build tile ────────────────────────────────────────────────────────
        tile = build_xai_tile(
            img_tensor.cpu(), gt, dets.cpu(),
            cams, cls_probs_list,
            input_hw, class_names, sample_idx, font,
        )
        tiles.append(tile)

    gcam.remove_hooks()

    # ── Stack tiles vertically ────────────────────────────────────────────────
    total_h = sum(t.height for t in tiles) + (len(tiles) - 1) * 8
    total_w = max(t.width  for t in tiles)
    canvas  = Image.new("RGB", (total_w, total_h + 36), (12, 12, 18))

    # Header banner
    hdr_draw = ImageDraw.Draw(canvas)
    hdr_draw.rectangle([0, 0, total_w, 36], fill=(20, 20, 40))
    hdr_draw.text((10, 10),
                  "EyeWave — Explainable AI  |  "
                  "Grad-CAM: warm colours = regions the model focused on  |  "
                  "Bars: per-class probability for each detection",
                  fill=(180, 210, 255), font=load_font(11))

    y = 36
    for tile in tiles:
        canvas.paste(tile, (0, y))
        y += tile.height + 8

    canvas.save(output_path)
    print(f"\n[XAI] Saved → {output_path}")
    print("      Warm (red/yellow) = high attention regions")
    print("      Cool (blue)       = low attention regions")
    print("      Bars show class probability distribution per detection")


# ── YOLO-safe Grad-CAM (no backward hooks) ────────────────────────────────────

class GradCAMYOLO:
    """
    Grad-CAM variant that avoids register_full_backward_hook entirely.

    YOLO's internal modules (C2f, DFL, etc.) use in-place operations that
    corrupt backward hooks. Instead, this class:
      1. Uses a forward hook to capture activations (WITH grad retained)
      2. Computes gradients via torch.autograd.grad() — no backward hook needed

    This is fully compatible with YOLO v8/v11 and any model with in-place ops.
    """

    def __init__(self, model: nn.Module, layer_name: Optional[str] = None):
        self.model = model
        self.activations = None
        self._hooks = []

        if layer_name is None:
            layer_name = GradCAM._find_last_conv(model)
        self.layer_name = layer_name
        print(f"[GradCAM-YOLO] Hook layer: {layer_name}")

        target = dict(model.named_modules())[layer_name]
        self._hooks.append(
            target.register_forward_hook(self._save_activations)
        )

    def _save_activations(self, module, input, output):
        # Keep activations IN the computation graph (retain_grad later)
        self.activations = output

    def compute(self,
                tensor: torch.Tensor,
                score: torch.Tensor) -> np.ndarray:
        """
        Args:
            tensor : (1, C, H, W) input
            score  : scalar tensor with grad_fn
        Returns:
            cam : (H, W) float32 heatmap in [0, 1]
        """
        acts = self.activations   # (1, C, h, w) — still in graph

        if acts is None:
            return np.zeros((tensor.shape[2], tensor.shape[3]), dtype=np.float32)

        # Compute gradients w.r.t. activations using autograd.grad
        # This avoids backward hooks entirely
        grads = torch.autograd.grad(
            outputs=score,
            inputs=acts,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]

        if grads is None:
            return np.zeros((tensor.shape[2], tensor.shape[3]), dtype=np.float32)

        # Global average pooling of gradients → channel weights
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * acts.detach()).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input resolution
        H, W = tensor.shape[2], tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


# ── YOLO backend XAI ──────────────────────────────────────────────────────────

def visualize_xai_yolo(
    data_dir: str,
    ckpt_path: str,
    n_images: int = 6,
    conf_override: float = None,
    class_names: list = None,
    seed: int = 42,
    output_path: str = "xai_viz_yolo.png",
    layer_name: str = None,
    data_yaml: str = "dataset_subset.yaml",
    imgsz: int = 640,
):
    """
    Grad-CAM visualisation using a YOLO (Ultralytics) model.
    Uses the same tile layout as the EyeWave backend.
    """
    from ultralytics import YOLO as UltralyticsYOLO
    from torchvision.transforms import functional as TF

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[XAI-YOLO] Device: {device}")

    # ── Load YOLO model ───────────────────────────────────────────────────────
    yolo_model = UltralyticsYOLO(ckpt_path)
    print(f"[XAI-YOLO] Model: {ckpt_path}")

    # Access the underlying torch model for Grad-CAM
    torch_model = yolo_model.model
    torch_model.to(device)
    torch_model.eval()
    for p in torch_model.parameters():
        p.requires_grad_(True)

    # Determine class names from model
    if class_names is None:
        if hasattr(yolo_model, "names") and yolo_model.names:
            class_names = [yolo_model.names[i] for i in sorted(yolo_model.names.keys())]
        else:
            class_names = [f"cls_{i}" for i in range(5)]
    n_cls = len(class_names)

    conf = conf_override if conf_override is not None else 0.25
    print(f"[XAI-YOLO] Classes: {class_names}  conf: {conf}")

    # ── GradCAM hook (YOLO-safe, no backward hooks) ──────────────────────────
    gcam = GradCAMYOLO(torch_model, layer_name)

    # ── Collect images from dataset ──────────────────────────────────────────
    img_dir = Path(data_dir) / "images"
    lbl_dir = Path(data_dir) / "labels"

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = sorted([
        p for p in img_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ])

    if not all_images:
        raise FileNotFoundError(f"No images found in {img_dir}")

    random.seed(seed)
    idxs = random.sample(range(len(all_images)), min(n_images, len(all_images)))
    print(f"[XAI-YOLO] Processing {len(idxs)} images …")

    font  = load_font(10)
    tiles = []

    for sample_idx, i in enumerate(idxs):
        img_path = all_images[i]
        pil_img  = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_img.size

        # Load GT labels (YOLO format: cls cx cy w h)
        lbl_stem = img_path.stem
        # Search in labels dir (handle train/val subdirs)
        lbl_path = None
        for candidate in [lbl_dir / f"{lbl_stem}.txt",
                          lbl_dir / "train" / f"{lbl_stem}.txt",
                          lbl_dir / "val" / f"{lbl_stem}.txt"]:
            if candidate.exists():
                lbl_path = candidate
                break

        gt_boxes = []
        if lbl_path and lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, w, h = map(float, parts)
                        # Format: batch_idx, cx, cy, w, h, cls (match EyeWave GT format)
                        gt_boxes.append([0, cx, cy, w, h, cls])
        gt = torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros(0, 6)

        # ── YOLO inference ─────────────────────────────────────────────────────
        results = yolo_model.predict(
            img_path, conf=conf, imgsz=imgsz, device=device, verbose=False,
        )
        result = results[0]

        # Convert results to [cx,cy,w,h,conf,cls] normalised format
        dets_list = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                c   = box.conf[0].item()
                cls = int(box.cls[0].item())
                cx  = ((x1 + x2) / 2) / orig_w
                cy  = ((y1 + y2) / 2) / orig_h
                bw  = (x2 - x1) / orig_w
                bh  = (y2 - y1) / orig_h
                dets_list.append([cx, cy, bw, bh, c, cls])
        dets = torch.tensor(dets_list, dtype=torch.float32) if dets_list else torch.zeros(0, 6)

        n_dets = dets.shape[0]
        print(f"  Image {sample_idx+1} ({img_path.name}): GT={gt.shape[0]}  Pred={n_dets}")

        # ── Prepare tensor for Grad-CAM ────────────────────────────────────────
        pil_resized = pil_img.resize((imgsz, imgsz))
        img_tensor  = TF.to_tensor(pil_resized)   # (3, H, W)

        # ── Grad-CAM per detection ─────────────────────────────────────────────
        cams           = []
        cls_probs_list = []

        if n_dets > 0:
            for d in range(min(n_dets, 4)):
                inp_grad = img_tensor.unsqueeze(0).to(device).detach().clone()
                inp_grad.requires_grad_(True)

                try:
                    out = torch_model(inp_grad)
                    if isinstance(out, (list, tuple)):
                        pred = out[0]
                    else:
                        pred = out

                    cls_idx = int(dets[d, 5].item())

                    if pred.dim() == 3:  # (1, num_preds, 5+nc)
                        obj_scores = pred[0, :, 4] if pred.shape[2] > 5 else pred[0, :, :4].sum(dim=1)
                        best_cell  = obj_scores.argmax()

                        if pred.shape[2] > 5 + cls_idx:
                            score = pred[0, best_cell, 4] * pred[0, best_cell, 5 + cls_idx]
                        else:
                            score = pred[0, best_cell, 4]

                        if pred.shape[2] > 5:
                            cls_logits = pred[0, best_cell, 5: 5 + n_cls]
                            cls_probs  = torch.softmax(cls_logits, dim=0).detach().cpu().numpy()
                        else:
                            cls_probs = np.ones(n_cls) / n_cls
                    else:
                        score = pred.sum()
                        cls_probs = np.ones(n_cls) / n_cls

                    cls_probs_list.append(cls_probs)
                    cam = gcam.compute(inp_grad, score)
                    cams.append(cam)

                except Exception as e:
                    print(f"    [Warn] Grad-CAM failed for det {d}: {e}")
                    cams.append(np.zeros((imgsz, imgsz), dtype=np.float32))
                    cls_probs_list.append(np.ones(n_cls) / n_cls)
        else:
            # No detections — show raw attention from the model
            inp_grad = img_tensor.unsqueeze(0).to(device).detach().clone()
            inp_grad.requires_grad_(True)
            try:
                out = torch_model(inp_grad)
                pred = out[0] if isinstance(out, (list, tuple)) else out
                score = pred.mean()
                cam = gcam.compute(inp_grad, score)
                cams.append(cam)
            except Exception as e:
                print(f"    [Warn] Grad-CAM failed: {e}")
                cams.append(np.zeros((imgsz, imgsz), dtype=np.float32))
            cls_probs_list = [np.ones(n_cls) / n_cls]

        # ── For display, convert to greyscale-like tensor ──────────────────────
        display_tensor = TF.to_tensor(pil_resized.convert("L")).squeeze(0)
        display_tensor_3d = display_tensor.unsqueeze(0)

        tile = build_xai_tile(
            display_tensor_3d, gt, dets,
            cams, cls_probs_list,
            (imgsz, imgsz), class_names, sample_idx, font,
        )
        tiles.append(tile)

    gcam.remove_hooks()

    # ── Stack tiles vertically ────────────────────────────────────────────────
    total_h = sum(t.height for t in tiles) + (len(tiles) - 1) * 8
    total_w = max(t.width  for t in tiles)
    canvas  = Image.new("RGB", (total_w, total_h + 36), (12, 12, 18))

    hdr_draw = ImageDraw.Draw(canvas)
    hdr_draw.rectangle([0, 0, total_w, 36], fill=(20, 20, 40))
    hdr_draw.text((10, 10),
                  "EyeWave XAI (YOLO backend)  |  "
                  "Grad-CAM: warm colours = model focus  |  "
                  "Bars: per-class probability",
                  fill=(180, 210, 255), font=load_font(11))

    y = 36
    for tile in tiles:
        canvas.paste(tile, (0, y))
        y += tile.height + 8

    canvas.save(output_path)
    print(f"\n[XAI-YOLO] Saved → {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math

    parser = argparse.ArgumentParser(
        description="EyeWave XAI — Grad-CAM visualisation (EyeWave or YOLO backend)"
    )
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--backend",     choices=["eyewave", "yolo"], default="eyewave",
                        help="Model backend: 'eyewave' (custom) or 'yolo' (Ultralytics)")
    parser.add_argument("--ckpt",        default=None,
                        help="Checkpoint path (default: checkpoints/best_gpu.pth for eyewave, "
                             "runs/detect/eyewave_yolo11/weights/best.pt for yolo)")
    parser.add_argument("--n",           type=int,   default=6,
                        help="Number of images (default 6)")
    parser.add_argument("--conf",        type=float, default=None,
                        help="Override conf threshold (e.g. 0.3)")
    parser.add_argument("--class_names", nargs="+",  default=None)
    parser.add_argument("--layer",       default=None,
                        help="Target conv layer name for Grad-CAM "
                             "(auto-selects last Conv2d if omitted)")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--out",         default=None,
                        help="Output path (default: xai_viz.png / xai_viz_yolo.png)")
    parser.add_argument("--data_yaml",   default="dataset_subset.yaml",
                        help="[YOLO only] Dataset YAML config")
    parser.add_argument("--imgsz",       type=int,   default=640,
                        help="[YOLO only] Input image size")
    args = parser.parse_args()

    if args.backend == "yolo":
        ckpt = args.ckpt or "runs/detect/eyewave_yolo11/weights/best.pt"
        out  = args.out  or "xai_viz_yolo.png"
        visualize_xai_yolo(
            data_dir      = args.data_dir,
            ckpt_path     = ckpt,
            n_images      = args.n,
            conf_override = args.conf,
            class_names   = args.class_names,
            seed          = args.seed,
            output_path   = out,
            layer_name    = args.layer,
            data_yaml     = args.data_yaml,
            imgsz         = args.imgsz,
        )
    else:
        ckpt = args.ckpt or "checkpoints/best_gpu.pth"
        out  = args.out  or "xai_viz.png"
        visualize_xai(
            data_dir      = args.data_dir,
            ckpt_path     = ckpt,
            n_images      = args.n,
            conf_override = args.conf,
            class_names   = args.class_names,
            seed          = args.seed,
            output_path   = out,
            layer_name    = args.layer,
        )
