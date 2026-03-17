"""
EyeWave Detection Utilities
-----------------------------
  1. DetectionHead   — FCOS-style anchor-free head
  2. IoU functions   — IoU, GIoU, CIoU for loss and NMS
  3. EyeWaveLoss     — CIoU + BCE(conf) + CE(cls)
  4. nms             — batched NMS for post-processing
  5. decode_grid     — convert raw head output to absolute boxes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List


# ══════════════════════════════════════════════════════════════════════════════
# 1. Detection Head
# ══════════════════════════════════════════════════════════════════════════════

class DetectionHead(nn.Module):
    """
    Much deeper detection head for better features.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        out_ch = 5 + num_classes        

        # Multi-layer context mixer
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # Second stage to increase capacity
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        )
        self.pred = nn.Conv2d(in_channels, out_ch, 1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """Init confidence bias and box size bias."""
        # 1% occupancy prior
        nn.init.constant_(self.pred.bias[4], -4.595)   
        # Start boxes even larger. log(3.5) / 56 is ~0.6 normalized size
        nn.init.constant_(self.pred.bias[2], 3.5)  # tw
        nn.init.constant_(self.pred.bias[3], 3.5)  # th
        nn.init.normal_(self.pred.weight, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) feature map

        Returns:
            raw_grid: (B, H*W, 5 + num_classes) — raw logits / offsets
        """
        x = self.pre(x)
        x = self.pred(x)                              # (B, 5+C, H, W)
        B, _, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, -1)  # (B, H*W, 5+C)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# 2. IoU Utilities
# ══════════════════════════════════════════════════════════════════════════════

def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """[cx, cy, w, h] → [x1, y1, x2, y2]"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2,
                        cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """[x1, y1, x2, y2] → [cx, cy, w, h]"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2,
                        x2 - x1, y2 - y1], dim=-1)


def box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """
    Pairwise IoU between two sets of boxes in xyxy format.

    Args:
        boxes_a: (N, 4)
        boxes_b: (M, 4)

    Returns:
        iou: (N, M)
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    inter_x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-7)


def ciou_loss(pred_boxes: Tensor, target_boxes: Tensor) -> Tensor:
    """
    Complete IoU loss for a matched pair of boxes (both in cxcywh, normalised).

    CIoU = 1 - IoU + distance_penalty + aspect_ratio_penalty

    Args:
        pred_boxes:   (N, 4) [cx, cy, w, h] normalised 0..1
        target_boxes: (N, 4) [cx, cy, w, h] normalised 0..1

    Returns:
        loss: (N,)
    """
    p_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    t_xyxy = box_cxcywh_to_xyxy(target_boxes)

    # Intersection
    inter_x1 = torch.max(p_xyxy[:, 0], t_xyxy[:, 0])
    inter_y1 = torch.max(p_xyxy[:, 1], t_xyxy[:, 1])
    inter_x2 = torch.min(p_xyxy[:, 2], t_xyxy[:, 2])
    inter_y2 = torch.min(p_xyxy[:, 3], t_xyxy[:, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area_p = pred_boxes[:, 2]   * pred_boxes[:, 3]
    area_t = target_boxes[:, 2] * target_boxes[:, 3]
    union  = area_p + area_t - inter
    iou    = inter / (union + 1e-7)

    # Enclosing box for distance penalty
    enc_x1 = torch.min(p_xyxy[:, 0], t_xyxy[:, 0])
    enc_y1 = torch.min(p_xyxy[:, 1], t_xyxy[:, 1])
    enc_x2 = torch.max(p_xyxy[:, 2], t_xyxy[:, 2])
    enc_y2 = torch.max(p_xyxy[:, 3], t_xyxy[:, 3])
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

    # Centre distance penalty
    d2 = (pred_boxes[:, 0] - target_boxes[:, 0]) ** 2 + \
         (pred_boxes[:, 1] - target_boxes[:, 1]) ** 2

    # Aspect ratio consistency penalty (v)
    atan_t = torch.atan(target_boxes[:, 2] / (target_boxes[:, 3] + 1e-7))
    atan_p = torch.atan(pred_boxes[:, 2]   / (pred_boxes[:, 3]   + 1e-7))
    v      = (4 / (torch.pi ** 2)) * (atan_t - atan_p) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou = 1 - iou + d2 / c2 + alpha * v
    return ciou


# ══════════════════════════════════════════════════════════════════════════════
# 3. EyeWave Detection Loss
# ══════════════════════════════════════════════════════════════════════════════

class EyeWaveLoss(nn.Module):
    """
    Combined detection loss:
        L = λ_ciou · CIoU(pred_box, gt_box)    [positive cells only]
          + λ_conf · BCE(conf, obj_mask)        [all cells]
          + λ_cls  · CE(cls_logits, gt_cls)     [positive cells only]

    Target format (per image):
        targets: (N_gt, 6) — [batch_idx, cx, cy, w, h, class_id]
                              all coordinates normalised 0..1

    Grid assignment:
        A cell at (i, j) is positive if the IoU between the cell's
        implicit centre box and any GT box exceeds pos_iou_thresh.
        We use the GT box whose centre falls in that cell as primary
        (FCOS-style centre-point assignment).
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.lambda_ciou = cfg.train.lambda_ciou
        self.lambda_conf = cfg.train.lambda_conf
        self.lambda_cls  = cfg.train.lambda_cls
        self.lambda_attn = getattr(cfg.train, "lambda_attn", 5.0)
        self.pos_thresh  = cfg.train.pos_iou_thresh
        self.num_cls     = cfg.model.num_classes
        self.stride      = cfg.model.stride

    def compute_attention_loss(self, sal: Tensor, targets: Tensor, feat_hw: Tuple[int, int]) -> Tensor:
        """
        Force saliency map to be high ONLY at eye locations.
        Target is a Gaussian mask centered at each eye.
        """
        B, C, H, W = sal.shape
        device = sal.device
        
        # Create target mask (B, 1, H, W)
        mask = torch.zeros((B, 1, H, W), device=device)
        
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij"
        )
        
        for t in targets:
            bi = int(t[0].item())
            cx, cy = t[1].item(), t[2].item()
            
            # Target center in feature map coords
            tx, ty = cx * W, cy * H
            
            # Gaussian: exp(-((x-tx)^2 + (y-ty)^2) / (2 * sigma^2))
            # Sigma = 1.5 grid cells (approx 6 pixels at stride 4)
            sigma = 1.5
            dist_sq = (gx - tx)**2 + (gy - ty)**2
            gaussian = torch.exp(-dist_sq / (2 * sigma**2))
            
            mask[bi, 0] = torch.max(mask[bi, 0], gaussian)
            
        # Saliency maps are per-channel, so we average them or take max
        sal_mean = sal.mean(dim=1, keepdim=True)
        
        # MSE loss between predicted saliency and Gaussian target
        return F.mse_loss(sal_mean, mask)

    def forward(
        self,
        raw_grid: Tensor,      # (B, H*W, 5+num_cls)  raw from DetectionHead
        saliencies: List[Tensor], # [sal1, sal2]
        targets: Tensor,       # (N_gt, 6) [bidx, cx, cy, w, h, cls]
        input_hw: Tuple[int, int],
    ) -> Tuple[Tensor, dict]:
        """
        Returns:
            total_loss: scalar
            loss_dict:  {ciou, conf, cls, attn} for logging
        """
        device = raw_grid.device
        B      = raw_grid.shape[0]
        H_in, W_in = input_hw
        feat_h = H_in // self.stride
        feat_w = W_in // self.stride

        # ── Attention Loss ────────────────────────────────────────────────────
        loss_attn = torch.tensor(0.0, device=device)
        if saliencies and targets.numel() > 0:
            # Saliency maps have different resolutions:
            # sal1: H/2, W/2
            # sal2: H/4, W/4
            for i, sal in enumerate(saliencies):
                s_h, s_w = sal.shape[2], sal.shape[3]
                loss_attn += self.compute_attention_loss(sal, targets, (s_h, s_w))
            loss_attn /= len(saliencies)

        # ── Decode predictions ────────────────────────────────────────────────
        pred = decode_grid(raw_grid, input_hw, self.stride)
        # pred: (B, H*W, 5+num_cls)
        # [:,:,0:4] = normalised cxcywh  (after sigmoid/exp)
        # [:,:,4]   = conf (after sigmoid)
        # [:,:,5:]  = class logits (raw)

        # ── Build target tensors ──────────────────────────────────────────────
        obj_mask   = torch.zeros(B, feat_h * feat_w,          device=device)  # 1=positive
        tgt_box    = torch.zeros(B, feat_h * feat_w, 4,       device=device)  # gt cxcywh
        tgt_cls    = torch.zeros(B, feat_h * feat_w, dtype=torch.long, device=device)

        if targets.numel() > 0:
            for t in targets:
                bi   = int(t[0].item())
                cx, cy, w, h = t[1].item(), t[2].item(), t[3].item(), t[4].item()
                cls  = int(t[5].item())

                gi = int(cx * feat_w)
                gj = int(cy * feat_h)

                # Assign 3x3 neighborhood if center is within GT box to increase positive samples
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = gi + dx, gj + dy
                        if 0 <= nx < feat_w and 0 <= ny < feat_h:
                            cell_cx = (nx + 0.5) / feat_w
                            cell_cy = (ny + 0.5) / feat_h
                            if abs(cell_cx - cx) <= w/2 and abs(cell_cy - cy) <= h/2:
                                cell_idx = ny * feat_w + nx
                                obj_mask[bi, cell_idx] = 1.0
                                tgt_box[bi, cell_idx]  = torch.tensor([cx, cy, w, h], device=device)
                                tgt_cls[bi, cell_idx]  = cls

                # Guarantee at least the center cell is assigned even if box is tiny
                gi = min(gi, feat_w - 1)
                gj = min(gj, feat_h - 1)
                cell_idx = gj * feat_w + gi
                obj_mask[bi, cell_idx] = 1.0
                tgt_box[bi, cell_idx]  = torch.tensor([cx, cy, w, h], device=device)
                tgt_cls[bi, cell_idx]  = cls

        pos_mask = obj_mask.bool()   # (B, H*W)

        # ── Confidence loss (Focal Loss) ──────────────────────────────────────
        conf_pred = pred[..., 4]                            # (B, H*W) — already sigmoid
        
        # Focal Loss handles the extreme background (3000+) vs foreground (1-4) imbalance
        alpha = 0.25
        gamma = 2.0
        pt = torch.where(pos_mask, conf_pred, 1.0 - conf_pred)
        alpha_t = torch.where(pos_mask, alpha, 1.0 - alpha)
        
        bce = F.binary_cross_entropy(conf_pred, obj_mask, reduction="none")
        focal_loss = alpha_t * (1.0 - pt) ** gamma * bce
        
        # Normalize by number of positives, not total cells, to prevent loss vanishing
        num_pos = pos_mask.sum().item()
        normalizer = max(num_pos, 1)
        loss_conf = focal_loss.sum() / normalizer

        # ── Box + class loss (positive cells only) ────────────────────────────
        if num_pos > 0:
            pred_boxes_pos = pred[..., :4][pos_mask]        # (P, 4)
            gt_boxes_pos   = tgt_box[pos_mask]              # (P, 4)
            loss_ciou = ciou_loss(pred_boxes_pos, gt_boxes_pos).mean()

            cls_logits_pos = raw_grid[..., 5:][pos_mask]    # (P, num_cls) raw
            gt_cls_pos     = tgt_cls[pos_mask].long()       # (P,)

            # Guard: catch class IDs that are out of range before cross_entropy
            num_cls = cls_logits_pos.shape[-1]
            if gt_cls_pos.min() < 0 or gt_cls_pos.max() >= num_cls:
                bad = gt_cls_pos[(gt_cls_pos < 0) | (gt_cls_pos >= num_cls)].unique()
                raise ValueError(
                    f"Invalid class IDs in targets: {bad.tolist()}  "
                    f"(valid range: 0..{num_cls-1}). "
                    f"Check class_offset in get_loaders() — "
                    f"your labels may be 0-indexed already (use class_offset=0)."
                )
            loss_cls = F.cross_entropy(cls_logits_pos, gt_cls_pos)
        else:
            loss_ciou = torch.tensor(0.0, device=device, requires_grad=True)
            loss_cls  = torch.tensor(0.0, device=device, requires_grad=True)

        total = (self.lambda_ciou * loss_ciou
                 + self.lambda_conf * loss_conf
                 + self.lambda_cls  * loss_cls
                 + self.lambda_attn * loss_attn)

        return total, {
            "ciou": loss_ciou.item(),
            "conf": loss_conf.item(),
            "cls":  loss_cls.item(),
            "attn": loss_attn.item(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Grid Decode
# ══════════════════════════════════════════════════════════════════════════════

def decode_grid(
    raw_grid: Tensor,
    input_hw: Tuple[int, int],
    stride: int,
) -> Tensor:
    """
    Convert raw DetectionHead output to normalised [cx, cy, w, h, conf, cls*].

    tx, ty → sigmoid → offset within cell → normalised centre
    tw, th → exp      → normalised w, h  (clamped for stability)
    conf   → sigmoid
    cls    → unchanged (raw logits, softmax applied elsewhere)

    Args:
        raw_grid:  (B, H*W, 5+num_cls)
        input_hw:  (H_in, W_in) of the input image
        stride:    backbone stride (4 for EyeWave)

    Returns:
        decoded:   (B, H*W, 5+num_cls)  — cx,cy,w,h in [0,1]; conf in [0,1]
    """
    device = raw_grid.device
    H_in, W_in = input_hw
    feat_h = H_in // stride
    feat_w = W_in // stride

    # Build cell grid once
    gy, gx = torch.meshgrid(
        torch.arange(feat_h, device=device, dtype=torch.float32),
        torch.arange(feat_w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # (feat_h*feat_w, 2) — [gx, gy] for each cell
    grid_xy = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    decoded = raw_grid.clone()

    # cx, cy — sigmoid offset + cell index, normalised by feature map size
    decoded[..., 0] = (torch.sigmoid(raw_grid[..., 0]) + grid_xy[..., 0]) / feat_w
    decoded[..., 1] = (torch.sigmoid(raw_grid[..., 1]) + grid_xy[..., 1]) / feat_h

    # w, h — exp of raw, normalised by feature map size, clamped for stability
    decoded[..., 2] = torch.exp(raw_grid[..., 2].clamp(-4, 4)) / feat_w
    decoded[..., 3] = torch.exp(raw_grid[..., 3].clamp(-4, 4)) / feat_h

    # conf — sigmoid
    decoded[..., 4] = torch.sigmoid(raw_grid[..., 4])

    # cls — left as raw logits (softmax in inference / CE in loss)
    return decoded


# ══════════════════════════════════════════════════════════════════════════════
# 5. Non-Maximum Suppression
# ══════════════════════════════════════════════════════════════════════════════

def nms(
    decoded_grid: Tensor,          # (H*W, 5+num_cls) — single image, already decoded
    conf_thresh: float = 0.3,
    iou_thresh: float  = 0.45,
    max_detections: int = 2,
) -> Tensor:
    """
    Batched NMS for a single image.

    Steps:
        1. Filter cells below conf_thresh
        2. Sort by confidence descending
        3. Greedy IoU suppression
        4. Return top max_detections boxes

    Returns:
        detections: (D, 6) — [cx, cy, w, h, conf, class_id]
                    D ≤ max_detections
    """
    # Class score = conf × max_class_prob
    conf   = decoded_grid[:, 4]                          # (H*W,)
    cls_prob = torch.softmax(decoded_grid[:, 5:], dim=-1)
    max_cls_prob, cls_id = cls_prob.max(dim=-1)
    score = conf * max_cls_prob                          # joint confidence

    # Filter by threshold
    keep = score > conf_thresh
    if not keep.any():
        return torch.zeros(0, 6, device=decoded_grid.device)

    boxes_cxcywh = decoded_grid[keep, :4]
    scores       = score[keep]
    classes      = cls_id[keep].float()

    # Sort by score descending
    order = scores.argsort(descending=True)
    boxes_cxcywh = boxes_cxcywh[order]
    scores       = scores[order]
    classes      = classes[order]

    # Convert to xyxy for IoU computation
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)

    # Greedy suppression
    kept_indices: List[int] = []
    suppressed = torch.zeros(len(boxes_xyxy), dtype=torch.bool,
                             device=decoded_grid.device)

    for i in range(len(boxes_xyxy)):
        if suppressed[i]:
            continue
        kept_indices.append(i)
        if len(kept_indices) >= max_detections:
            break
        # Compute IoU with remaining boxes
        iou = box_iou(boxes_xyxy[i:i+1], boxes_xyxy[i+1:]).squeeze(0)
        suppress_mask = iou > iou_thresh
        suppressed[i+1:][suppress_mask] = True

    idx = torch.tensor(kept_indices, device=decoded_grid.device)
    detections = torch.stack([
        boxes_cxcywh[idx, 0],
        boxes_cxcywh[idx, 1],
        boxes_cxcywh[idx, 2],
        boxes_cxcywh[idx, 3],
        scores[idx],
        classes[idx],
    ], dim=-1)  # (D, 6)

    return detections