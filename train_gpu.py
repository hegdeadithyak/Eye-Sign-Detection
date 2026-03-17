import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import CFG, EyeWaveConfig
from model import build_model
from detection import EyeWaveLoss, decode_grid, nms, box_iou, box_cxcywh_to_xyxy
from dataset import get_loaders



@torch.no_grad() 
def compute_map50(
    model: nn.Module,
    loader,
    cfg: EyeWaveConfig,
    device: torch.device,
    iou_thresh: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Steps performed:
        1. Decode model outputs into bounding boxes and class predictions
        2. For each image, match predicted boxes to GT boxes using IoU
        3. Count true positives, false positives, and GT boxes for mAP@0
        4. Calculate classification accuracy on correctly localized detections

    
    Returns:
        map50:    mean Average Precision at IoU=0.5
        cls_acc:  accuracy on correctly localised detections
        max_iou:  highest IoU seen in the entire val set (for debugging)
    """
    model.eval()
    input_hw = cfg.model.train_input_size

    all_tp, all_fp, all_gt = 0, 0, 0
    cls_correct, cls_total = 0, 0
    max_iou = 0.0

    for images, targets in loader:
        images  = images.to(device)
        targets = targets.to(device)
        B = images.shape[0] 

        raw_grid, _ = model(images)
        decoded  = decode_grid(raw_grid, input_hw, cfg.model.stride)

        for i in range(B):
            gt_mask = (targets[:, 0] == i) 
            gt      = targets[gt_mask]
            n_gt    = gt.shape[0]
            all_gt += n_gt

            if n_gt > 0:
                # [cx, cy, w, h, conf, cls...]
                pred_boxes_raw = decoded[i, :, :4]
                gt_boxes_raw   = gt[:, 1:5]
                iou_raw = box_iou(box_cxcywh_to_xyxy(pred_boxes_raw), 
                                 box_cxcywh_to_xyxy(gt_boxes_raw))
                max_iou = max(max_iou, iou_raw.max().item())

            dets = nms(
                decoded[i],
                conf_thresh=cfg.infer.conf_thresh_full,
                iou_thresh=cfg.infer.nms_iou_thresh,
                max_detections=cfg.model.max_detections,
            )

            if dets.shape[0] == 0 or n_gt == 0:
                all_fp += dets.shape[0]
                continue

            pred_xyxy = box_cxcywh_to_xyxy(dets[:, :4])
            gt_xyxy   = box_cxcywh_to_xyxy(gt[:, 1:5])
            iou_mat   = box_iou(pred_xyxy, gt_xyxy)
            
            matched_gt = set()
            for d in range(dets.shape[0]):
                best_iou, best_g = iou_mat[d].max(dim=0)
                best_g_idx = best_g.item()
                if best_iou >= iou_thresh and best_g_idx not in matched_gt:
                    all_tp    += 1
                    matched_gt.add(best_g_idx)
                    pred_cls = int(dets[d, 5].item())
                    gt_cls   = int(gt[best_g_idx, 5].item())
                    cls_total   += 1
                    cls_correct += int(pred_cls == gt_cls)
                else:
                    all_fp += 1

    precision = all_tp / (all_tp + all_fp + 1e-7)
    recall    = all_tp / (all_gt + 1e-7)
    map50     = (2 * precision * recall) / (precision + recall + 1e-7)
    cls_acc   = cls_correct / (cls_total + 1e-7)
    return map50, cls_acc, max_iou




def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: EyeWaveLoss,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    input_hw: Tuple[int, int],
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    totals = {"total": 0.0, "ciou": 0.0, "conf": 0.0, "cls": 0.0, "attn": 0.0}

    for images, targets in loader:
        images  = images.to(device,  non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with autocast(device_type, enabled=use_amp):
            raw_grid, saliencies = model(images)
            loss, loss_dict = criterion(raw_grid, saliencies, targets, input_hw)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        totals["total"] += loss.item()
        for k, v in loss_dict.items():
            totals[k] += v

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def train(
    data_dir: str, 
    cfg: EyeWaveConfig = CFG, 
    use_amp: bool = True, 
    train_fraction: float = 1.0,
    resume_ckpt: str = None
) -> nn.Module:
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GPU Trainer] Device: {device}  |  AMP: {use_amp}")

    model      = build_model(cfg.model, cfg.infer).to(device)
    print(f"[GPU Trainer] Parameters: {model.count_parameters():,}")

    train_loader, val_loader = get_loaders(
        data_dir, cfg.model, cfg.train,
        class_offset=0,
        train_fraction=train_fraction,
    )
    criterion  = EyeWaveLoss(cfg)
    optimizer  = AdamW(model.parameters(), lr=2e-3,
                       weight_decay=cfg.train.weight_decay)
    scheduler  = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs,
                                   eta_min=cfg.train.lr_min)
    use_amp = use_amp and torch.cuda.is_available()  # disable AMP on CPU
    scaler = GradScaler("cuda", enabled=use_amp)

    # Disable pin_memory warning on CPU
    if device.type == "cpu":
        train_loader.pin_memory = False
        val_loader.pin_memory = False

    save_dir   = Path(cfg.train.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_map   = 0.0
    best_iou   = 0.0
    input_hw   = cfg.model.train_input_size
    start_epoch = 1

    if resume_ckpt and Path(resume_ckpt).exists():
        print(f"[GPU Trainer] Resuming from {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        
        # Filter out shape mismatches (e.g. loading 2-stage ckpt into 1-stage model)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in ckpt["model_state"].items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        ignored = set(ckpt["model_state"].keys()) - set(pretrained_dict.keys())
        if ignored:
            print(f"[GPU Trainer] Ignored {len(ignored)} missing/mismatched keys from checkpoint.")
            
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "map50" in ckpt:
            best_map = ckpt["map50"]
        if "max_iou" in ckpt:
            best_iou = ckpt["max_iou"]
        print(f"[GPU Trainer] Restored epoch {start_epoch-1}, best mAP: {best_map:.4f}")

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        t0 = time.time()

        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, input_hw, use_amp,
        )

        # Use extremely low threshold in early epochs to show progress (debug)
        old_conf = cfg.infer.conf_thresh_full
        if epoch <= 10:
            cfg.infer.conf_thresh_full = 0.001
            print(f" [Debug] Val threshold: {cfg.infer.conf_thresh_full}")

        map50, cls_acc, max_iou = compute_map50(model, val_loader, cfg, device)
        
        # Restore threshold
        cfg.infer.conf_thresh_full = old_conf
        
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:3d}/{cfg.train.epochs}] "
            f"Loss: {train_losses['total']:.3f} "
            f"(CIoU:{train_losses['ciou']:.3f} "
            f"Conf:{train_losses['conf']:.3f} "
            f"Cls:{train_losses['cls']:.3f} "
            f"Attn:{train_losses['attn']:.3f})  "
            f"| mAP@0.5: {map50:.4f}  ClsAcc: {cls_acc:.4f} "
            f"| MaxIoU: {max_iou:.4f} "
            f"| LR: {scheduler.get_last_lr()[0]:.2e} "
            f"| {elapsed:.1f}s"
        )

        # Save based on mAP or MaxIoU if mAP is not yet moving
        if map50 > best_map or (map50 == 0 and max_iou > best_iou):
            if map50 > best_map:
                best_map = map50
            if max_iou > best_iou:
                best_iou = max_iou

            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "map50":       map50,
                "max_iou":     max_iou,
                "cls_acc":     cls_acc,
                "cfg":         cfg,
            }, save_dir / "best_gpu.pth")

        if epoch % cfg.train.save_every == 0:
            torch.save({"epoch": epoch, "model_state": model.state_dict()},
                       save_dir / f"epoch_{epoch:03d}.pth")

    print(f"\n[GPU Trainer] Done. Best mAP@0.5: {best_map:.4f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        required=True)
    parser.add_argument("--no_amp",          action="store_true")
    parser.add_argument("--train_fraction",  type=float, default=1.0,
                        help="Fraction of training data to use. e.g. 0.1 = 10%%")
    parser.add_argument("--resume",          type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. checkpoints/best_gpu.pth)")
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir, 
        use_amp=not args.no_amp, 
        train_fraction=args.train_fraction,
        resume_ckpt=args.resume
    )