"""
EyeWave — Quantization Aware Training (QAT)
-------------------------------------------
Pipeline:
FP32 checkpoint → Fuse → QAT → Fine-tune → INT8 convert

Usage:
python train_qat.py --data_dir DATASET --train_fraction 0.1
"""

import argparse
import copy
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import CFG, EyeWaveConfig
from model import build_model, FUSE_MODULES
from detection import EyeWaveLoss, decode_grid, nms, box_iou, box_cxcywh_to_xyxy
from dataset import get_loaders


# ─────────────────────────────────────────────────────────────
# Fusion
# ─────────────────────────────────────────────────────────────

def fuse_model(model: nn.Module):

    fused = 0
    skipped = 0

    for pair in FUSE_MODULES:
        try:
            tq.fuse_modules(model, [pair], inplace=True)
            fused += 1
        except Exception:
            skipped += 1

    print(f"[QAT] Fusion complete — fused: {fused} skipped: {skipped}")
    return model


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device,
    input_hw,
):

    model.train()

    totals = {"total":0.0,"ciou":0.0,"conf":0.0,"cls":0.0,"attn":0.0}

    for images, targets in loader:

        images  = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        raw_grid, saliencies = model(images)

        loss, loss_dict = criterion(raw_grid, saliencies, targets, input_hw)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(),1.0)

        optimizer.step()

        totals["total"] += loss.item()

        for k,v in loss_dict.items():
            totals[k] += v

    n = len(loader)

    return {k:v/n for k,v in totals.items()}


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_map50(model, loader, cfg, device):

    model.eval()

    input_hw = cfg.model.train_input_size

    tp = fp = gt_total = 0
    cls_correct = cls_total = 0

    for images, targets in loader:

        images  = images.to(device)
        targets = targets.to(device)

        raw, _ = model(images)
        decoded = decode_grid(raw, input_hw, cfg.model.stride)

        for i in range(images.shape[0]):

            gt_mask = targets[:,0] == i
            gt = targets[gt_mask]

            gt_total += gt.shape[0]

            dets = nms(
                decoded[i],
                conf_thresh=cfg.infer.conf_thresh_full,
                iou_thresh=cfg.infer.nms_iou_thresh,
                max_detections=cfg.model.max_detections,
            )

            if dets.shape[0] == 0 or gt.shape[0] == 0:
                fp += dets.shape[0]
                continue

            pred = box_cxcywh_to_xyxy(dets[:,:4])
            gtxy = box_cxcywh_to_xyxy(gt[:,1:5])

            iou = box_iou(pred, gtxy)

            matched=set()

            for d in range(dets.shape[0]):

                best_iou,best_g=iou[d].max(0)

                if best_iou>=0.5 and best_g.item() not in matched:

                    tp+=1
                    matched.add(best_g.item())

                    cls_total+=1
                    cls_correct+=int(
                        dets[d,5].item()==gt[best_g.item(),5].item()
                    )

                else:
                    fp+=1

    precision = tp/(tp+fp+1e-7)
    recall    = tp/(gt_total+1e-7)

    map50=(2*precision*recall)/(precision+recall+1e-7)

    cls_acc = cls_correct/(cls_total+1e-7)

    return map50, cls_acc


# ─────────────────────────────────────────────────────────────
# Main QAT routine
# ─────────────────────────────────────────────────────────────

def train_qat(data_dir: str, train_fraction: float, cfg: EyeWaveConfig = CFG):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[QAT] Device: {device} Backend: {cfg.qat.backend}")

    # build model
    model = build_model(cfg.model, cfg.infer).to(device)

    ckpt_path = Path(cfg.qat.pretrained_ckpt)

    if ckpt_path.exists():

        ckpt = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(ckpt["model_state"])

        print(f"[QAT] Loaded checkpoint {ckpt_path}")

    else:

        print("[QAT] WARNING: using random weights (QAT from scratch is NOT recommended)")
        # Increase LR slightly if training from scratch
        cfg.qat.lr *= 10
        print(f"[QAT] Increased LR to {cfg.qat.lr:.2e} for scratch training")

    # fuse modules
    model.eval()
    fuse_model(model)
    print("[QAT] Modules fused")

    # prepare QAT
    torch.backends.quantized.engine = cfg.qat.backend

    model.qconfig = get_default_qat_qconfig(cfg.qat.backend)

    model.train()

    prepare_qat(model, inplace=True)

    print("[QAT] FakeQuant inserted")

    # loaders
    qat_train_cfg = copy.copy(cfg.train)
    qat_train_cfg.batch_size = cfg.qat.batch_size

    train_loader, val_loader = get_loaders(
        data_dir,
        cfg.model,
        qat_train_cfg,
        class_offset=0,  # YOLO standard is 0-indexed
        train_fraction=train_fraction,
    )

    criterion = EyeWaveLoss(cfg).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.qat.lr,
        weight_decay=cfg.train.weight_decay
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.qat.epochs,
        eta_min=1e-7
    )

    save_dir = Path(cfg.qat.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_map = 0.0

    input_hw = cfg.model.train_input_size

    # training loop
    for epoch in range(1, cfg.qat.epochs+1):

        start = time.time()

        if epoch >= cfg.qat.freeze_observer_epoch:
            model.apply(tq.disable_observer)

        if epoch >= cfg.qat.freeze_bn_epoch:
            model.apply(nn.intrinsic.qat.freeze_bn_stats)

        losses = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            input_hw
        )

        # Use lower threshold in early scratch epochs to show progress
        is_scratch = not ckpt_path.exists()
        conf_t = cfg.infer.conf_thresh_full
        if is_scratch and epoch <= 3:
            conf_t = 0.05
            print(f" [Debug] Using lower conf_thresh: {conf_t}")

        # Temporary config override for validation
        old_t = cfg.infer.conf_thresh_full
        cfg.infer.conf_thresh_full = conf_t
        
        map50, cls_acc = compute_map50(
            model,
            val_loader,
            cfg,
            device
        )
        
        cfg.infer.conf_thresh_full = old_t

        scheduler.step()

        print(
            f"QAT [{epoch}/{cfg.qat.epochs}] "
            f"Loss:{losses['total']:.3f} "
            f"mAP@0.5:{map50:.4f} "
            f"ClsAcc:{cls_acc:.4f} "
            f"| {time.time()-start:.1f}s"
        )

        if map50>best_map:

            best_map=map50

            torch.save(
                {
                    "epoch":epoch,
                    "model_state":model.state_dict(),
                    "map50":map50
                },
                save_dir/"best_qat.pth"
            )

    print(f"[QAT] Best mAP@0.5 {best_map:.4f}")

    # convert to INT8
    model.eval()

    model_int8=copy.deepcopy(model)

    convert(model_int8,inplace=True)

    print("[QAT] Converted to INT8")

    map50_int8,cls_acc_int8=compute_map50(
        model_int8,
        val_loader,
        cfg,
        device
    )

    torch.save(
        {
            "model_state":model_int8.state_dict(),
            "map50":map50_int8,
            "cls_acc":cls_acc_int8
        },
        save_dir/"eyewave_int8.pth"
    )

    print("[QAT] INT8 model saved")

    return model_int8


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument("--data_dir",required=True)

    parser.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset used"
    )

    args=parser.parse_args()

    train_qat(
        data_dir=args.data_dir,
        train_fraction=args.train_fraction
    )