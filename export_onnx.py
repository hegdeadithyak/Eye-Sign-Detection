"""
EyeWave — ONNX Export + Benchmark
-----------------------------------
Exports FP32 and/or INT8 models to ONNX with the full
decode+NMS pipeline baked in (single inference call at runtime).

ONNX output tensor: (B, max_detections, 6)
    Each row: [cx, cy, w, h, conf, class_id]  normalised 0..1

Usage:
    python export_onnx.py --mode both
    python export_onnx.py --mode fp32 --fp32_ckpt checkpoints/best_gpu.pth
    python export_onnx.py --mode int8 --int8_ckpt checkpoints/eyewave_int8.pth
"""

import argparse
import copy
import time
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from config import CFG, ModelConfig, InferenceConfig, ONNXConfig
from model import build_model, FUSE_MODULES
from detection import decode_grid, nms

try:
    import onnx
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("WARNING: onnxruntime not installed — run: pip install onnxruntime")


# ══════════════════════════════════════════════════════════════════════════════
# Wrapper: bakes decode + NMS into the ONNX graph
# ══════════════════════════════════════════════════════════════════════════════

class EyeWaveWithPostProcess(nn.Module):
    """
    Wraps EyeWave backbone + head with decode+NMS so the full
    pipeline is a single ONNX graph call.

    Note: NMS uses fixed inference input size (partial face).
    Output: (B, max_det, 6) — zero-padded if fewer than max_det detections.
    """

    def __init__(self, model, model_cfg: ModelConfig, infer_cfg: InferenceConfig) -> None:
        super().__init__()
        self.model       = model
        self.model_cfg   = model_cfg
        self.infer_cfg   = infer_cfg
        self.input_hw    = model_cfg.infer_input_size
        self.stride      = model_cfg.stride
        self.max_det     = model_cfg.max_detections
        self.conf_thresh = infer_cfg.conf_thresh_partial
        self.iou_thresh  = infer_cfg.nms_iou_thresh

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, max_det, 6) — [cx, cy, w, h, conf, cls]
        """
        raw, _ = self.model(x)
        dec   = decode_grid(raw, self.input_hw, self.stride)
        B     = x.shape[0]
        out   = torch.zeros(B, self.max_det, 6, device=x.device)

        for i in range(B):
            dets = nms(dec[i], self.conf_thresh, self.iou_thresh, self.max_det)
            n    = min(dets.shape[0], self.max_det)
            if n > 0:
                out[i, :n] = dets[:n]
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Load helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_fp32(ckpt_path: str, cfg=CFG) -> nn.Module:
    model = build_model(cfg.model, cfg.infer)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_int8(ckpt_path: str, cfg=CFG) -> nn.Module:
    import torch.ao.quantization as tq
    from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert

    model = build_model(cfg.model, cfg.infer)
    model.eval()

    fused = 0
    for pair in FUSE_MODULES:
        try:
            tq.fuse_modules(model, [pair], inplace=True)
            fused += 1
        except Exception:
            pass
    model.qconfig = get_default_qat_qconfig(cfg.qat.backend)
    prepare_qat(model, inplace=True)
    convert(model, inplace=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Export
# ══════════════════════════════════════════════════════════════════════════════

def export_onnx(
    model: nn.Module,
    cfg=CFG,
    output_path: str = "eyewave.onnx",
) -> None:
    wrapped = EyeWaveWithPostProcess(model, cfg.model, cfg.infer)
    wrapped.eval()

    C, (H, W) = cfg.model.input_channels, cfg.model.infer_input_size
    dummy = torch.randn(1, C, H, W)

    dynamic_axes = {}
    if cfg.onnx.dynamic_batch:
        dynamic_axes = {
            cfg.onnx.input_name:  {0: "batch_size"},
            cfg.onnx.output_name: {0: "batch_size"},
        }

    torch.onnx.export(
        wrapped,
        dummy,
        output_path,
        opset_version=cfg.onnx.opset,
        input_names=[cfg.onnx.input_name],
        output_names=[cfg.onnx.output_name],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        verbose=False,
    )
    print(f"[Export] Saved: {output_path}")

    # Verify ONNX graph
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)
    print(f"[Export] Graph check ✓  |  "
          f"Output shape: (B, {cfg.model.max_detections}, 6)  "
          f"[cx, cy, w, h, conf, cls]")


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_onnx(onnx_path: str, cfg=CFG, n: int = 300, warmup: int = 30) -> None:
    if not ORT_AVAILABLE:
        return
    C, (H, W) = cfg.model.input_channels, cfg.model.infer_input_size
    inp  = np.random.randn(1, C, H, W).astype(np.float32)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    feed = {cfg.onnx.input_name: inp}

    for _ in range(warmup):
        sess.run(None, feed)

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    print(
        f"[Benchmark] {Path(onnx_path).name}  "
        f"Mean: {sum(times)/len(times):.2f}ms  "
        f"P50: {times[n//2]:.2f}ms  "
        f"P95: {times[int(n*.95)]:.2f}ms  "
        f"P99: {times[int(n*.99)]:.2f}ms"
    )


def benchmark_pytorch(model: nn.Module, cfg=CFG, n: int = 300, warmup: int = 30) -> None:
    C, (H, W) = cfg.model.input_channels, cfg.model.infer_input_size
    dummy = torch.randn(1, C, H, W)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    print(
        f"[Benchmark] PyTorch  "
        f"Mean: {sum(times)/len(times):.2f}ms  "
        f"P50: {times[n//2]:.2f}ms  "
        f"P95: {times[int(n*.95)]:.2f}ms"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def export(
    mode: str,
    fp32_ckpt: Optional[str],
    int8_ckpt: Optional[str],
    output_dir: str,
    cfg=CFG,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if mode in ("fp32", "both") and fp32_ckpt and Path(fp32_ckpt).exists():
        print("\n── FP32 ─────────────────────────────────────────────")
        m = load_fp32(fp32_ckpt, cfg)
        p = str(Path(output_dir) / "eyewave_fp32.onnx")
        export_onnx(m, cfg, p)
        benchmark_pytorch(m, cfg)
        benchmark_onnx(p, cfg)

    if mode in ("int8", "both") and int8_ckpt and Path(int8_ckpt).exists():
        print("\n── INT8 ─────────────────────────────────────────────")
        m = load_int8(int8_ckpt, cfg)
        p = str(Path(output_dir) / "eyewave_int8.onnx")
        export_onnx(m, cfg, p)
        benchmark_pytorch(m, cfg)
        benchmark_onnx(p, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["fp32", "int8", "both"], default="both")
    parser.add_argument("--fp32_ckpt",  default="checkpoints/best_gpu.pth")
    parser.add_argument("--int8_ckpt",  default="checkpoints/eyewave_int8.pth")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    export(args.mode, args.fp32_ckpt, args.int8_ckpt, args.output_dir)