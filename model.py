import torch
import torch.nn as nn
from torch import Tensor
from torch.ao.quantization import QuantStub, DeQuantStub
from typing import List, Optional, Tuple

from blocks import (
    LearnedWaveletStem,
    CrossSubbandSaliency,
    SubbandDisentangledBlock,
    SpectralChannelAttention,
)
from detection import DetectionHead, decode_grid, nms
from config import ModelConfig, InferenceConfig


class EyeWave(nn.Module):
    """
    EyeWave: Anatomy-Driven Wavelet Architecture for Eye Signal Detection.

    New in this version:
        CrossSubbandSaliency (CSS) inserted between LWS and SDB at both stages.
        Focuses subbands on eye regions using the HL×LH×(1−HH) identity,
        proven pose-invariant on real face images.

    Args:
        model_cfg:  ModelConfig
        infer_cfg:  InferenceConfig
    """

    def __init__(
        self,
        model_cfg: Optional[ModelConfig]     = None,
        infer_cfg: Optional[InferenceConfig] = None,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg or ModelConfig()
        self.infer_cfg = infer_cfg or InferenceConfig()

        # C and C2 are channel counts for the two stages
        C  = self.model_cfg.base_channels
        C2 = C * 2

        self.quant   = QuantStub()
        self.dequant = DeQuantStub()
        self.stem1 = LearnedWaveletStem(self.model_cfg.input_channels, C)
        self.css1  = CrossSubbandSaliency(C)           # NEW: focus on eyes
        self.sdb1  = SubbandDisentangledBlock(C, C)
        
        self.stem2 = LearnedWaveletStem(C, C2)
        self.css2  = CrossSubbandSaliency(C2)         # NEW: re-focus at finer scale
        self.sdb2  = SubbandDisentangledBlock(C2, C2)

        self.sca  = SpectralChannelAttention(C2, reduction=4)
        self.head = DetectionHead(C2, self.model_cfg.num_classes)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Returns:
            raw_grid: (B, H*W, 5+num_classes)
            saliencies: List of [sal1, sal2]
        """
        x = self.quant(x)

        # Stage 1: DWT → CSS focus → SDB
        ll, hl, lh, hh     = self.stem1(x)
        ll, hl, lh, hh, sal1 = self.css1(ll, hl, lh, hh)    # focus on eyes
        x                  = self.sdb1(ll, hl, lh, hh)

        # Stage 2: DWT → CSS focus → SDB
        ll, hl, lh, hh     = self.stem2(x)
        ll, hl, lh, hh, sal2 = self.css2(ll, hl, lh, hh)    # re-focus at H/4 scale
        x                  = self.sdb2(ll, hl, lh, hh)

        # Spectral channel attention + detection head
        x        = self.sca(x)
        raw_grid = self.head(x)
        
        raw_grid = self.dequant(raw_grid)
        return raw_grid, [sal1, sal2]

    @torch.no_grad()
    def predict(self, x: Tensor, partial_face: bool = True) -> List[Tensor]:
        """
        Full inference: forward → decode → NMS.

        Returns:
            List of (D, 6) per image — [cx, cy, w, h, conf, class_id]
        """
        self.eval()
        raw_grid, _ = self(x)
        input_hw   = (x.shape[2], x.shape[3])
        conf_thresh = (self.infer_cfg.conf_thresh_partial if partial_face
                       else self.infer_cfg.conf_thresh_full)
        decoded    = decode_grid(raw_grid, input_hw, self.model_cfg.stride)

        return [
            nms(decoded[i], conf_thresh,
                self.infer_cfg.nms_iou_thresh,
                self.model_cfg.max_detections)
            for i in range(x.shape[0])
        ]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def feature_map_size(self, input_hw: Tuple[int, int]) -> Tuple[int, int]:
        H, W = input_hw
        return H // self.model_cfg.stride, W // self.model_cfg.stride

FUSE_MODULES = [
    # SDB stage 1 — Conv+BN only (index 0, 1)
    ["sdb1.branch_ll.0", "sdb1.branch_ll.1"],
    ["sdb1.hl_3.0",      "sdb1.hl_3.1"],
    ["sdb1.hl_6.0",      "sdb1.hl_6.1"],
    ["sdb1.hl_9.0",      "sdb1.hl_9.1"],
    ["sdb1.lh_3.0",      "sdb1.lh_3.1"],
    ["sdb1.lh_6.0",      "sdb1.lh_6.1"],
    ["sdb1.lh_9.0",      "sdb1.lh_9.1"],
    ["sdb1.hh_noise.0",  "sdb1.hh_noise.1"],
    ["sdb1.fuse.0",      "sdb1.fuse.1"],
    # # SDB stage 2
    # ["sdb2.branch_ll.0", "sdb2.branch_ll.1"],
    # ["sdb2.hl_3.0",      "sdb2.hl_3.1"],
    # ["sdb2.hl_6.0",      "sdb2.hl_6.1"],
    # ["sdb2.hl_9.0",      "sdb2.hl_9.1"],
    # ["sdb2.lh_3.0",      "sdb2.lh_3.1"],
    # ["sdb2.lh_6.0",      "sdb2.lh_6.1"],
    # ["sdb2.lh_9.0",      "sdb2.lh_9.1"],
    # ["sdb2.hh_noise.0",  "sdb2.hh_noise.1"],
    # ["sdb2.fuse.0",      "sdb2.fuse.1"],
    # LWS stems — Conv+BN
    ["stem1.filter_ll",  "stem1.bn_ll"],
    ["stem1.filter_hl",  "stem1.bn_hl"],
    ["stem1.filter_lh",  "stem1.bn_lh"],
    ["stem1.filter_hh",  "stem1.bn_hh"],
    ["stem2.filter_ll",  "stem2.bn_ll"],
    ["stem2.filter_hl",  "stem2.bn_hl"],
    ["stem2.filter_lh",  "stem2.bn_lh"],
    ["stem2.filter_hh",  "stem2.bn_hh"],
    # Detection head pre-conv — Conv+BN only
    ["head.pre.1",       "head.pre.2"],
]


def build_model(
    model_cfg: Optional[ModelConfig]     = None,
    infer_cfg: Optional[InferenceConfig] = None,
) -> EyeWave:
    return EyeWave(model_cfg, infer_cfg)


if __name__ == "__main__":
    from config import CFG

    model = build_model(CFG.model, CFG.infer)
    print(f"Parameters: {model.count_parameters():,}")

    # Print per-block param counts
    blocks = {
        "stem1 + stem2 (LWS)": sum(p.numel() for n,p in model.named_parameters() if 'stem' in n),
        "css1 + css2 (CSS)":   sum(p.numel() for n,p in model.named_parameters() if 'css' in n),
        "sdb1 + sdb2 (SDB)":   sum(p.numel() for n,p in model.named_parameters() if 'sdb' in n),
        "sca (SCA)":           sum(p.numel() for n,p in model.named_parameters() if 'sca' in n),
        "head":                sum(p.numel() for n,p in model.named_parameters() if 'head' in n),
    }
    for name, count in blocks.items():
        print(f"  {name:<30} {count:>8,} params")

    # Training forward (full face)
    dummy_train = torch.randn(2, CFG.model.input_channels, *CFG.model.train_input_size)
    raw, _ = model(dummy_train)
    fh, fw = model.feature_map_size(CFG.model.train_input_size)
    print(f"\nTrain  {tuple(dummy_train.shape)} → raw grid {tuple(raw.shape)}")
    print(f"Feature map: {fh}×{fw} = {fh*fw} cells")

    # Inference forward (partial face)
    dummy_infer = torch.randn(1, CFG.model.input_channels, *CFG.model.infer_input_size)
    dets = model.predict(dummy_infer, partial_face=True)
    print(f"Infer  {tuple(dummy_infer.shape)} → detections {tuple(dets[0].shape)}")
