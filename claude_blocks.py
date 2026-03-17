"""
EyeWave Blocks
--------------
Three novel blocks:

  1. LearnedWaveletStem (LWS)        — trainable Haar-init DWT decomposition
  2. SubbandDisentangledBlock (SDB)  — orientation-matched depthwise convs per subband
  3. SpectralChannelAttention (SCA)  — channel attention weighted by subband energy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


# ── Helpers ───────────────────────────────────────────────────────────────────

def conv_bn_relu(
    in_ch: int,
    out_ch: int,
    kernel: int | Tuple[int, int],
    stride: int = 1,
    padding: int | Tuple[int, int] = 0,
    groups: int = 1,
) -> nn.Sequential:
    """Conv → BN → ReLU6 (fuse-friendly for QAT)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                  padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU6(inplace=True),
    )


# ── Block 1: Learned Wavelet Stem ─────────────────────────────────────────────

def dwt_haar(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Standard 2D Haar DWT."""
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) / 2
    hl = (x00 + x01 - x10 - x11) / 2
    lh = (x00 - x01 + x10 - x11) / 2
    hh = (x00 - x01 - x10 + x11) / 2
    return ll, hl, lh, hh


class LearnedWaveletStem(nn.Module):
    """
    Formal DWT-based stem.
    Uses fixed Haar transform followed by learnable 1×1 convolutions.

        Input:  (B, in_ch, H, W)
        Output: (LL, HL, LH, HH) each (B, out_ch, H/2, W/2)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.filter_ll = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.filter_hl = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.filter_lh = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.filter_hh = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        with torch.no_grad():
            for f in [self.filter_ll, self.filter_hl, self.filter_lh, self.filter_hh]:
                f.weight.fill_(1.0 / in_channels)

        self.bn_ll = nn.BatchNorm2d(out_channels)
        self.bn_hl = nn.BatchNorm2d(out_channels)
        self.bn_lh = nn.BatchNorm2d(out_channels)
        # HH has no BN — used as gate with raw sigmoid

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        ll_d, hl_d, lh_d, hh_d = dwt_haar(x)

        ll = F.relu6(self.bn_ll(self.filter_ll(ll_d)))
        hl = F.relu6(self.bn_hl(self.filter_hl(hl_d)))
        lh = F.relu6(self.bn_lh(self.filter_lh(lh_d)))
        hh = torch.sigmoid(self.filter_hh(hh_d))   # noise gate — no BN
        return ll, hl, lh, hh


# ── Block 2: Subband Disentangled Block ───────────────────────────────────────

class SubbandDisentangledBlock(nn.Module):
    """
    Each subband gets an orientation-matched depthwise conv:
      LL  → 3×3 DW  (isotropic: overall eye shape)
      HL  → 1×3 DW  (horizontal: eyelid edges / blink)
      LH  → 3×1 DW  (vertical: iris shift / gaze)
      HH  → sigmoid gate (noise suppression → multiplied into LL)

    Fusion: concat(LL_gated, HL, LH) → 1×1 conv → out_channels
    """

    def __init__(self, channels: int, out_channels: int) -> None:
        super().__init__()

        self.branch_ll = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU6(inplace=True),
        )
        self.branch_hl = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU6(inplace=True),
        )
        self.branch_lh = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU6(inplace=True),
        )
        # HH gate: channel-wise scalar in (0,1)
        self.hh_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.fuse = conv_bn_relu(channels * 3, out_channels, kernel=1)

    def forward(self, ll: Tensor, hl: Tensor, lh: Tensor, hh: Tensor) -> Tensor:
        ll_out = self.branch_ll(ll)
        hl_out = self.branch_hl(hl)
        lh_out = self.branch_lh(lh)

        gate = self.hh_gate(hh).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        ll_gated = ll_out * gate

        fused = torch.cat([ll_gated, hl_out, lh_out], dim=1)
        return self.fuse(fused)


# ── Block 3: Spectral Channel Attention ───────────────────────────────────────

class SpectralChannelAttention(nn.Module):
    """
    Channel attention weighted by spectral energy per channel,
    NOT by spatial pooling (unlike SE blocks).

    E_k = mean(|F_k|²) → normalised → attention weights via MLP → reweight channels
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        bottleneck = max(channels // reduction, 4)
        self.attention = nn.Sequential(
            nn.Linear(channels, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        energy = (x ** 2).mean(dim=(-2, -1))                          # (B, C)
        energy = energy / (energy.max(dim=1, keepdim=True).values + 1e-6)
        weights = self.attention(energy).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * weights
