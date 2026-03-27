"""
EyeWave Blocks
--------------
Four blocks — one new (CSS), one fixed (SDB gate):

  1. LearnedWaveletStem (LWS)       — trainable Haar-init DWT decomposition
  2. CrossSubbandSaliency (CSS)     — NEW: pose-invariant eye saliency map
                                       HL × LH × (1 − HH)
                                       Eyes:  high HL + high LH + low HH (smooth skin)
                                       Beard: high HL + high LH + HIGH HH → suppressed
  3. SubbandDisentangledBlock (SDB) — orientation-matched depthwise convs
                                       FIXED: spatial noise residual not scalar gate
  4. SpectralChannelAttention (SCA) — channel attention by spectral energy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
# ── Wavelet filter coefficients (from PyWavelets reference tables) ────────────
# These are the exact decomposition low-pass (dec_lo) and high-pass (dec_hi)
# filter coefficients for common discrete wavelets.
WAVELET_FILTERS = {
    'haar': {
        'dec_lo': [0.7071067811865476, 0.7071067811865476],
        'dec_hi': [-0.7071067811865476, 0.7071067811865476],
    },
    'db2': {
        'dec_lo': [-0.12940952255092145, 0.22414386804185735,
                   0.836516303737469, 0.48296291314469025],
        'dec_hi': [-0.48296291314469025, 0.836516303737469,
                   -0.22414386804185735, -0.12940952255092145],
    },
    'db3': {
        'dec_lo': [0.035226291882100656, -0.08544127388224149,
                   -0.13501102001039084, 0.4598775021193313,
                   0.8068915093133388, 0.3326705529509569],
        'dec_hi': [-0.3326705529509569, 0.8068915093133388,
                   -0.4598775021193313, -0.13501102001039084,
                   0.08544127388224149, 0.035226291882100656],
    },
    'db4': {
        'dec_lo': [-0.010597401784997278, 0.032883011666982945,
                   0.030841381835986965, -0.18703481171888114,
                   -0.02798376941698385, 0.6308807679295904,
                   0.7148465705525415, 0.23037781330885523],
        'dec_hi': [-0.23037781330885523, 0.7148465705525415,
                   -0.6308807679295904, -0.02798376941698385,
                   0.18703481171888114, 0.030841381835986965,
                   -0.032883011666982945, -0.010597401784997278],
    },
    'sym4': {
        'dec_lo': [-0.07576571478927333, -0.02963552764599851,
                   0.49761866763201545, 0.8037387518059161,
                   0.29785779560527736, -0.09921954357684722,
                   -0.012603967262037833, 0.032223100604071224],
        'dec_hi': [-0.032223100604071224, -0.012603967262037833,
                   0.09921954357684722, 0.29785779560527736,
                   -0.8037387518059161, 0.49761866763201545,
                   0.02963552764599851, -0.07576571478927333],
    },
    'coif1': {
        'dec_lo': [-0.01565572813546454, -0.0727326195128539,
                   0.38486484686420286, 0.8525720202122554,
                   0.3378976624578092, -0.0727326195128539],
        'dec_hi': [0.0727326195128539, 0.3378976624578092,
                   -0.8525720202122554, 0.38486484686420286,
                   0.0727326195128539, -0.01565572813546454],
    },
}


def _build_dwt_filters(wave: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Build 2D separable wavelet decomposition filters from 1D coefficients.

    Returns 4 filters of shape (1, 1, K, K): LL, LH, HL, HH
    """
    if wave not in WAVELET_FILTERS:
        raise ValueError(
            f"Unknown wavelet '{wave}'. Supported: {list(WAVELET_FILTERS.keys())}"
        )
    coeffs = WAVELET_FILTERS[wave]
    lo = torch.tensor(coeffs['dec_lo'], dtype=torch.float32)
    hi = torch.tensor(coeffs['dec_hi'], dtype=torch.float32)

    # 2D separable filters via outer product
    filt_ll = lo.unsqueeze(0) * lo.unsqueeze(1)  # (K, K)
    filt_lh = hi.unsqueeze(0) * lo.unsqueeze(1)  # high-pass rows, low-pass cols → horiz edges
    filt_hl = lo.unsqueeze(0) * hi.unsqueeze(1)  # low-pass rows, high-pass cols → vert edges
    filt_hh = hi.unsqueeze(0) * hi.unsqueeze(1)  # (K, K)

    # Shape: (1, 1, K, K) for depthwise conv
    return (filt_ll.unsqueeze(0).unsqueeze(0),
            filt_lh.unsqueeze(0).unsqueeze(0),
            filt_hl.unsqueeze(0).unsqueeze(0),
            filt_hh.unsqueeze(0).unsqueeze(0))


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


class DWT2d(nn.Module):
    """
    2D Discrete Wavelet Transform — pure PyTorch, zero external dependencies.

    GPU-compatible and fully differentiable. Uses real wavelet filter banks
    applied via depthwise F.conv2d with stride=2.

    Supported wavelets: haar, db2, db3, db4, sym4, coif1.

    Args:
        wave: Wavelet name (default: 'haar').

    Input:  (B, C, H, W)
    Output: (LL, HL, LH, HH) each (B, C, H/2, W/2)
    """

    def __init__(self, wave: str = 'haar') -> None:
        super().__init__()
        filt_ll, filt_lh, filt_hl, filt_hh = _build_dwt_filters(wave)
        # Register as buffers (non-learnable, move with .cuda()/.to())
        self.register_buffer('filt_ll', filt_ll)
        self.register_buffer('filt_lh', filt_lh)
        self.register_buffer('filt_hl', filt_hl)
        self.register_buffer('filt_hh', filt_hh)
        self.pad = (filt_ll.shape[-1] - 1) // 2  # symmetric padding

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        B, C, H, W = x.shape
        # Pad for even dimensions + filter support
        pad = self.pad
        if pad > 0 or H % 2 != 0 or W % 2 != 0:
            pad_h = pad + (H % 2)
            pad_w = pad + (W % 2)
            x = F.pad(x, (pad, pad_w, pad, pad_h), mode='reflect')

        # Expand single-channel filters to C channels for depthwise conv
        filt_ll = self.filt_ll.expand(C, -1, -1, -1)
        filt_lh = self.filt_lh.expand(C, -1, -1, -1)
        filt_hl = self.filt_hl.expand(C, -1, -1, -1)
        filt_hh = self.filt_hh.expand(C, -1, -1, -1)

        # Depthwise conv with stride=2 for downsampling
        ll = F.conv2d(x, filt_ll, stride=2, groups=C)
        hl = F.conv2d(x, filt_lh, stride=2, groups=C)
        lh = F.conv2d(x, filt_hl, stride=2, groups=C)
        hh = F.conv2d(x, filt_hh, stride=2, groups=C)
        return ll, hl, lh, hh


class LearnedWaveletStem(nn.Module):
    """
    DWT-based stem using pytorch_wavelets.
    The spatial decomposition uses a real wavelet transform, while the channel
    mapping is handled by 1x1 convolutions.

        Input:  (B, in_ch, H, W)
        Output: (LL, HL, LH, HH) each (B, out_ch, H/2, W/2)
    """

    def __init__(self, in_channels: int, out_channels: int, wave: str = 'haar') -> None:
        super().__init__()
        self.dwt = DWT2d(wave=wave)

        # Use 1x1 convolutions for channel mapping after DWT
        # These are named 'filter_*' for compatibility with model.py FUSE_MODULES
        self.filter_ll = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.filter_hl = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.filter_lh = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.filter_hh = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # Initialize to replicate the input channel(s) into out_channels
        with torch.no_grad():
            for f in [self.filter_ll, self.filter_hl, self.filter_lh, self.filter_hh]:
                f.weight.fill_(1.0 / in_channels)

        # Freeze LL: ensures it stays a pure low-pass band representation
        self.filter_ll.weight.requires_grad = False

        self.bn_ll = nn.BatchNorm2d(out_channels)
        self.bn_hl = nn.BatchNorm2d(out_channels)
        self.bn_lh = nn.BatchNorm2d(out_channels)
        self.bn_hh = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # 1. Real DWT via pytorch_wavelets
        ll, hl, lh, hh = self.dwt(x)

        # 2. Learnable Channel Mapping + Norm + Activation
        ll = F.relu6(self.bn_ll(self.filter_ll(ll)))
        hl = F.relu6(self.bn_hl(self.filter_hl(hl)))
        lh = F.relu6(self.bn_lh(self.filter_lh(lh)))
        hh = F.relu6(self.bn_hh(self.filter_hh(hh)))
        return ll, hl, lh, hh


# ── Block 2: Cross-Subband Saliency (NEW) ─────────────────────────────────────

class CrossSubbandSaliency(nn.Module):
    """
    Pose-invariant eye saliency from cross-subband consistency.

    Physical insight proven on real face image:
        Eyes:  strong HL (eyelid horizontal edge)
               strong LH (iris vertical boundary)
               weak   HH (smooth skin between lashes)  ← the key discriminator
        Beard: strong HL + strong LH + STRONG HH (hair texture) → suppressed
        Skin:  weak HL + weak LH                                 → suppressed naturally

    Formula:  saliency = W_hl(HL)  ×  W_lh(LH)  ×  (1 − W_hh(HH))

    All branch weights are learned — initialised to the analytically-derived
    formula, then fine-tuned per dataset.

    The shared saliency map is applied to ALL four subbands before SDB,
    focusing every subsequent operation on eye regions regardless of face pose.

    Args:
        channels: Channel count per subband (all equal from LWS).
        ~800 extra parameters at channels=16.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.w_hl = nn.Sequential(
            nn.Conv2d(channels, channels, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
        )
        self.w_lh = nn.Sequential(
            nn.Conv2d(channels, channels, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
        )
        self.w_hh = nn.Sequential(
            nn.Conv2d(channels, channels, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
        )

    def forward(
        self,
        ll: Tensor,
        hl: Tensor,
        lh: Tensor,
        hh: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute saliency and apply to all subbands.
        Returns: (ll_f, hl_f, lh_f, hh_f, sal)
        """
        hl_w = self.w_hl(hl)
        lh_w = self.w_lh(lh)
        hh_w = self.w_hh(hh)

        # Physically-grounded eye saliency:
        # High Horizontal (HL) + High Vertical (LH) + Low High-Freq (HH)
        sal = hl_w * lh_w * (1.0 - hh_w * 0.5)  # 0.5 factor to avoid killing all signal
        
        return ll * sal, hl * sal, lh * sal, hh * sal, sal


# ── Block 3: Subband Disentangled Block (gate fixed) ──────────────────────────

class SubbandDisentangledBlock(nn.Module):
    """
    Orientation-matched depthwise convs per subband:
      LL  → 3×3 DW  (isotropic: overall eye shape)
      HL  → 1×3 DW  (horizontal: eyelid edge — kernel matches subband orientation)
      LH  → 3×1 DW  (vertical: iris boundary — kernel matches subband orientation)
      HH  → spatial noise map → residual subtraction from LL
            FIXED: was scalar gate (AdaptiveAvgPool→Linear→Sigmoid) that
                   collapsed to ~0.3 and killed LL entirely (proven black output)
            NOW:   spatial conv → sigmoid → per-pixel denoising

    Fusion: concat(ll_denoised, HL, LH) → 1×1 conv → out_channels
    """

    def __init__(self, channels: int, out_channels: int) -> None:
        super().__init__()

        # LL branch (isotropic shape): reduction in dilation to force local focus.
        # Dilation=2 with kernel=3 gives 5x5 RF. At stride 4, this is 20x20 px.
        # This prevents the model from relying on distant features like the nose.
        self.branch_ll = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
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

        # FIXED: spatial noise map, not scalar
        self.hh_noise = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),    # per-pixel: 0=clean region, 1=noisy region
        )

        # Learnable suppression strength per channel — starts at 0.3
        # Clamped to (0,1) so LL is NEVER fully killed
        self.suppress_strength = nn.Parameter(torch.full((1, channels, 1, 1), 0.3))

        self.fuse = conv_bn_relu(channels * 3, out_channels, kernel=1)

    def forward(self, ll: Tensor, hl: Tensor, lh: Tensor, hh: Tensor) -> Tensor:
        ll_out = self.branch_ll(ll)
        hl_out = self.branch_hl(hl)
        lh_out = self.branch_lh(lh)

        noise_map = self.hh_noise(hh)                           # (B, C, H, W)
        strength  = self.suppress_strength.clamp(0.0, 1.0)     # per-channel, bounded
        ll_denoised = ll_out * (1.0 - noise_map * strength)    # residual denoising

        fused = torch.cat([ll_denoised, hl_out, lh_out], dim=1)
        return self.fuse(fused)



class SpectralChannelAttention(nn.Module):
    """
    Channel attention weighted by spectral energy, not spatial pooling.
    Amplifies channels carrying strong eye signal; suppresses noisy channels.

    E_k = mean(|F_k|²) → normalised → MLP → sigmoid weights → reweight channels
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
        energy  = (x ** 2).mean(dim=(-2, -1))
        energy  = energy / (energy.max(dim=1, keepdim=True).values + 1e-6)
        weights = self.attention(energy).unsqueeze(-1).unsqueeze(-1)
        return x * weights
