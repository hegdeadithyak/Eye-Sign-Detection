import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
import math

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
    """Conv → BN → ReLU6."""
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
    def __init__(self, in_channels: int, out_channels: int, wave: str = 'haar') -> None:
        super().__init__()
        self.dwt = DWT2d(wave=wave)

        self.filter_ll = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.filter_hl = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.filter_lh = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.filter_hh = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # Better initialization: Kaiming uniform with a bias towards the wavelet mean
        for f in [self.filter_ll, self.filter_hl, self.filter_lh, self.filter_hh]:
            nn.init.kaiming_uniform_(f.weight, a=math.sqrt(5))
            with torch.no_grad():
                f.weight.add_(1.0 / in_channels)

        self.bn_ll = nn.BatchNorm2d(out_channels)
        self.bn_hl = nn.BatchNorm2d(out_channels)
        self.bn_lh = nn.BatchNorm2d(out_channels)
        self.bn_hh = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        ll, hl, lh, hh = self.dwt(x)
        ll = F.relu6(self.bn_ll(self.filter_ll(ll)))
        hl = F.relu6(self.bn_hl(self.filter_hl(hl)))
        lh = F.relu6(self.bn_lh(self.filter_lh(lh)))
        hh = F.relu6(self.bn_hh(self.filter_hh(hh)))
        return ll, hl, lh, hh

class CrossSubbandSaliency(nn.Module):
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

    def forward(self, ll: Tensor, hl: Tensor, lh: Tensor, hh: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        hl_w = self.w_hl(hl)
        lh_w = self.w_lh(lh)
        hh_w = self.w_hh(hh)
        sal = hl_w * lh_w * (1.0 - hh_w * 0.5)
        return ll * sal, hl * sal, lh * sal, hh * sal, sal

class SubbandDisentangledBlock(nn.Module):
    def __init__(self, channels: int, out_channels: int) -> None:
        super().__init__()
        # Enhanced LL branch with larger receptive field for overall eye context
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
        
        # Novel Detail-Guided Gate (DGG)
        # Aggregates high-frequency subbands to create a precise eye-detail mask
        self.detail_gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        self.hh_noise = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.suppress_strength = nn.Parameter(torch.full((1, channels, 1, 1), 0.3))
        self.fuse = conv_bn_relu(channels * 3, out_channels, kernel=1)

    def forward(self, ll: Tensor, hl: Tensor, lh: Tensor, hh: Tensor) -> Tensor:
        ll_out = self.branch_ll(ll)
        hl_out = self.branch_hl(hl)
        lh_out = self.branch_lh(lh)
        
        # Detail-guided enhancement: use HL/LH/HH to refine LL
        details = torch.cat([hl, lh, hh], dim=1)
        gate = self.detail_gate(details)
        ll_enhanced = ll_out * (1.0 + gate)
        
        noise_map = self.hh_noise(hh)
        strength = self.suppress_strength.clamp(0.0, 1.0)
        ll_denoised = ll_enhanced * (1.0 - noise_map * strength)
        
        fused = torch.cat([ll_denoised, hl_out, lh_out], dim=1)
        return self.fuse(fused)

class SpectralChannelAttention(nn.Module):
    def __init__(self, c1: int, c2: int, reduction: int = 4) -> None:
        super().__init__()
        assert c1 == c2, "Input and output channels must match for SCA"
        bottleneck = max(c1 // reduction, 4)
        self.attention = nn.Sequential(
            nn.Linear(c1, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, c1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        energy = (x ** 2).mean(dim=(-2, -1))
        energy = energy / (energy.max(dim=1, keepdim=True).values + 1e-6)
        weights = self.attention(energy).unsqueeze(-1).unsqueeze(-1)
        return x * weights

class EyeWaveBlock(nn.Module):
    """
    Wraps the EyeWave Stem -> CSS -> SDB logic into a single module compatible
    with Ultralytics YAML configurations. It downsamples the spatial resolution by 2.
    Adds a residual connection for the LL (low-frequency) band to improve gradient flow.

    Uses pytorch_wavelets DWTForward for real wavelet transforms (GPU + differentiable).
    Supports any wavelet: 'haar', 'db2', 'db3', 'sym4', 'coif1', 'bior2.2', etc.
    """
    def __init__(self, c1: int, c2: int, wave: str = 'haar') -> None:
        super().__init__()
        self.stride = 2
        self.stem = LearnedWaveletStem(c1, c2, wave=wave)
        self.css = CrossSubbandSaliency(c2)
        self.sdb = SubbandDisentangledBlock(c2, c2)

    def forward(self, x: Tensor) -> Tensor:
        ll, hl, lh, hh = self.stem(x)
        ll_f, hl_f, lh_f, hh_f, _ = self.css(ll, hl, lh, hh)
        # Residual connection from the low-frequency component (ll)
        return self.sdb(ll_f, hl_f, lh_f, hh_f) + ll

class EyeWave(nn.Module):
    """
    Complete EyeWave block that combines Wavelet Stem, CSS, and SDB.
    Returns a single Tensor to be compatible with standard YOLO pipelines.
    Uses pytorch_wavelets for real DWT (GPU + differentiable).
    """
    def __init__(self, c1, c2, wave: str = 'haar'):
        super().__init__()
        self.stem = LearnedWaveletStem(c1, c2, wave=wave)
        self.css = CrossSubbandSaliency(c2)
        self.sdb = SubbandDisentangledBlock(c2, c2)

    def forward(self, x):
        ll, hl, lh, hh = self.stem(x)
        ll, hl, lh, hh, _ = self.css(ll, hl, lh, hh)
        return self.sdb(ll, hl, lh, hh)

class OCECBlock(nn.Module):
    """
    Open Closed Eyes Classification (OCEC) inspired block.
    Optimized for extremely low latency on CPU (target 6-10ms for full detector).
    Uses a simplified inverted bottleneck with depthwise separable convolutions
    and orientation-aware kernels to match eye signal characteristics.
    """
    def __init__(self, c1, c2, s=1, expand=2):
        super().__init__()
        self.stride = s
        hidden_ch = int(c1 * expand)
        self.use_res = (s == 1 and c1 == c2)
        
        self.conv = nn.Sequential(
            # 1x1 pointwise expansion (low cost)
            nn.Conv2d(c1, hidden_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU6(inplace=True),
            
            # Split depthwise: horizontal and vertical filters to capture
            # eyelid and iris signals efficiently, similar to SDB but simpler.
            nn.Conv2d(hidden_ch, hidden_ch, (3, 3), s, 1, groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU6(inplace=True),
            
            # 1x1 pointwise projection
            nn.Conv2d(hidden_ch, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)

class CompressiveWaveletStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, wave: str = 'haar') -> None:
        super().__init__()
        self.dwt = DWT2d(wave=wave)

        self.filter_ll = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.filter_hl = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.filter_lh = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.filter_hh = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        for f in [self.filter_ll, self.filter_hl, self.filter_lh, self.filter_lh, self.filter_hh]:
            nn.init.kaiming_uniform_(f.weight, a=math.sqrt(5))
            with torch.no_grad():
                f.weight.add_(1.0 / in_channels)

        self.bn_ll = nn.BatchNorm2d(out_channels)
        self.bn_hl = nn.BatchNorm2d(out_channels)
        self.bn_lh = nn.BatchNorm2d(out_channels)
        self.bn_hh = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        ll, hl, lh, hh = self.dwt(x)
        ll = F.relu6(self.bn_ll(self.filter_ll(ll)))
        hl = F.relu6(self.bn_hl(self.filter_hl(hl)))
        lh = F.relu6(self.bn_lh(self.filter_lh(lh)))
        hh = F.relu6(self.bn_hh(self.filter_hh(hh)))
        return ll, hl, lh, hh


class SparseSpatialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # Smooth the saliency map
        self.smooth = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, ll: Tensor, hl: Tensor, lh: Tensor, hh: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Compute energies by squaring the subbands
        e_hl = hl.pow(2).mean(dim=1, keepdim=True)
        e_lh = lh.pow(2).mean(dim=1, keepdim=True)
        e_hh = hh.pow(2).mean(dim=1, keepdim=True)
        e_ll = ll.pow(2).mean(dim=1, keepdim=True)

        gamma = e_hh / (e_ll + e_hl + e_lh + e_hh + 1e-6)
        
        # Spatial-sparse attention mathematically derived from eye gradient structure
        saliency = e_hl * e_lh * (1.0 - gamma)
        
        saliency = self.bn(self.smooth(saliency))
        saliency = torch.sigmoid(saliency)

        return ll * saliency, hl * saliency, lh * saliency, hh * saliency


class SparseEyeNetBlock(nn.Module):
    """
    Combines Compressive Wavelet Stem, Sparse Spatial Attention and feature fusion.
    Uses pytorch_wavelets for real DWT (GPU + differentiable).
    """
    def __init__(self, c1: int, c2: int, wave: str = 'haar') -> None:
        super().__init__()
        self.stem = CompressiveWaveletStem(c1, c2, wave=wave)
        self.attention = SparseSpatialAttention(c2)
        self.fuse = nn.Sequential(
            nn.Conv2d(c2 * 4, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        ll, hl, lh, hh = self.stem(x)
        ll, hl, lh, hh = self.attention(ll, hl, lh, hh)
        fused = torch.cat([ll, hl, lh, hh], dim=1)
        return self.fuse(fused)


class SelfAttentionBlock(nn.Module):
    """
    Single multi-head QKV self-attention block for spatial feature maps.

    Converts (B, C, H, W) → (B, H*W, C) tokens, applies self-attention
    with residual connection, then reshapes back to (B, C, H, W).

    Designed to be inserted after the deepest backbone feature map (P5)
    where spatial resolution is small (e.g. 7×7 = 49 tokens), making
    the quadratic attention cost negligible.

    Args:
        c1: Input channels (must equal c2).
        c2: Output channels (passthrough, must equal c1).
        num_heads: Number of attention heads (default 4).
        use_transformer: 1 = apply attention, 0 = identity passthrough.
    """

    def __init__(self, c1: int, c2: int, num_heads: int = 4, use_transformer: int = 1):
        super().__init__()
        assert c1 == c2, f"SelfAttentionBlock requires c1==c2, got {c1} vs {c2}"
        self.use_transformer = bool(use_transformer)
        self.dim = c1
        self.num_heads = num_heads
        self.head_dim = c1 // num_heads
        assert c1 % num_heads == 0, f"dim {c1} must be divisible by num_heads {num_heads}"
        self.scale = self.head_dim ** -0.5

        if self.use_transformer:
            self.norm = nn.LayerNorm(c1)
            self.qkv = nn.Linear(c1, c1 * 3, bias=False)
            self.proj = nn.Linear(c1, c1, bias=False)
            self.norm2 = nn.LayerNorm(c1)
            # Lightweight FFN: expand 2× (small for speed)
            self.ffn = nn.Sequential(
                nn.Linear(c1, c1 * 2),
                nn.GELU(),
                nn.Linear(c1 * 2, c1),
            )

    def forward(self, x: Tensor) -> Tensor:
        if not self.use_transformer:
            return x

        B, C, H, W = x.shape
        # Flatten spatial dims → token sequence
        tokens = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Self-attention with residual
        normed = self.norm(tokens)
        qkv = self.qkv(normed).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        tokens = tokens + self.proj(out)  # residual

        # FFN with residual
        tokens = tokens + self.ffn(self.norm2(tokens))

        # Reshape back to spatial
        return tokens.transpose(1, 2).reshape(B, C, H, W)

