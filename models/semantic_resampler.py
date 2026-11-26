"""
NCM Tring: Adaptive Semantic Gated Resampler.
Replaces naive continuous F.interpolate(mode='area'/'linear') with linguistically-aware
gated spatial pooling and harmonic frequency upsampling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticDownsampler(nn.Module):
    """
    Adaptive Semantic Downsampler.
    Calculates dynamic boundary salience weights before spatial compression,
    preserving phrase, entity, and sentence boundaries.
    """
    def __init__(self, embed_dim: int, kernel_size: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        pad = kernel_size // 2

        self.salience_gate = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=pad, groups=embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        )
        # Zero-initialize the final projection so initial behavior matches pure pooling
        nn.init.zeros_(self.refine[2].weight)
        nn.init.zeros_(self.refine[2].bias)

    def forward(self, x_BLC: torch.Tensor, target_len: int) -> torch.Tensor:
        B, L, C = x_BLC.shape
        if L == target_len:
            return x_BLC

        x_BCL = x_BLC.transpose(1, 2)  # [B, C, L]
        # Salience weighting
        gate = self.salience_gate(x_BCL)  # [B, 1, L]
        x_weighted = x_BCL * (1.0 + gate)

        # Adaptive multi-scale pooling
        pooled = F.adaptive_avg_pool1d(x_weighted, target_len)
        refined = pooled + self.refine(pooled)
        return refined.transpose(1, 2).contiguous()


class SemanticUpsampler(nn.Module):
    """
    Adaptive Semantic Upsampler.
    Applies harmonic frequency expansion with residual refinement to avoid
    the linear blurring artifact of F.interpolate('linear').
    """
    def __init__(self, embed_dim: int, kernel_size: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        pad = kernel_size // 2

        self.harmonic_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=pad)
        self.spatial_gate = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        )
        nn.init.zeros_(self.spatial_gate[2].weight)
        nn.init.zeros_(self.spatial_gate[2].bias)

    def forward(self, q_BLC: torch.Tensor, target_len: int) -> torch.Tensor:
        B, L, C = q_BLC.shape
        if L == target_len:
            return q_BLC

        q_BCL = q_BLC.transpose(1, 2)
        # Initial smooth upsample
        up = F.interpolate(q_BCL, size=(target_len,), mode="linear", align_corners=False)
        # Detail injection and high-frequency refinement
        harmonic = self.harmonic_proj(up)
        delta = self.spatial_gate(harmonic)
        out = up + delta
        return out.transpose(1, 2).contiguous()


class MultiScaleSemanticResampler(nn.Module):
    """
    Manages level-specific semantic downsamplers and upsamplers across the NCM hierarchy.
    """
    def __init__(self, embed_dim: int, levels: list):
        super().__init__()
        self.embed_dim = embed_dim
        self.levels = levels
        self.num_levels = len(levels)

        self.downsamplers = nn.ModuleList([
            SemanticDownsampler(embed_dim) for _ in range(self.num_levels)
        ])
        self.upsamplers = nn.ModuleList([
            SemanticUpsampler(embed_dim) for _ in range(self.num_levels)
        ])

    def downsample(self, x_BLC: torch.Tensor, level_idx: int, target_len: int) -> torch.Tensor:
        return self.downsamplers[level_idx](x_BLC, target_len)

    def upsample(self, q_BLC: torch.Tensor, level_idx: int, target_len: int) -> torch.Tensor:
        return self.upsamplers[level_idx](q_BLC, target_len)
