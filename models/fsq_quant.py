"""
NCM Tring: Finite Scalar Quantization (FSQ) Engine.
Eliminates codebook collapse, straight-through estimator divergence,
and EMA dead-code restarts with 100% analytical codebook utilization.
"""

import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.quant import VectorQuantizerBaseClass, _build_quant_resi


def default_fsq_levels(vocab_size: int) -> List[int]:
    """
    Selects optimal factorized dimensions for a given target vocabulary size.
    For standard vocabulary sizes (e.g. 512, 1024, 4096), selects balanced odd/even levels.
    """
    candidates = {
        512: [8, 8, 8],
        1000: [8, 5, 5, 5],
        1024: [8, 8, 4, 4],
        2048: [8, 8, 8, 4],
        4096: [8, 8, 8, 8],
    }
    if vocab_size in candidates:
        return candidates[vocab_size]

    # Dynamic factorizer aiming for small dimensional vector
    dim = 4
    base = int(round(vocab_size ** (1.0 / dim)))
    levels = [base] * dim
    prod = math.prod(levels)
    while prod < vocab_size:
        levels[0] += 1
        prod = math.prod(levels)
    return levels


class FSQLevel(nn.Module):
    """
    Analytical Finite Scalar Quantization level.
    Maps continuous embeddings directly into discrete scalar grids without distance searches.
    """
    def __init__(self, embed_dim: int, levels: List[int]):
        super().__init__()
        self.embed_dim = embed_dim
        self.levels = [int(l) for l in levels]
        self.dim = len(self.levels)
        self.codebook_size = math.prod(self.levels)

        self.project_in = nn.Linear(embed_dim, self.dim, bias=False)
        self.project_out = nn.Linear(self.dim, embed_dim, bias=False)

        # Precompute basis for fast coordinate-to-index conversion
        basis = []
        curr = 1
        for l in reversed(self.levels):
            basis.append(curr)
            curr *= l
        basis = list(reversed(basis))
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.long), persistent=False)

        half_levels = [(l - 1) / 2.0 for l in self.levels]
        self.register_buffer("half_levels", torch.tensor(half_levels, dtype=torch.float32), persistent=False)
        self.register_buffer("levels_tensor", torch.tensor(self.levels, dtype=torch.float32), persistent=False)

        # Precompute analytical explicit codebook for downstream compatibility with tied heads and embeddings
        self._init_codebook_table()

    def _init_codebook_table(self):
        with torch.no_grad():
            indices = torch.arange(self.codebook_size, dtype=torch.long)
            coords = []
            rem = indices
            for b in self.basis:
                val = torch.div(rem, b, rounding_mode="floor")
                coords.append(val)
                rem = rem % b
            coords = torch.stack(coords, dim=-1).float()  # [V, d]
            q_coords = coords - self.half_levels
            table = self.project_out(q_coords)
            self.codebook_table = nn.Parameter(table, requires_grad=False)

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable bounding and rounding for arbitrary levels.
        z: [..., dim]
        Returns:
            z_ste: quantized continuous vector [..., dim]
            indices: discrete codes [..., 1]
        """
        L = self.levels_tensor.to(z.device)
        half_l = (L - 1.0) / 2.0
        z_norm = (torch.tanh(z) + 1.0) / 2.0
        z_scaled = z_norm * (L - 1.0)
        s = torch.round(z_scaled).clamp(min=0.0)
        s = torch.min(s, L - 1.0)

        # Straight-through estimator
        s_ste = (s - z_scaled).detach() + z_scaled
        z_ste = s_ste - half_l

        indices = (s.long() * self.basis.to(z.device)).sum(dim=-1)
        return z_ste, indices

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decodes flat indices directly into quantized vectors in embed_dim space.
        """
        coords = []
        rem = indices
        for b in self.basis:
            val = torch.div(rem, b, rounding_mode="floor")
            coords.append(val)
            rem = rem % b
        coords = torch.stack(coords, dim=-1).float()
        half_l = (self.levels_tensor.to(indices.device) - 1.0) / 2.0
        q_coords = coords - half_l
        return self.project_out(q_coords)

    def forward(self, x: torch.Tensor):
        if torch.is_floating_point(x):
            z = self.project_in(x)
            z_q, indices = self.quantize(z)
            out = self.project_out(z_q)
            return out, indices
        else:
            return self.indices_to_codes(x)

    @property
    def weight(self):
        """Emulate embedding weight for tied heads."""
        return self.codebook_table


class MultiscaleResidualFSQ(VectorQuantizerBaseClass):
    """
    NCM Tring: Multi-Scale Residual Finite Scalar Quantizer.
    Completely replaces standard VQ with hierarchical residual FSQ.
    """
    def __init__(
        self,
        vqvae_vocab_size,
        embed_dim: int,
        levels: List[int],
        fsq_levels: Optional[List[int]] = None,
        quant_resi=None,
        commitment_cost: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.levels = [int(l) for l in levels]
        self.num_levels = len(self.levels)
        self.commitment_cost = commitment_cost

        # Determine vocabulary per level
        if isinstance(vqvae_vocab_size, (list, tuple)):
            self.vqvae_vocab_sizes = [int(v) for v in vqvae_vocab_size]
        else:
            self.vqvae_vocab_sizes = [int(vqvae_vocab_size)] * self.num_levels

        # Build FSQ level modules
        self.fsq_modules = nn.ModuleList()
        for i in range(self.num_levels):
            v_size = self.vqvae_vocab_sizes[i]
            lvl_dims = fsq_levels if fsq_levels is not None else default_fsq_levels(v_size)
            fsq_mod = FSQLevel(embed_dim, lvl_dims)
            self.fsq_modules.append(fsq_mod)
            # Update actual codebook size
            self.vqvae_vocab_sizes[i] = fsq_mod.codebook_size

        # Embeddings alias for compatibility with NextConceptTransformer
        self.embeddings = self.fsq_modules

        self.quant_resi = _build_quant_resi(embed_dim, self.num_levels, quant_resi)
        self.quant_resi_enabled = self.quant_resi is not None
        self._qr_num_levels = self.num_levels

        # Codebook hit tracking
        for i, vs in enumerate(self.vqvae_vocab_sizes):
            self.register_buffer(f'codebook_hit_{i}', torch.zeros(vs, dtype=torch.long), persistent=False)

    def _apply_quant_resi(self, h_BLC: torch.Tensor, level_idx: int) -> torch.Tensor:
        if not self.quant_resi_enabled:
            return h_BLC
        h_BCL = h_BLC.transpose(-1, -2).contiguous()
        at = level_idx / max(self._qr_num_levels - 1, 1)
        h_BCL = self.quant_resi[at](h_BCL)
        return h_BCL.transpose(-1, -2).contiguous()

    def _down_for_quant(self, z_BLC: torch.Tensor, target_len: int) -> torch.Tensor:
        z_BCL = z_BLC.transpose(-1, -2)
        zd_BCL = F.interpolate(z_BCL, size=(target_len,), mode="area")
        return zd_BCL.transpose(-1, -2).contiguous()

    def _up_to_full(self, q_BLC: torch.Tensor, target_len: int) -> torch.Tensor:
        q_BCL = q_BLC.transpose(-1, -2)
        qu_BCL = F.interpolate(q_BCL, size=(target_len,), mode="linear")
        return qu_BCL.transpose(-1, -2).contiguous()

    def forward(self, z: torch.Tensor):
        B, L, C = z.shape
        z_rest = z.clone()
        z_hat = torch.zeros_like(z)

        total_loss = 0.0

        for i, level in enumerate(self.levels):
            fsq_mod = self.fsq_modules[i]
            if i != self.num_levels - 1:
                cur_input = self._down_for_quant(z_rest, level)
            else:
                cur_input = z_rest

            q_BLC, indices = fsq_mod(cur_input)

            if i != self.num_levels - 1:
                quantized = self._up_to_full(q_BLC, L)
            else:
                quantized = q_BLC

            quantized = self._apply_quant_resi(quantized, i)

            z_hat = z_hat + quantized
            z_rest = z_rest - quantized.detach()

            # Soft commitment regularization for representation alignment
            total_loss = total_loss + self.commitment_cost * F.mse_loss(z_hat, z.detach())
            getattr(self, f'codebook_hit_{i}')[indices.view(-1)] = 1

        total_loss = total_loss / self.num_levels
        # Straight-through identity from z to z_hat
        z_hat = z + (z_hat - z).detach()

        per_level_util = [
            torch.sum(getattr(self, f'codebook_hit_{i}')).item() / self.vqvae_vocab_sizes[i]
            for i in range(self.num_levels)
        ]
        global_util = sum(per_level_util) / len(per_level_util)
        return z_hat, total_loss, per_level_util, global_util, 0.0

    @torch.no_grad()
    def z_to_idx(self, z: torch.Tensor, return_zhats: bool = False, return_zlevels: bool = False):
        B, L, C = z.shape
        z_rest = z.clone()
        z_hat = torch.zeros_like(z)

        z_hats = []
        z_levels = []
        idx = []

        for i, level in enumerate(self.levels):
            fsq_mod = self.fsq_modules[i]
            if i != self.num_levels - 1:
                cur_input = self._down_for_quant(z_rest, level)
            else:
                cur_input = z_rest

            q_BLC, indices = fsq_mod(cur_input)
            idx.append(indices)

            if i != self.num_levels - 1:
                quantized = self._up_to_full(q_BLC, L)
            else:
                quantized = q_BLC

            quantized = self._apply_quant_resi(quantized, i)
            z_hat = z_hat + quantized
            z_rest = z_rest - quantized
            z_hats.append(z_hat)
            z_levels.append(
                F.interpolate(z_hat.transpose(-1, -2), size=(level,), mode="area").transpose(-1, -2)
                if i != self.num_levels - 1 else z_hat
            )

        idx = torch.cat(idx, dim=-1)
        response = (idx,)
        if return_zhats:
            response = response + (z_hats,)
        if return_zlevels:
            response = response + (z_levels,)
        return idx if not (return_zhats or return_zlevels) else response

    @torch.no_grad()
    def idx_to_zhats(self, idx: torch.Tensor, return_zlevels: bool = True):
        z_hats = []
        z_levels = []
        z_hat = torch.zeros(idx.shape[0], self.levels[-1], self.embed_dim, device=idx.device)

        level_accum = 0
        for level_idx in range(self.num_levels):
            fsq_mod = self.fsq_modules[level_idx]
            sub_idx = idx[:, level_accum:level_accum + self.levels[level_idx]]
            emb_at_level = fsq_mod.indices_to_codes(sub_idx)

            if level_idx != self.num_levels - 1:
                z_q_level = self._up_to_full(emb_at_level, self.levels[-1])
            else:
                z_q_level = emb_at_level

            z_q_level = self._apply_quant_resi(z_q_level, level_idx)
            z_hat.add_(z_q_level)
            z_hats.append(z_hat)

            if return_zlevels:
                z_levels.append(
                    F.interpolate(z_hat.transpose(-1, -2), size=(self.levels[level_idx],), mode="area").transpose(-1, -2)
                    if level_idx != self.num_levels - 1 else z_hat
                )
            level_accum += self.levels[level_idx]

        results = (z_hats,)
        if return_zlevels:
            results = results + (z_levels,)
        return results

    @torch.no_grad()
    def idx_to_z_q(self, indices: torch.Tensor) -> torch.Tensor:
        z_hat = torch.zeros(indices.shape[0], self.levels[-1], self.embed_dim, device=indices.device)
        level_accum = 0
        for level_idx in range(self.num_levels):
            fsq_mod = self.fsq_modules[level_idx]
            sub_idx = indices[:, level_accum:level_accum + self.levels[level_idx]]
            emb_at_level = fsq_mod.indices_to_codes(sub_idx)

            if level_idx != self.num_levels - 1:
                z_q_level = self._up_to_full(emb_at_level, self.levels[-1])
            else:
                z_q_level = emb_at_level

            z_q_level = self._apply_quant_resi(z_q_level, level_idx)
            z_hat.add_(z_q_level)
            level_accum += self.levels[level_idx]
        return z_hat

    @torch.no_grad()
    def idx_to_ncm_input(self, idx: torch.Tensor) -> torch.Tensor:
        next_concepts = []
        z_hat = torch.zeros(idx.shape[0], self.levels[-1], self.embed_dim, device=idx.device)
        level_accum = 0
        for level_idx in range(self.num_levels - 1):
            fsq_mod = self.fsq_modules[level_idx]
            sub_idx = idx[:, level_accum:level_accum + self.levels[level_idx]]
            emb_at_level = fsq_mod.indices_to_codes(sub_idx)

            z_q_level = self._up_to_full(emb_at_level, self.levels[-1])
            z_q_level = self._apply_quant_resi(z_q_level, level_idx)
            z_hat.add_(z_q_level)
            next_concepts.append(
                F.interpolate(
                    z_hat.transpose(-1, -2), size=(self.levels[level_idx + 1],), mode="area"
                ).transpose(-1, -2)
            )
            level_accum += self.levels[level_idx]
        return torch.cat(next_concepts, dim=1)

    @torch.no_grad()
    def get_next_autoregressive_input(self, level_idx: int, z_hat: torch.Tensor, z_q_level: torch.Tensor):
        if level_idx != self.num_levels - 1:
            upsampled = self._up_to_full(z_q_level, self.levels[-1])
            upsampled = self._apply_quant_resi(upsampled, level_idx)
            z_hat.add_(upsampled)
            next_input = F.interpolate(
                z_hat.transpose(-1, -2), size=(self.levels[level_idx + 1],), mode="area"
            ).transpose(-1, -2)
            return z_hat, next_input
        else:
            z_q_level = self._apply_quant_resi(z_q_level, level_idx)
            z_hat.add_(z_q_level)
            return z_hat, z_hat
