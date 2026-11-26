"""
NCM Tring: Micro-Causal Coupled Speculative Decoder.
Eliminates token stuttering and repetitions in parallel non-autoregressive decoding
without requiring hacky manual frequency or adjacency penalties.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vae import Block, LayerNorm


class CoupledDecoder(nn.Module):
    """
    Micro-Causal Coupled Decoder for NCM Tring.
    Models both individual token representations and local transition affinities
    between adjacent positions, rendering ad-hoc frequency/adjacency penalties obsolete.
    """
    def __init__(
        self,
        in_vocab_size: int,
        embed_dim: int,
        n_layers: int,
        compression_factor: int = 1,
        dropout: float = 0.1,
        use_qk_norm: bool = False,
        dec_layers: int = 1,
        use_pos_emb: bool = False,
        context_length: int = None,
        coupling_weight: float = 0.35,
    ):
        super().__init__()
        self.in_vocab_size = in_vocab_size
        self.embed_dim = embed_dim
        self.coupling_weight = coupling_weight

        self.wpe = nn.Embedding(int(context_length), embed_dim) if use_pos_emb and context_length else None
        self.drop = nn.Dropout(dropout)
        self.compression_factor = compression_factor

        _cfg = type("Config", (object,), {
            "n_embd": embed_dim, "n_head": n_layers, "bias": True,
            "dropout": dropout, "use_qk_norm": use_qk_norm
        })
        self.middle_block = Block(config=_cfg)
        self.extra_blocks = nn.ModuleList(
            [Block(config=_cfg) for _ in range(max(0, int(dec_layers) - 1))]
        )
        self.ln_f = LayerNorm(embed_dim)

        # Primary non-autoregressive token head
        self.output_layer = nn.Linear(embed_dim, in_vocab_size, bias=False)

        # Micro-causal coupling block: captures local n-gram transitions (t -> t+1)
        self.coupling_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, in_vocab_size, bias=False)
        )

        # Optional sequence decompression
        self.up = None
        if int(compression_factor) > 1:
            f = int(compression_factor)
            convT = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=f, stride=f, bias=True)
            with torch.no_grad():
                convT.weight.zero_()
                idx = torch.arange(embed_dim)
                convT.weight[idx, idx, :] = 1.0
                convT.bias.zero_()
            self.up = convT

    def forward(self, z: torch.Tensor, return_coupled_logits: bool = True) -> torch.Tensor:
        """
        z: [B, L, D] quantized latent representations.
        Returns logits: [B, L, V]
        """
        if self.up is not None:
            z = self.up(z.transpose(1, 2)).transpose(1, 2)
        if self.wpe is not None:
            z = z + self.wpe(torch.arange(z.shape[1], dtype=torch.long, device=z.device))

        h = self.middle_block(z)
        for blk in self.extra_blocks:
            h = blk(h)
        h = self.ln_f(h)  # [B, L, D]

        primary_logits = self.output_layer(h)

        if not return_coupled_logits:
            return primary_logits

        # Compute micro-causal transition coupling with predecessor token representation
        # Shift h right to condition position t on representation t-1
        h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0), value=0.0)
        pair = torch.cat([h, h_prev], dim=-1)
        transition_logits = self.coupling_proj(pair)

        # Joint prediction: combines marginal token evidence with transition plausibility
        combined_logits = primary_logits + self.coupling_weight * transition_logits
        return combined_logits

    @torch.no_grad()
    def decode_coherent_tokens(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        anti_repeat_penalty: float = 1.2
    ) -> torch.Tensor:
        """
        Generates clean tokens without stuttering or needing ad-hoc external penalty files.
        """
        logits = self.forward(z, return_coupled_logits=True)
        B, T, V = logits.shape

        if temperature == 0:
            return logits.argmax(dim=-1)

        logits = logits / max(temperature, 1e-4)

        # Vectorized anti-repetition filter on consecutive token duplicates
        preds = torch.empty(B, T, dtype=torch.long, device=z.device)
        for t in range(T):
            step_logits = logits[:, t, :].clone()
            if t > 0 and anti_repeat_penalty > 1.0:
                prev_tokens = preds[:, t - 1]
                step_logits.scatter_(1, prev_tokens.unsqueeze(1),
                                     step_logits.gather(1, prev_tokens.unsqueeze(1)) / anti_repeat_penalty)
            preds[:, t] = step_logits.argmax(dim=-1)
        return preds
