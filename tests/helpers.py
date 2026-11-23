import contextlib
import io

import torch
from omegaconf import OmegaConf

from models.vqvae import VQVAE


def tiny_vqvae(*, learned=False, quant_resi=True, dec_positional=False):
    config = OmegaConf.create({
        "clss": "multiscale_residual_vector_quantizer",
        "levels": {"use_manual_levels": True, "manual_levels": [1, 2, 4, 8]},
        "learned_l1_sampling": learned,
        "learned_l1_cascade": learned,
        "quant_resi": {"enabled": quant_resi, "ratio": 0.5, "share_mode": 0},
    })
    with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
        torch.manual_seed(42)
        return VQVAE(
            in_vocab_size=17, context_length=8, embed_dim=8,
            vqvae_vocab_size=[8] * 4, n_layers=2,
            vector_quantizer_config=config, compression_factor=1,
            dropout=0.0, pre_quant_groupnorm=2,
            dec_positional=dec_positional,
        ).eval()
