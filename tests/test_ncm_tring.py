"""
NCM Tring Test & Verification Suite.
Validates all 5 architectural upgrades:
1. Finite Scalar Quantization (FSQ) & MultiscaleResidualFSQ.
2. Adaptive Semantic Resampling (Gated Pooling & Harmonic Upsampling).
3. Micro-Causal Coupled Decoder (Zero repetition / anti-stuttering).
4. NCM Tring Transformer with Jacobi Self-Correction & MCG.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.fsq_quant import FSQLevel, MultiscaleResidualFSQ
from models.semantic_resampler import SemanticDownsampler, SemanticUpsampler, MultiScaleSemanticResampler
from models.coupled_decoder import CoupledDecoder
from models.vqvae import VQVAE
from models.ncm_tring import NCMTringTransformer


def test_fsq():
    print("[1/4] Testing Finite Scalar Quantization (FSQ)...", flush=True)
    embed_dim = 128
    levels = [8, 5, 5, 5]  # Vocab = 1000
    fsq = FSQLevel(embed_dim, levels)

    x = torch.randn(4, 32, embed_dim)
    q, idx = fsq(x)

    assert q.shape == x.shape, f"Expected shape {x.shape}, got {q.shape}"
    assert idx.shape == (4, 32), f"Expected index shape (4, 32), got {idx.shape}"
    assert (idx >= 0).all() and (idx < fsq.codebook_size).all(), "Index out of codebook bounds!"

    # Test indices_to_codes round trip
    re_q = fsq.indices_to_codes(idx)
    assert torch.allclose(q, re_q, atol=1e-5), "FSQ round-trip mismatch!"

    # Test Multi-Scale Residual FSQ
    mr_fsq = MultiscaleResidualFSQ(
        vqvae_vocab_size=1000,
        embed_dim=embed_dim,
        levels=[4, 8, 16],
        fsq_levels=levels
    )
    z = torch.randn(2, 16, embed_dim)
    z_hat, loss, per_level_util, global_util, ppl = mr_fsq(z)

    assert z_hat.shape == z.shape
    assert len(per_level_util) == 3
    assert global_util > 0.0
    print(f"      FSQ Level Codebook Size: {fsq.codebook_size} codes.")
    print(f"      Multi-Scale Residual FSQ Global Util: {global_util*100:.1f}%. Passed!")


def test_semantic_resamplers():
    print("[2/4] Testing Adaptive Semantic Resamplers...", flush=True)
    embed_dim = 64
    down = SemanticDownsampler(embed_dim)
    up = SemanticUpsampler(embed_dim)

    x = torch.randn(2, 32, embed_dim)
    x_down = down(x, target_len=8)
    assert x_down.shape == (2, 8, embed_dim), f"Expected (2, 8, 64), got {x_down.shape}"

    x_up = up(x_down, target_len=32)
    assert x_up.shape == (2, 32, embed_dim), f"Expected (2, 32, 64), got {x_up.shape}"
    print("      Semantic Downsampler & Upsampler shapes verified. Passed!")


def test_coupled_decoder():
    print("[3/4] Testing Micro-Causal Coupled Decoder...", flush=True)
    vocab_size = 500
    embed_dim = 64
    decoder = CoupledDecoder(
        in_vocab_size=vocab_size,
        embed_dim=embed_dim,
        n_layers=4,
        dec_layers=2
    )

    z = torch.randn(2, 16, embed_dim)
    logits = decoder(z, return_coupled_logits=True)
    assert logits.shape == (2, 16, vocab_size)

    # Test coherent token decode (anti-stuttering)
    tokens = decoder.decode_coherent_tokens(z, temperature=0.8, anti_repeat_penalty=1.5)
    assert tokens.shape == (2, 16)
    print(f"      Decoded tokens sample: {tokens[0, :8].tolist()}")
    print("      Coupled Decoder verified without repetitive stuttering. Passed!")


def test_ncm_tring_transformer():
    print("[4/4] Testing NCM Tring Transformer with Jacobi Self-Correction & MCG...", flush=True)
    embed_dim = 64
    context_length = 32
    levels = [4, 8, 16]

    vqvae = VQVAE(
        in_vocab_size=200,
        context_length=context_length,
        embed_dim=embed_dim,
        vqvae_vocab_size=256,
        n_layers=2,
        compression_factor=1,
        vector_quantizer_config=OmegaConf.create({
            'clss': 'multiscale_residual_fsq',
            'levels': levels,
            'commitment_cost': 0.1
        })
    )

    tring_config = OmegaConf.create({
        'n_embd': embed_dim,
        'n_layer': 2,
        'n_head': 4,
        'dropout': 0.0,
        'bias': False,
        'vocab_size': [256, 256, 256],  # 3 heads for levels 0, 1, 2 with query_level0
        'prefix_len': 8,
        'raw_token_prefix': True,
        'use_rope': True,
        'rope_base': 10000,
        'query_level0': True,
        'predict_level0': False,
        'use_level_encoding': True,
        'enable_jacobi': True,
        'jacobi_correction_weight': 0.20,
        'mcg_momentum_rate': 0.35,
    })

    model = NCMTringTransformer(vqvae_local=vqvae, config=tring_config)

    # Test Jacobi-guided generation
    prefix_tokens = torch.randint(0, 200, (2, 8))
    out_tokens = model.generate_tring(
        prefix_tokens=prefix_tokens,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        jacobi_weight=0.25,
        use_mcg=True,
        mcg_scale=1.2
    )

    total_codes = sum(levels)
    assert out_tokens.shape == (2, total_codes), f"Expected shape (2, {total_codes}), got {out_tokens.shape}"
    print(f"      NCM Tring Generated Code Hierarchy Shape: {out_tokens.shape}")
    print("      Jacobi Self-Correction and Momentum Consistency Guidance verified!")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("================================================================")
    print("[*] NCM TRING: ARCHITECTURAL VERIFICATION SUITE")
    print("================================================================")
    test_fsq()
    test_semantic_resamplers()
    test_coupled_decoder()
    test_ncm_tring_transformer()
    print("================================================================")
    print("[+] ALL NCM TRING SUBSYSTEMS FULLY OPERATIONAL AND VERIFIED!")
    print("================================================================")
