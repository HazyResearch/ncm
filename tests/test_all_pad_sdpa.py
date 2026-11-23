import os
import sys

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.next_concept import NextConceptTransformer  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402


def _build_model(
    *,
    query_level0=False,
    predict_level0=False,
    l0_autoregressive=False,
    use_ctx_encoder=False,
    use_rope=True,
    dropout=0.1,
    prefix_len=6,
    levels=(2, 3, 4),
    vocab_size=16,
    in_vocab_size=32,
    prefix_pad_token_id=0,
):
    torch.manual_seed(42)
    embed_dim = 16
    levels = list(levels)
    vqvae_cfg = OmegaConf.create(
        {
            "clss": "multiscale_residual_vector_quantizer",
            "decay": 0.99,
            "epsilon": 1e-5,
            "commitment_cost": 0.25,
            "quant_resi": {"enabled": True, "ratio": 0.5, "share_mode": 0},
            "levels": {"use_manual_levels": True, "manual_levels": levels},
        }
    )
    vqvae = VQVAE(
        in_vocab_size=in_vocab_size,
        context_length=max(prefix_len, sum(levels)),
        embed_dim=embed_dim,
        vqvae_vocab_size=[vocab_size] * len(levels),
        n_layers=2,
        vector_quantizer_config=vqvae_cfg,
        compression_factor=1,
        dropout=0.0,
    ).eval()

    n_heads_out = len(levels) if query_level0 else len(levels) - 1
    model_cfg = OmegaConf.create(
        {
            "n_layer": 2,
            "n_head": 4,
            "n_embd": 32,
            "bias": True,
            "dropout": dropout,
            "prefix_len": prefix_len,
            "prefix_pad_token_id": prefix_pad_token_id,
            "vocab_size": [vocab_size] * n_heads_out,
            "attn_pattern": "block_diagonal",
            "query_level0": query_level0,
            "l0_autoregressive": l0_autoregressive,
            "predict_level0": predict_level0,
            "use_level_encoding": True,
            "use_rope": use_rope,
            "rope_base": 10000,
            "use_qk_norm": True,
            "raw_token_prefix": False,
            "contextual_prefix": False,
            "use_flex_attention": False,
            "ctx_encoder": {
                "enabled": use_ctx_encoder,
                "n_layer": 1,
                "vocab_size": in_vocab_size,
            },
        }
    )
    return NextConceptTransformer(vqvae_local=vqvae, config=model_cfg)


def _forward_inputs(
    model,
    batch_size,
    prefix_len,
    seq_target_len,
    *,
    use_ctx_encoder=False,
    query_level0=False,
    l0_autoregressive=False,
    vocab_size=16,
):
    torch.manual_seed(123)
    embed_dim = model.vqvae_local[0].embed_dim
    if query_level0:
        l0 = model.levels[0]
        if l0_autoregressive:
            l0_idx = torch.randint(0, vocab_size, (batch_size, l0))
            l0_in = model.build_l0_ar_input(l0_idx)
        else:
            l0_in = model.build_l0_query_input(batch_size)
        rest = torch.randn(batch_size, seq_target_len - l0, embed_dim)
        x_target = torch.cat([l0_in, rest], dim=1)
    else:
        x_target = torch.randn(batch_size, seq_target_len, embed_dim)

    if use_ctx_encoder:
        x = x_target
        prefix_ids = torch.randint(1, 32, (batch_size, prefix_len))
    else:
        x_prefix = torch.randn(batch_size, prefix_len, embed_dim)
        x = torch.cat([x_prefix, x_target], dim=1)
        prefix_ids = None
    return x, prefix_ids


@pytest.fixture
def assert_no_all_masked_sdpa_rows(monkeypatch):
    orig_sdpa = torch.nn.functional.scaled_dot_product_attention

    def checked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                assert attn_mask.any(dim=-1).all(), "SDPA bool mask has an all-False query row"
            else:
                assert torch.isfinite(attn_mask).any(dim=-1).all(), (
                    "SDPA additive mask has an all--inf query row"
                )
        return orig_sdpa(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs
        )

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", checked_sdpa)


@pytest.mark.parametrize(
    "query_level0,predict_level0,l0_autoregressive,use_ctx_encoder",
    [
        (True, False, False, False),
        (True, False, True, False),
        (False, True, False, False),
        (False, False, False, False),
        (True, False, False, True),
        (False, True, False, True),
    ],
)
def test_train_forward_backward_finite_with_all_pad_row(
    assert_no_all_masked_sdpa_rows,
    query_level0,
    predict_level0,
    l0_autoregressive,
    use_ctx_encoder,
):
    prefix_len = 6
    levels = (2, 3, 4)
    model = _build_model(
        query_level0=query_level0,
        predict_level0=predict_level0,
        l0_autoregressive=l0_autoregressive,
        use_ctx_encoder=use_ctx_encoder,
        dropout=0.1,
        prefix_len=prefix_len,
        levels=levels,
    )
    model.train()

    target_len = sum(levels) if query_level0 else sum(levels[1:])
    x, prefix_ids = _forward_inputs(
        model,
        3,
        prefix_len,
        target_len,
        use_ctx_encoder=use_ctx_encoder,
        query_level0=query_level0,
        l0_autoregressive=l0_autoregressive,
    )
    if use_ctx_encoder:
        prefix_ids[1, 3:] = 0
        prefix_ids[2, :] = 0

    # Row 0: unpadded; Row 1: partially padded; Row 2: all-pad prefix.
    prefix_pad_mask = torch.tensor(
        [
            [False, False, False, False, False, False],
            [False, False, False, True, True, True],
            [True, True, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    logits = model(x, prefix_ids=prefix_ids, prefix_pad_mask=prefix_pad_mask)
    assert torch.isfinite(logits).all()

    targets = torch.randint(0, 16, (3, logits.shape[1]))
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )
    assert torch.isfinite(loss)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"missing gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"


@pytest.mark.parametrize(
    "query_level0,predict_level0",
    [
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_partially_padded_row_logits_match_alone_vs_with_all_pad_row(
    assert_no_all_masked_sdpa_rows, query_level0, predict_level0
):
    prefix_len = 6
    levels = (2, 3, 4)
    model = _build_model(
        query_level0=query_level0,
        predict_level0=predict_level0,
        dropout=0.0,
        prefix_len=prefix_len,
        levels=levels,
    )
    model.eval()

    target_len = sum(levels) if query_level0 else sum(levels[1:])
    x_single, _ = _forward_inputs(model, 1, prefix_len, target_len, query_level0=query_level0)
    x_other, _ = _forward_inputs(model, 1, prefix_len, target_len, query_level0=query_level0)
    x_mixed = torch.cat([x_single, x_other], dim=0)

    partial_mask = torch.tensor([[False, False, False, True, True, True]], dtype=torch.bool)
    all_pad_mask = torch.ones(1, prefix_len, dtype=torch.bool)
    mixed_mask = torch.cat([partial_mask, all_pad_mask], dim=0)

    with torch.no_grad():
        logits_alone = model(x_single, prefix_pad_mask=partial_mask)
        logits_mixed = model(x_mixed, prefix_pad_mask=mixed_mask)

    assert torch.isfinite(logits_mixed).all()
    assert torch.allclose(logits_mixed[:1], logits_alone, atol=1e-6, rtol=0.0)


def test_level0_slice_handles_all_pad_prefix_mask():
    model = _build_model(query_level0=False, predict_level0=True, prefix_len=6, levels=(2, 3, 4))
    hidden = torch.randn(3, 6, 32)
    mask = torch.tensor(
        [
            [False, False, False, False, False, False],
            [False, False, False, True, True, True],
            [True, True, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    sliced = model._level0_slice(hidden, mask)
    assert sliced.shape == (3, 2, 32)
    assert torch.isfinite(sliced).all()
    assert torch.equal(sliced[2], hidden[2, :2])


@pytest.mark.parametrize("use_kv_cache", [True, False])
@pytest.mark.parametrize(
    "query_level0,predict_level0,l0_autoregressive,supply_idx,cfg_scale",
    [
        (True, False, False, False, 0.0),
        (True, False, True, False, 0.0),
        (True, False, False, False, 1.5),
        (False, True, False, False, 0.0),
        (False, True, False, False, 1.5),
        (False, False, False, True, 0.0),
    ],
)
def test_generate_finite_with_all_pad_row(
    assert_no_all_masked_sdpa_rows,
    use_kv_cache,
    query_level0,
    predict_level0,
    l0_autoregressive,
    supply_idx,
    cfg_scale,
):
    prefix_len = 6
    levels = (2, 3, 4)
    model = _build_model(
        query_level0=query_level0,
        predict_level0=predict_level0,
        l0_autoregressive=l0_autoregressive,
        dropout=0.0,
        prefix_len=prefix_len,
        levels=levels,
        prefix_pad_token_id=0,
    )
    model.eval()

    prefix_tokens = torch.tensor(
        [
            [3, 4, 5, 6, 7, 8],
            [3, 4, 5, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    idx = torch.randint(0, 16, (3, levels[0])) if supply_idx else None

    torch.manual_seed(7)
    tokens, z_hat = model.generate(
        idx,
        prefix_tokens,
        temperature=1.0,
        top_k=4,
        top_p=0.9,
        use_kv_cache=use_kv_cache,
        cfg_scale=cfg_scale,
        cfg_null_eos=1,
        return_latent=True,
    )
    assert tokens.shape == (3, sum(levels))
    assert torch.isfinite(z_hat).all()


@pytest.mark.parametrize("use_kv_cache", [True, False])
@pytest.mark.parametrize("query_level0,predict_level0", [(True, False), (False, True)])
def test_generate_partially_padded_row_matches_alone_vs_with_all_pad_row(
    assert_no_all_masked_sdpa_rows, use_kv_cache, query_level0, predict_level0
):
    prefix_len = 6
    levels = (2, 3, 4)
    model = _build_model(
        query_level0=query_level0,
        predict_level0=predict_level0,
        dropout=0.0,
        prefix_len=prefix_len,
        levels=levels,
        prefix_pad_token_id=0,
    )
    model.eval()

    partial_prefix = torch.tensor([[3, 4, 5, 0, 0, 0]], dtype=torch.long)
    all_pad_prefix = torch.zeros(1, prefix_len, dtype=torch.long)
    mixed_prefix = torch.cat([partial_prefix, all_pad_prefix], dim=0)

    tokens_alone, z_alone = model.generate(
        None,
        partial_prefix,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        use_kv_cache=use_kv_cache,
        return_latent=True,
    )
    tokens_mixed, z_mixed = model.generate(
        None,
        mixed_prefix,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        use_kv_cache=use_kv_cache,
        return_latent=True,
    )

    assert torch.isfinite(z_mixed).all()
    assert torch.equal(tokens_mixed[:1], tokens_alone)
    assert torch.allclose(z_mixed[:1], z_alone, atol=1e-6, rtol=0.0)
