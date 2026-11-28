import os
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.next_concept import NextConceptTransformer  # noqa: E402
from models.vqvae import VQVAE  # noqa: E402
from utils.generation import Generator  # noqa: E402


def _make_tiny_vqvae(levels=(2, 4, 8), vocab_size=16, embed_dim=8, in_vocab_size=32, context_length=8):
    return VQVAE(
        in_vocab_size=in_vocab_size,
        context_length=context_length,
        embed_dim=embed_dim,
        vqvae_vocab_size=[vocab_size for _ in levels],
        n_layers=1,
        dropout=0.0,
        vector_quantizer_config=OmegaConf.create({
            "clss": "multiscale_residual_vector_quantizer",
            "decay": 0.99,
            "commitment_cost": 0.25,
            "levels": {
                "use_manual_levels": True,
                "manual_levels": list(levels),
            },
        }),
    ).eval()


def _make_tiny_model(
    *,
    levels=(2, 4, 8),
    vocab_size=16,
    embed_dim=8,
    prefix_len=4,
    query_level0=True,
    l0_autoregressive=False,
    predict_level0=False,
    prefix_pad_token_id=0,
):
    vqvae = _make_tiny_vqvae(levels=levels, vocab_size=vocab_size, embed_dim=embed_dim)
    n_heads_target = len(levels) if query_level0 else len(levels) - 1
    cfg = OmegaConf.create({
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
        "dropout": 0.0,
        "bias": True,
        "vocab_size": [vocab_size] * n_heads_target,
        "prefix_len": prefix_len,
        "raw_token_prefix": True,
        "prefix_pad_token_id": prefix_pad_token_id,
        "query_level0": query_level0,
        "l0_autoregressive": l0_autoregressive,
        "predict_level0": predict_level0,
        "use_level_encoding": True,
        "use_rope": True,
        "rope_base": 10000,
        "use_qk_norm": True,
        "use_flex_attention": False,
        "attn_pattern": "block_diagonal",
    })
    model = NextConceptTransformer(vqvae_local=vqvae, config=cfg).eval()
    return model, vqvae


def test_sample_level_raw_clamps_top_k_above_vocab_size():
    model, _ = _make_tiny_model()
    logits = torch.randn(2, 4, 8)

    sampled = model._sample_level_raw(logits, temperature=1.0, top_k=50, top_p=0.9)

    assert sampled.shape == (2, 4)
    assert sampled.dtype == torch.long
    assert (sampled >= 0).all() and (sampled < 8).all()


@pytest.mark.parametrize("top_p", [0.0, 0.9, 1.0])
def test_sample_level_raw_top_k_one_uses_argmax_without_sort_or_multinomial(monkeypatch, top_p):
    model, _ = _make_tiny_model()
    logits = torch.tensor([[[0.1, 2.5, -0.3, 1.2], [4.0, 3.9, -1.0, 0.5]]])

    def _forbidden(*args, **kwargs):
        raise AssertionError("sampling fallback should not be called for greedy top_k=1")

    monkeypatch.setattr(torch, "topk", _forbidden)
    monkeypatch.setattr(torch, "sort", _forbidden)
    monkeypatch.setattr(torch, "multinomial", _forbidden)

    sampled = model._sample_level_raw(logits, temperature=1.0, top_k=1, top_p=top_p)

    assert torch.equal(sampled, logits.argmax(dim=-1))


def test_sample_level_raw_temperature_zero_returns_argmax(monkeypatch):
    model, _ = _make_tiny_model()
    logits = torch.tensor([[[0.2, 0.9, 0.4], [1.5, 1.4, -2.0]]])

    def _forbidden(*args, **kwargs):
        raise AssertionError("sort/multinomial should not be called when temperature == 0")

    monkeypatch.setattr(torch, "sort", _forbidden)
    monkeypatch.setattr(torch, "multinomial", _forbidden)

    sampled = model._sample_level_raw(logits, temperature=0.0, top_k=50, top_p=0.9)

    assert torch.equal(sampled, logits.argmax(dim=-1))


@pytest.mark.parametrize("l0_ar", [False, True])
def test_generate_soft_decisions_return_codes_false_skips_sampling_and_matches_latent(monkeypatch, l0_ar):
    model, _ = _make_tiny_model(query_level0=True, l0_autoregressive=l0_ar)
    prefix = torch.randint(1, 32, (2, 4))

    torch.manual_seed(123)
    codes_ref, z_q_ref = model.generate(
        None, prefix, temperature=1.0, top_k=8, top_p=0.9,
        soft_decisions=True, soft_temperature=0.7, return_latent=True,
    )

    def _forbid_sample(*args, **kwargs):
        raise AssertionError("_sample_level should not run when soft_decisions=True and return_codes=False")

    monkeypatch.setattr(model, "_sample_level", _forbid_sample)

    codes_skip, z_q_skip = model.generate(
        None, prefix, temperature=1.0, top_k=8, top_p=0.9,
        soft_decisions=True, soft_temperature=0.7, return_latent=True, return_codes=False,
    )

    assert codes_ref.shape == (2, sum(model.levels))
    assert codes_skip is None
    assert torch.allclose(z_q_skip, z_q_ref, atol=0.0, rtol=0.0)


def test_generate_soft_decisions_trajectory_skips_sampling_and_matches_baseline(monkeypatch):
    model, _ = _make_tiny_model(query_level0=True, l0_autoregressive=False)
    prefix = torch.randint(1, 32, (2, 4))

    torch.manual_seed(45)
    _, traj_ref = model.generate(
        None, prefix, temperature=1.0, top_k=8, top_p=0.9,
        soft_decisions=True, soft_temperature=1.0, return_latent=True, return_trajectory=True,
    )

    monkeypatch.setattr(
        model, "_sample_level",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("_sample_level called")),
    )

    codes_skip, traj_skip = model.generate(
        None, prefix, temperature=1.0, top_k=8, top_p=0.9,
        soft_decisions=True, soft_temperature=1.0, return_latent=True, return_trajectory=True,
        return_codes=False,
    )

    assert codes_skip is None
    assert traj_skip.shape == (2, len(model.levels), model.levels[-1], model.vqvae_local[0].embed_dim)
    assert torch.equal(traj_skip, traj_ref)


def test_generate_predict_level0_soft_decisions_skips_sampling_and_matches_latent(monkeypatch):
    model, _ = _make_tiny_model(query_level0=False, predict_level0=True)
    prefix = torch.randint(1, 32, (2, 4))

    torch.manual_seed(99)
    _, z_q_ref = model.generate(
        None, prefix, temperature=1.0, top_k=8, top_p=0.9,
        soft_decisions=True, soft_temperature=0.8, return_latent=True,
    )

    monkeypatch.setattr(
        model, "_sample_level",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("_sample_level called")),
    )

    codes_skip, z_q_skip = model.generate(
        None, prefix, temperature=1.0, top_k=8, top_p=0.9,
        soft_decisions=True, soft_temperature=0.8, return_latent=True, return_codes=False,
    )

    assert codes_skip is None
    assert torch.equal(z_q_skip, z_q_ref)


def test_generate_return_codes_false_without_return_latent_raises():
    model, _ = _make_tiny_model()
    prefix = torch.randint(1, 32, (1, 4))

    with pytest.raises(ValueError, match="return_latent=True"):
        model.generate(None, prefix, return_latent=False, return_codes=False)


def test_generate_hard_decisions_with_return_codes_false_preserves_latent():
    model, _ = _make_tiny_model()
    prefix = torch.randint(1, 32, (2, 4))

    torch.manual_seed(7)
    codes_ref, z_q_ref = model.generate(
        None, prefix, temperature=1.0, top_k=8, top_p=0.9,
        soft_decisions=False, return_latent=True,
    )

    torch.manual_seed(7)
    codes_none, z_q_none = model.generate(
        None, prefix, temperature=1.0, top_k=8, top_p=0.9,
        soft_decisions=False, return_latent=True, return_codes=False,
    )

    assert codes_ref is not None
    assert codes_none is None
    assert torch.equal(z_q_none, z_q_ref)


def test_generator_soft_decisions_skips_sample_level(monkeypatch, tmp_path):
    model, vqvae = _make_tiny_model(levels=(2, 4, 8), vocab_size=16, embed_dim=8, prefix_len=4)
    fake_pre = SimpleNamespace(eos_token_id=1, pad_token_id=0, decode=lambda ids, **kw: "ok")
    fake_tok = SimpleNamespace(vqvae=vqvae, pre_tokenizer=fake_pre)

    monkeypatch.setattr("utils.generation.VQVAETokenizer.from_config", lambda cfg: fake_tok)
    monkeypatch.setattr("utils.generation.NextConceptTransformer", lambda vqvae_local, config: model)
    monkeypatch.setattr("utils.generation.torch.load", lambda *a, **kw: {"model": model.state_dict()})
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    monkeypatch.setattr("utils.generation.fuse_token_picker", lambda picker, *a: picker)

    cfg = OmegaConf.create({
        "checkpoints": {"generator": "gen.pt", "tokenizer": "tok.pt"},
        "tokenizer": {"pre_tokenizer": {}, "vqvae": {}},
        "model": {},
        "generation": {
            "temperature": 1.0,
            "top_k": 8,
            "top_p": 0.9,
            "cfg_scale": 0.0,
            "soft_decisions": True,
            "soft_temperature": 0.5,
        },
        "eval": {},
    })
    options = SimpleNamespace(
        kv_cache=True, resize_matmul=False, compile=False, cuda_graphs=False, precision="fp32",
    )

    gen = Generator(cfg, tmp_path, options)
    assert gen.sampling["return_codes"] is False

    monkeypatch.setattr(
        model, "_sample_level",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("_sample_level should be skipped")),
    )
    prefix = torch.randint(1, 32, (2, 4))
    tokens = gen._generate(prefix)
    assert tokens.shape == (2, 32)


@pytest.mark.parametrize("use_kv_cache", [True, False])
def test_generate_cfg_soft_decisions_skips_sampling_and_matches_baseline(monkeypatch, use_kv_cache):
    model, _ = _make_tiny_model(query_level0=True, l0_autoregressive=False)
    prefix = torch.randint(1, 32, (2, 4))

    torch.manual_seed(17)
    _, z_q_ref = model.generate(
        None, prefix, temperature=1.0, top_k=8, top_p=0.9,
        use_kv_cache=use_kv_cache, cfg_scale=1.5, cfg_null_eos=1,
        soft_decisions=True, soft_temperature=0.6, return_latent=True,
    )

    monkeypatch.setattr(
        model, "_sample_level",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("_sample_level called")),
    )

    codes_skip, z_q_skip = model.generate(
        None, prefix, temperature=1.0, top_k=8, top_p=0.9,
        use_kv_cache=use_kv_cache, cfg_scale=1.5, cfg_null_eos=1,
        soft_decisions=True, soft_temperature=0.6, return_latent=True, return_codes=False,
    )

    assert codes_skip is None
    assert z_q_skip.shape == (2, model.levels[-1], model.vqvae_local[0].embed_dim)
    assert torch.equal(z_q_skip, z_q_ref)
