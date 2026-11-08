"""Reference backend: the unmodified upstream PyTorch model, run on Apple's MPS (or CPU).

The upstream code only hard-codes `.cuda()` in its eval scripts; the model classes themselves
are device-agnostic, flex-attention is disabled in every released config, and nothing needs
Triton. So we import the original modules and just move them to `mps`.
"""
from __future__ import annotations

import contextlib
import io
import os

import numpy as np
import torch

from .tasks import Task, ensure_upstream_importable


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    return "mps" if torch.backends.mps.is_available() else "cpu"


class _MPSSafeF:
    """Stand-in for torch.nn.functional inside models/quant.py: MPS cannot do adaptive (area)
    pooling when the input length isn't a multiple of the output length, which the multiscale
    VQ needs between levels. We do that case as an exact pooling-matrix multiply instead."""

    def __init__(self, F):
        self._F = F
        self._mats = {}

    def __getattr__(self, k):
        return getattr(self._F, k)

    def interpolate(self, x, size=None, mode="nearest", **kw):
        if mode == "area" and x.device.type == "mps":
            L_in, L_out = x.shape[-1], (size[0] if isinstance(size, (tuple, list)) else int(size))
            if L_in % L_out:
                key = (L_in, L_out, x.dtype)
                if key not in self._mats:
                    M = torch.zeros(L_out, L_in)
                    for i in range(L_out):
                        s, e = (i * L_in) // L_out, -((-(i + 1) * L_in) // L_out)
                        M[i, s:e] = 1.0 / (e - s)
                    self._mats[key] = M.T.to(device=x.device, dtype=x.dtype)
                return x @ self._mats[key]
        return self._F.interpolate(x, size=size, mode=mode, **kw)


def _patch_upstream_for_mps():
    import models.quant as q
    if not isinstance(q.F, _MPSSafeF):
        q.F = _MPSSafeF(q.F)


def _torch_load(path):
    # The released checkpoints are plain dicts of tensors + a few python scalars; try the safe
    # loader first and only fall back to full unpickling (what upstream does) if required.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_vqvae(task: Task, device: str = "cpu"):
    """Just the frozen multiscale tokenizer (for encoding text into codes)."""
    ensure_upstream_importable()
    from models.vqvae import VQVAE
    _patch_upstream_for_mps()
    with contextlib.redirect_stdout(io.StringIO()):
        vq = VQVAE(**task.cfg.tokenizer.vqvae)
        vq.load_state_dict(_torch_load(task.cfg.tokenizer.vqvae.checkpoint_path)["model"], strict=True)
    return vq.to(device).eval()


class TorchNCM:
    name = "torch"

    def __init__(self, task: Task, device: str | None = None, quiet: bool = True):
        ensure_upstream_importable()
        from models.next_concept import NextConceptTransformer
        from models.vqvae import VQVAE
        _patch_upstream_for_mps()

        self.task = task
        self.device = pick_device(device)
        cfg = task.cfg
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
            vqvae = VQVAE(**cfg.tokenizer.vqvae)
            vqvae.load_state_dict(_torch_load(cfg.tokenizer.vqvae.checkpoint_path)["model"], strict=True)
            self.vqvae = vqvae.to(self.device).eval()
            self.net = NextConceptTransformer(vqvae_local=self.vqvae, config=cfg.model)
            sd = _torch_load(os.path.join(task.ckpt_dir, cfg.checkpoints.generator))
            self.net.load_state_dict(sd["model"], strict=True)
            self.net = self.net.to(self.device).eval()
        self.inference = {k: v.cpu().numpy() for k, v in (sd.get("inference") or {}).items()
                          if torch.is_tensor(v)}
        self.levels = list(self.vqvae.vq.levels)

    def sync(self):
        if self.device == "mps":
            torch.mps.synchronize()

    @torch.no_grad()
    def generate(self, prefix: np.ndarray, l0: np.ndarray | None = None, *, temperature=1.0,
                 top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=50256, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
        x = torch.as_tensor(prefix, dtype=torch.long, device=self.device)
        idx = torch.as_tensor(l0, dtype=torch.long, device=self.device) if l0 is not None else None
        g = self.net.generate(idx, x, temperature=temperature, top_k=top_k, top_p=top_p,
                              cfg_scale=cfg_scale, cfg_null_eos=cfg_null_eos)
        return g

    @torch.no_grad()
    def decode_logits(self, codes, all_levels: bool = False):
        """codes [B, sum(levels)] -> decoder logits. With all_levels, one array per level
        (the coarse-to-fine view: z_hat after level k, pooled to length l_k, then decoded)."""
        codes = torch.as_tensor(codes, dtype=torch.long, device=self.device)
        _, zl = self.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
        if not all_levels:
            return self.vqvae.decoder(zl[-1]).float().cpu().numpy()
        return [self.vqvae.decoder(z).float().cpu().numpy() for z in zl]

    @torch.no_grad()
    def encode_text_ids(self, ids: np.ndarray):
        """Raw token ids [B, context_length] -> multiscale codes (tokenizer encode)."""
        x = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        return self.vqvae.encode(x)

    def to_numpy(self, codes):
        return codes.cpu().numpy() if torch.is_tensor(codes) else np.asarray(codes)

    # ---- instrumentation for the MLX parity check
    @torch.no_grad()
    def generate_traced(self, prefix, l0=None, **kw):
        """Greedy generate that also records the logits produced at every level."""
        logs = []
        orig = self.net._sample_level_raw

        def rec(logits, temperature, top_k, top_p):
            logs.append(logits.float().cpu().numpy())
            return orig(logits, 0, top_k, top_p)

        self.net._sample_level_raw = rec
        try:
            g = self.generate(prefix, l0, temperature=0, **kw)
        finally:
            self.net._sample_level_raw = orig
        return g.cpu().numpy(), logs
