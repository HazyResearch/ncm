"""Backend-independent text side of NCM: pre-tokenizers, prefixes, decoder-logit post-processing.

Mirrors the logic in upstream scripts/eval_shard.py so both the PyTorch and MLX backends
produce text the same way the paper's evaluation does.
"""
from __future__ import annotations

import numpy as np
import torch

from .tasks import Task, ensure_upstream_importable


class TextSide:
    def __init__(self, task: Task, inference: dict[str, np.ndarray]):
        ensure_upstream_importable()
        import hydra
        from utils.registry import tokenizers

        self.task = task
        cfg = task.cfg
        G = cfg.generation
        pre_cfg = cfg.tokenizer.pre_tokenizer
        self.pre = hydra.utils.get_class(tokenizers[pre_cfg.name]).from_config(pre_cfg)
        self.EOS = self.pre.eos_token_id
        self.PAD = self.pre.pad_token_id
        self.P = int(cfg.model.prefix_len)
        self.levels = list(cfg.tokenizer.vqvae.vector_quantizer_config.levels.manual_levels)
        self.cut_at_eos = bool(G.get("cut_at_eos", True))

        # Source-side tokenizer for string prompts (XSum passages, WMT14 German sources).
        self._src_tok = None
        if pre_cfg.name == "hf":
            from transformers import AutoTokenizer
            self._src_tok = AutoTokenizer.from_pretrained(str(pre_cfg.model_id))

        # Default decoding penalties from the task config (see eval_shard.pick_tokens).
        self.default_freq = float(G.get("frequency_penalty", 0.0))
        self.default_mu = float(G.get("frequency_deficit", 0.0))
        self.default_adj = str(G.get("adjacency_penalty", "") or "")
        self.allow = None
        if G.get("frequency_allowance"):
            self.allow = torch.from_numpy(np.asarray(inference["token_unigram"])).double()
        self.l0_unigram = None
        if G.get("level0_unigram"):
            self.l0_unigram = torch.from_numpy(np.asarray(inference["level0_unigram"])).float()

    # ------------------------------------------------------------------ prompts
    def prefix(self, item) -> list[int]:
        """Prompt -> fixed-length prefix of raw token ids, right-padded (eval_shard.prefix)."""
        if isinstance(item, str):
            if self._src_tok is not None:
                ids = self._src_tok(item)["input_ids"]
            else:
                ids = self.pre.encode(item)
        else:
            ids = list(item)
        pad = self.PAD if self.PAD is not None else 0
        return (ids[: self.P] + [pad] * self.P)[: self.P]

    def uncond_prefix(self, batch: int) -> np.ndarray:
        return np.full((batch, 1), self.EOS, dtype=np.int64)

    def sample_level0(self, batch: int, generator: torch.Generator | None = None, mode: str = "unigram"):
        """Unconditional tasks seed level 0 from the training unigram over level-0 codes."""
        if self.l0_unigram is None:
            return None
        p = self.l0_unigram
        if mode == "uniform":
            p = torch.ones_like(p) / p.numel()
        elif mode == "active":
            p = (p > 0).float() / (p > 0).sum()
        elif mode.startswith("smooth:"):
            eps = float(mode.split(":", 1)[1])
            p = (1 - eps) * p + eps * torch.ones_like(p) / p.numel()
        n0 = self.levels[0]
        return torch.multinomial(p, batch * n0, replacement=True, generator=generator).view(batch, n0).numpy()

    # ------------------------------------------------------------------ outputs
    def pick_tokens(self, logits: np.ndarray, freq=None, mu=None, adj=None) -> np.ndarray:
        """Decoder logits [B,T,V] -> token ids [B,T]. Greedy, optionally with the paper's
        frequency / adjacency penalties (sequential over positions, done on CPU in float64)."""
        freq = self.default_freq if freq is None else freq
        mu = self.default_mu if mu is None else mu
        adj = self.default_adj if adj is None else adj
        ADJ = [float(v) for v in adj.split(",")] if adj else None
        if freq <= 0 and not ADJ:
            return logits.argmax(-1)
        lg = torch.from_numpy(np.asarray(logits)).double()
        B, T, V = lg.shape
        allow = (self.allow[:V] * T)[None, :] if self.allow is not None else None
        al = ADJ[0] if ADJ else 0.0
        aw = int(ADJ[1]) if ADJ and len(ADJ) > 1 else 1
        ad = ADJ[2] if ADJ and len(ADJ) > 2 else 1.0
        cnt = torch.zeros(B, V, dtype=torch.float64)
        out = torch.empty(B, T, dtype=torch.long)
        hist = []
        for t in range(T):
            step = lg[:, t, :].clone()
            if freq > 0:
                excess = (cnt - allow).clamp_min(0.0) if allow is not None else cnt
                step -= freq * excess
                if mu > 0 and allow is not None:
                    step += mu * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
            for j, h in enumerate(reversed(hist[-aw:])):
                step.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=torch.float64))
            p = step.argmax(-1)
            out[:, t] = p
            hist.append(p)
            cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=torch.float64))
        return out.numpy()

    def decode(self, ids, cut_at_eos=None) -> str:
        ids = [int(t) for t in ids]
        if self.task.name.startswith("lm1b"):
            # LM1B packs BERT-tokenised sentences as "[CLS] … [SEP] [CLS] …"; its "EOS" is [CLS],
            # which opens every sample, so cut-at-EOS would leave nothing. The paper scores the raw
            # ids with specials kept (FLM protocol), so show them that way too, minus padding.
            return self.pre.hf_tokenizer.decode([t for t in ids if t != self.PAD], skip_special_tokens=False)
        cut = self.cut_at_eos if cut_at_eos is None else cut_at_eos
        if cut and self.EOS is not None and self.EOS in ids:
            ids = ids[: ids.index(self.EOS)]
        ids = [t for t in ids if t not in (self.EOS, self.PAD)]
        return self.pre.decode(ids, skip_special_tokens=True)
