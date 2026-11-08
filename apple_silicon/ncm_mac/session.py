"""High-level API: load a task on a backend and turn prompts into text (with timings)."""
from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .tasks import Task, load_task, require_checkpoint
from .textio import TextSide


@dataclass
class Result:
    codes: np.ndarray
    texts: list[str]
    level_texts: list[list[str]] | None = None      # [sample][level]
    t_generate: float = 0.0
    t_decode: float = 0.0
    t_post: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def t_total(self):
        return self.t_generate + self.t_decode + self.t_post


class NCMSession:
    def __init__(self, task: str | Task, backend: str = "mlx", device: str | None = None, dtype: str | None = None):
        self.task = load_task(task) if isinstance(task, str) else task
        require_checkpoint(self.task)
        self.backend = backend
        if backend == "mlx":
            from .convert import convert
            from .mlx_ncm import MLXNCM
            if not os.path.exists(os.path.join(self.task.mlx_dir, "weights.safetensors")):
                print(f"[convert] {self.task.name}: writing MLX weights to {self.task.mlx_dir}", flush=True)
                convert(self.task)
            self.model = MLXNCM(self.task.mlx_dir, dtype=dtype)
        elif backend == "torch":
            from .torch_backend import TorchNCM
            self.model = TorchNCM(self.task, device=device)
        else:
            raise ValueError(backend)
        self.text = TextSide(self.task, self.model.inference)
        self.levels = self.text.levels

    # ------------------------------------------------------------------ settings
    def sampling(self, **ov):
        G = self.task.gen
        kw = dict(temperature=float(G.get("temperature", 1.0)), top_k=int(G.get("top_k", 0)),
                  top_p=float(G.get("top_p", 1.0)), cfg_scale=float(G.get("cfg_scale", 0.0)))
        kw.update({k: v for k, v in ov.items() if v is not None})
        kw["cfg_null_eos"] = self.text.EOS if self.text.EOS is not None else 0
        return kw

    # ------------------------------------------------------------------ core
    def run(self, prefixes: np.ndarray, l0: np.ndarray | None = None, *, seed: int | None = None,
            show_levels: bool = False, freq=None, mu=None, adj=None, **sampling) -> Result:
        kw = self.sampling(**sampling)
        m = self.model
        t0 = time.perf_counter()
        codes = m.generate(prefixes, l0, seed=seed, **kw)
        m.sync()
        if self.backend == "mlx":
            import mlx.core as mx
            mx.eval(codes)
        t1 = time.perf_counter()
        codes_np = m.to_numpy(codes)
        if show_levels:
            per_level = m.decode_logits(codes_np, all_levels=True)
            final = per_level[-1]
        else:
            final = m.decode_logits(codes_np)
        t2 = time.perf_counter()
        ids = self.text.pick_tokens(final, freq=freq, mu=mu, adj=adj)
        texts = [self.text.decode(r) for r in ids]
        t3 = time.perf_counter()
        level_texts = None
        if show_levels:
            level_texts = [[self.text.decode(lg[b].argmax(-1)) for lg in per_level[:-1]] + [texts[b]]
                           for b in range(len(texts))]
        return Result(codes=codes_np, texts=texts, level_texts=level_texts, t_generate=t1 - t0,
                      t_decode=t2 - t1, t_post=t3 - t2, extra={"raw_ids": ids})

    def unconditional(self, n: int, seed: int = 0, l0_mode: str = "unigram", **kw) -> Result:
        g = torch.Generator().manual_seed(seed)
        l0 = self.text.sample_level0(n, g, l0_mode)
        return self.run(self.text.uncond_prefix(n), l0, seed=seed, **kw)

    def conditional(self, prompts: list, seed: int | None = 0, **kw) -> Result:
        pre = np.array([self.text.prefix(p) for p in prompts], dtype=np.int64)
        return self.run(pre, None, seed=seed, **kw)

    def mbr(self, prompts: list, k: int, seed: int | None = 0, **kw) -> Result:
        """Test-time scaling (paper §4.5): sample k candidates per prompt in one batch and keep
        the one with the highest mean similarity to the others (minimum Bayes risk).
        Utility: ROUGE-L for summarisation, chrF otherwise."""
        expanded = [p for p in prompts for _ in range(k)]
        r = self.conditional(expanded, seed=seed, **kw)
        if str(self.task.cfg.eval.metric) == "rouge":
            from rouge_score import rouge_scorer
            sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            sim = lambda a, b: sc.score(b, a)["rougeL"].fmeasure
        else:
            import sacrebleu
            sim = lambda a, b: sacrebleu.sentence_chrf(a, [b]).score
        chosen, cands = [], []
        for i in range(len(prompts)):
            c = r.texts[i * k:(i + 1) * k]
            scores = [np.mean([sim(c[a], c[b]) for b in range(k) if b != a]) if k > 1 else 0.0 for a in range(k)]
            chosen.append(c[int(np.argmax(scores))])
            cands.append(list(zip(c, scores)))
        r.texts = chosen
        r.extra["candidates"] = cands
        return r

    def encode(self, texts: list) -> np.ndarray:
        """Text (or raw token-id lists) -> the tokenizer's code hierarchy [B, sum(levels)]."""
        import torch
        if not hasattr(self, "_vq_enc"):
            from .torch_backend import load_vqvae
            self._vq_enc = load_vqvae(self.task, "cpu")
        T = int(self.task.cfg.tokenizer.vqvae.context_length)
        pad = self.text.PAD if self.text.PAD is not None else 0
        rows = []
        for t in texts:
            ids = self.text.pre.encode(t) + ([self.text.EOS] if self.text.EOS is not None else []) \
                if isinstance(t, str) else list(t)
            rows.append((ids + [pad] * T)[:T])
        with torch.no_grad():
            return self._vq_enc.encode(torch.tensor(rows)).numpy()

    def decode_codes(self, codes: np.ndarray, **pen) -> list[str]:
        ids = self.text.pick_tokens(self.model.decode_logits(codes), **pen)
        return [self.text.decode(r) for r in ids]

    def regenerate(self, codes_prefix: np.ndarray, seed: int = 0, **sampling) -> np.ndarray:
        """Keep the given leading codes (whole levels), sample the remaining levels."""
        kw = self.sampling(**sampling)
        n = codes_prefix.shape[0]
        pre = self.text.uncond_prefix(n) if self.task.kind == "uncond" else None
        l0 = codes_prefix[:, : self.levels[0]]
        codes = self.model.generate(pre, l0, seed=seed, force_codes=codes_prefix, **kw)
        return self.model.to_numpy(codes)

    def level_offset(self, k: int) -> int:
        """Number of codes in levels 0..k-1."""
        return int(sum(self.levels[:k]))

    # ------------------------------------------------------------------ data
    def load_split(self, split: str = "test"):
        d = self.task.cfg.data
        if not d or not d.get(split):
            return None, None
        items = pickle.load(open(os.path.join(self.task.data_dir, d[split]), "rb"))
        refs = pickle.load(open(os.path.join(self.task.data_dir, d[f"{split}_refs"]), "rb"))
        return items, refs

    def ref_text(self, r):
        return r if isinstance(r, str) else self.text.decode(r)
