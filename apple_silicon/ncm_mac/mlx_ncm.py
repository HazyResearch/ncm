"""Pure-MLX inference port of Next-Concept Modeling (HazyResearch/ncm).

Covers everything the released checkpoints use:
  * generator: pre-LN transformer, QK-RMSNorm, per-level RoPE (positions reset per level),
    level embeddings, block-diagonal prefix/level masking, prefix-pad masking
  * level-0 modes: given (unconditional unigram seed), learned queries (query_level0),
    predicted from the prefix (predict_level0)
  * output heads: codebook-tied distance heads, shared head with per-level bias, per-level linear
  * post-upsample residual convs (LM1B), classifier-free guidance
  * multiscale residual VQ: learned (cascaded, optionally GroupNorm'd) level-0 upsampler,
    linear upsampling for other levels, area pooling, per-level quant_resi (Phi) convs
  * VQ-VAE decoder (transformer block(s) + LN + vocab projection)

Weights come from `ncm_mac.convert` (safetensors). Layout here is NLC everywhere.
"""
from __future__ import annotations

import json
import math
import os

import mlx.core as mx
import numpy as np

NEG_INF = float("-inf")


# ----------------------------------------------------------------------------- helpers
def _linear(x, w, b=None):
    y = x @ w.T
    return y if b is None else y + b


def _ln(x, p, prefix, eps=1e-6):
    return mx.fast.layer_norm(x, p[prefix + ".weight"], p.get(prefix + ".bias"), eps)


def _rope(x, cos, sin):
    half = x.shape[-1] // 2
    rot = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rot * sin


def _rope_tables(positions: np.ndarray, head_dim: int, base: float):
    half = head_dim // 2
    inv = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / half))
    f = positions.astype(np.float32)[:, None] * inv[None, :]
    cos = np.concatenate([np.cos(f), np.cos(f)], -1)
    sin = np.concatenate([np.sin(f), np.sin(f)], -1)
    return mx.array(cos), mx.array(sin)


def _area_matrix(L_in: int, L_out: int) -> np.ndarray:
    """F.interpolate(mode='area') == adaptive_avg_pool1d, as an [L_out, L_in] matrix."""
    M = np.zeros((L_out, L_in), np.float32)
    for i in range(L_out):
        s = (i * L_in) // L_out
        e = -((-(i + 1) * L_in) // L_out)
        M[i, s:e] = 1.0 / (e - s)
    return M


def _linear_interp_matrix(L_in: int, L_out: int) -> np.ndarray:
    """F.interpolate(mode='linear', align_corners=False), as an [L_out, L_in] matrix."""
    M = np.zeros((L_out, L_in), np.float32)
    scale = L_in / L_out
    for i in range(L_out):
        src = max((i + 0.5) * scale - 0.5, 0.0)
        i0 = min(int(math.floor(src)), L_in - 1)
        i1 = min(i0 + 1, L_in - 1)
        w1 = src - i0
        M[i, i0] += 1.0 - w1
        M[i, i1] += w1
    return M


def _plan_cascade_strides(total_stride, base=4):
    S, strides = int(total_stride), []
    while S > 1:
        f = min(base, S)
        while f > 1 and S % f != 0:
            f -= 1
        if f <= 1:
            strides.append(S)
            S = 1
        else:
            strides.append(f)
            S //= f
    return strides or [1]


def _group_norm(x, groups, eps=1e-5):
    """Affine-free GroupNorm over (channels-in-group, length) for an NLC tensor."""
    B, L, C = x.shape
    g = x.reshape(B, L, groups, C // groups)
    mean = g.mean(axis=(1, 3), keepdims=True)
    var = ((g - mean) ** 2).mean(axis=(1, 3), keepdims=True)
    return ((g - mean) * mx.rsqrt(var + eps)).reshape(B, L, C)


def _conv1d_same(x, w, b):
    """x NLC, w MLX layout [out, k, in]."""
    return mx.conv1d(x, w, padding=w.shape[1] // 2) + b


# ----------------------------------------------------------------------------- transformer
class _Blocks:
    """Stack of upstream `Block`s (models/common.py or models/vae.py — same math)."""

    def __init__(self, W, prefix, n, n_head, n_kv_head=None, qk_norm=True):
        self.layers = []
        for i in range(n):
            pre = f"{prefix}{i}." if n is not None else prefix
            self.layers.append({k[len(pre):]: v for k, v in W.items() if k.startswith(pre)})
        self.n_head = n_head
        self.n_kv = n_kv_head or n_head
        self.qk_norm = qk_norm

    def _attn(self, x, p, mask, rope):
        B, T, C = x.shape
        H, Hk = self.n_head, self.n_kv
        hd = C // H
        qkv = _linear(x, p["attn.c_attn.weight"], p.get("attn.c_attn.bias"))
        q, k, v = mx.split(qkv, [C, C + Hk * hd], axis=-1)
        q = q.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, Hk, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, Hk, hd).transpose(0, 2, 1, 3)
        if self.qk_norm and "attn.q_norm.weight" in p:
            q = mx.fast.rms_norm(q, p["attn.q_norm.weight"], 1e-6)
            k = mx.fast.rms_norm(k, p["attn.k_norm.weight"], 1e-6)
        if rope is not None:
            cos, sin = rope
            cos, sin = cos.astype(q.dtype), sin.astype(q.dtype)
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        if mask is not None:
            mask = mask.astype(q.dtype)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(hd), mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        return _linear(y, p["attn.c_proj.weight"], p.get("attn.c_proj.bias"))

    def _mlp(self, x, p):
        h = _linear(x, p["mlp.c_fc.weight"], p.get("mlp.c_fc.bias"))
        h = mx.erf(h * (1.0 / math.sqrt(2.0))) * h * 0.5 + h * 0.5     # exact (erf) GELU
        return _linear(h, p["mlp.c_proj.weight"], p.get("mlp.c_proj.bias"))

    def __call__(self, x, mask=None, rope=None):
        for p in self.layers:
            x = x + self._attn(_ln(x, p, "ln_1"), p, mask, rope)
            x = x + self._mlp(_ln(x, p, "ln_2"), p)
        return x


# ----------------------------------------------------------------------------- VQ + decoder
class MultiscaleVQ:
    def __init__(self, W, vcfg, embed_dim):
        q = vcfg["vector_quantizer_config"]
        self.levels = list(q["levels"]["manual_levels"])
        n = len(self.levels)
        L = self.levels[-1]
        self.D = embed_dim
        self.emb = [W[f"vq.embeddings.{i}.weight"] for i in range(n)]

        # quant_resi: PhiNonShared (share_mode 0) -> nearest tick to level/(n-1)
        qr = q.get("quant_resi") or {}
        self.qr_enabled = bool(qr.get("enabled", True))
        if self.qr_enabled:
            assert int(qr.get("share_mode", 0)) == 0 and not qr.get("learnable_ratio", False)
            assert str(qr.get("arch", "blend")) == "blend"
            ratio = qr.get("ratio", 0.5)
            ratios = [float(r) for r in ratio] if isinstance(ratio, (list, tuple)) else [float(ratio)] * n
            K = n
            ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
            self.phi = []
            for i in range(n):
                j = int(np.argmin(np.abs(ticks - i / max(n - 1, 1))))
                r = abs(ratios[j])
                if r <= 1e-6:
                    self.phi.append(None)
                else:
                    w = W[f"vq.quant_resi.{j}.weight"].transpose(0, 2, 1)  # [out,in,k] -> [out,k,in]
                    self.phi.append((w, W[f"vq.quant_resi.{j}.bias"], r))

        # level-0 upsampler
        self.learned_l1 = bool(q.get("learned_l1_sampling", False)) or bool(q.get("learned_all_sampling", False))
        assert not q.get("learned_all_sampling", False), "learned_all_sampling not used by released models"
        self.up_ops = []
        if self.learned_l1:
            stride = L // self.levels[0]
            if q.get("learned_l1_cascade", False):
                strides = list(reversed(_plan_cascade_strides(stride, int(q.get("learned_l1_cascade_base", 4)))))
                inter = bool(q.get("learned_l1_cascade_norm", False))
                k = 0
                for i, s in enumerate(strides):
                    if inter and i > 0:
                        self.up_ops.append(("gn", 8))
                        k += 1
                    self.up_ops.append(("convT", self._convT(W, f"vq.learned_up_l1.{k}"), s))
                    k += 1
            else:
                self.up_ops.append(("convT", self._convT(W, "vq.learned_up_l1"), stride))

        self.up_mats = [mx.array(_linear_interp_matrix(l, L)) for l in self.levels]
        self.area_mats = [mx.array(_area_matrix(L, l)) for l in self.levels]

    @staticmethod
    def _convT(W, pre):
        w = W[pre + ".weight"]                     # torch ConvTranspose1d: [in, out, k]
        cin, cout, k = w.shape
        wm = w.transpose(0, 2, 1).reshape(cin, k * cout)
        return wm, W[pre + ".bias"], cout

    def up_to_full(self, q, i):
        if i == 0 and self.learned_l1:
            x = q
            for op in self.up_ops:
                if op[0] == "gn":
                    x = _group_norm(x, op[1])
                else:
                    (wm, b, cout), s = op[1], op[2]
                    B, T, _ = x.shape
                    x = (x @ wm).reshape(B, T * s, cout) + b
            return x
        return self.up_mats[i] @ q

    def apply_phi(self, h, i):
        if not self.qr_enabled or self.phi[i] is None:
            return h
        w, b, r = self.phi[i]
        return h * (1 - r) + _conv1d_same(h, w, b) * r

    def area(self, z, i):
        return self.area_mats[i] @ z

    def next_input(self, level_idx, z_hat, z_q_level):
        up = self.apply_phi(self.up_to_full(z_q_level, level_idx), level_idx)
        z_hat = z_hat + up
        return z_hat, self.area(z_hat, level_idx + 1)

    def codes_to_zlevels(self, codes):
        """idx_to_zhats(..., return_zlevels=True)[1]"""
        B = codes.shape[0]
        n = len(self.levels)
        z_hat = mx.zeros((B, self.levels[-1], self.D))
        out, acc = [], 0
        for i, l in enumerate(self.levels):
            e = self.emb[i][codes[:, acc:acc + l]]
            zq = self.up_to_full(e, i) if i != n - 1 else e
            z_hat = z_hat + self.apply_phi(zq, i)
            out.append(self.area(z_hat, i) if i != n - 1 else z_hat)
            acc += l
        return out


class Decoder:
    def __init__(self, W, vcfg):
        n_head = int(vcfg["n_layers"])        # upstream quirk: n_layers is the decoder head count
        n_blocks = int(vcfg.get("dec_layers", 1))
        self.wpe = W.get("dec.wpe.weight")
        sub = {}
        for k, v in W.items():
            if k.startswith("dec.middle_block."):
                sub["b.0." + k[len("dec.middle_block."):]] = v
            elif k.startswith("dec.extra_blocks."):
                rest = k[len("dec.extra_blocks."):]
                j, tail = rest.split(".", 1)
                sub[f"b.{int(j) + 1}.{tail}"] = v
        self.blocks = _Blocks(sub, "b.", n_blocks, n_head, qk_norm=bool(vcfg.get("use_qk_norm", False)))
        self.ln_f = W["dec.ln_f.weight"]
        self.out = W["dec.output_layer.weight"]
        assert int(vcfg.get("compression_factor", 1)) == 1

    def __call__(self, z):
        if self.wpe is not None:
            z = z + self.wpe[: z.shape[1]]
        x = self.blocks(z)
        x = mx.fast.layer_norm(x, self.ln_f, None, 1e-6)
        return x @ self.out.T


# ----------------------------------------------------------------------------- generator
class _Head:
    def __init__(self, kind, **kw):
        self.kind = kind
        self.__dict__.update(kw)

    def __call__(self, h):
        if self.kind == "tied":
            z = h @ self.proj.T
            out = (2.0 * (z.astype(mx.float32) @ self.E.T) - self.E_sq) / self.d_scale * self.inv_T
        else:
            out = (h @ self.w.T).astype(mx.float32)
        return out if self.bias is None else out + self.bias


class MLXNCM:
    name = "mlx"

    def __init__(self, weights_dir: str, dtype: str | None = None):
        meta = json.load(open(os.path.join(weights_dir, "config.json")))
        self.cfg = cfg = meta["config"]
        W = mx.load(os.path.join(weights_dir, "weights.safetensors"))
        if dtype:   # runtime cast of generator/decoder weights (VQ stays fp32)
            dt = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[dtype]
            W = {k: (v.astype(dt) if (k.startswith(("gen.", "dec.")) and v.dtype in (mx.float32, mx.float16, mx.bfloat16)) else v)
                 for k, v in W.items()}
        self.compute_dtype = W["gen.transformer.ln_f.weight"].dtype
        m, vcfg = cfg["model"], cfg["tokenizer"]["vqvae"]
        self.m = m
        self.inference = {k[len("inference."):]: np.array(v) for k, v in W.items() if k.startswith("inference.")}

        self.vq = MultiscaleVQ(W, vcfg, int(vcfg["embed_dim"]))
        self.decoder = Decoder(W, vcfg)
        self.enc_wte = W["enc_wte"]
        self.levels = self.vq.levels

        for flag in ("parallel_block", "normformer", "use_layerscale", "use_rezero", "use_swiglu",
                     "use_geglu", "use_relu2", "l0_autoregressive", "contextual_prefix"):
            assert not m.get(flag, False), f"{flag} not supported by this port"
        assert m.get("use_rope", False), "only RoPE checkpoints are released"
        assert not (m.get("ctx_encoder") or {}).get("enabled", False)
        assert m.get("raw_token_prefix", False)
        assert float(m.get("mlp_ratio", 4.0)) == 4.0

        G = {k[len("gen."):]: v for k, v in W.items() if k.startswith("gen.")}
        self.G = G
        self.P = int(m["prefix_len"])
        self.n_embd = int(m["n_embd"])
        self.query_level0 = bool(m.get("query_level0", False))
        self.predict_level0 = bool(m.get("predict_level0", False))
        self.lvl_first = 0 if self.query_level0 else 1
        levels_t = self.levels[self.lvl_first:]
        self.blocks = _Blocks(G, "transformer.blocks.", int(m["n_layer"]), int(m["n_head"]),
                              m.get("n_kv_heads"), qk_norm=bool(m.get("use_qk_norm", False)))
        self.wte_w, self.wte_b = G["transformer.wte.weight"], G.get("transformer.wte.bias")
        self.ln_f = ("transformer.ln_f", G)
        lvl_1L = np.concatenate([np.full(l, i) for i, l in enumerate(levels_t)])
        if m.get("use_level_encoding", False):
            self.level_pos = G["transformer.wle.weight"][mx.array(lvl_1L)][None]
        else:
            self.level_pos = mx.zeros((1, len(lvl_1L), self.n_embd), dtype=self.compute_dtype)

        hd = self.n_embd // int(m["n_head"])
        base = float(m.get("rope_base", 10000))
        self.rope_gen = [_rope_tables(np.concatenate([np.arange(self.P), np.arange(l)]), hd, base) for l in levels_t]

        self.puc = None
        if (m.get("post_upsample_conv") or {}).get("enabled", False):
            n = len(self.levels) - (0 if self.query_level0 else 1)
            self.puc = [(G[f"post_upsample_convs.{i}.weight"].transpose(0, 2, 1), G[f"post_upsample_convs.{i}.bias"]) for i in range(n)]

        self.l0_queries = G.get("l0_queries")
        self.prefix_pad_id = m.get("prefix_pad_token_id")
        self.heads = self._build_heads(m)
        self.lm_head_l0 = None
        if self.predict_level0:
            if m.get("tie_heads_to_codebook", False):
                self.lm_head_l0 = self._tied("lm_head_l0", self.vq.emb[0])
            else:
                self.lm_head_l0 = _Head("linear", w=G["lm_head_l0.weight"], bias=G.get("lm_head_l0.bias"))

    # -- heads
    def _tied(self, pre, E):
        E = E.astype(mx.float32)
        E_sq = (E ** 2).sum(-1)
        d_scale = mx.maximum(E_sq.mean(), 1e-12)
        lit = self.G.get(pre + ".log_inv_T")
        inv_T = mx.exp(lit.astype(mx.float32)) if lit is not None else mx.array(1.0)
        b = self.G.get(pre + ".bias")
        return _Head("tied", proj=self.G[pre + ".proj.weight"], E=E, E_sq=E_sq, d_scale=d_scale, inv_T=inv_T,
                     bias=None if b is None else b.astype(mx.float32))

    def _build_heads(self, m):
        n = len(m["vocab_size"])
        G = self.G
        if m.get("shared_output_head", False):
            w = G["lm_heads.head.weight"]
            if m.get("shared_head_per_level_bias", False):
                return [_Head("linear", w=w, bias=G[f"lm_heads.biases.{i}"].astype(mx.float32)) for i in range(n)]
            b = G.get("lm_heads.head.bias")
            return [_Head("linear", w=w, bias=None if b is None else b.astype(mx.float32))] * n
        if m.get("tie_heads_to_codebook", False):
            return [self._tied(f"lm_heads.{i}", self.vq.emb[i + self.lvl_first]) for i in range(n)]
        return [_Head("linear", w=G[f"lm_heads.{i}.weight"],
                      bias=(G[f"lm_heads.{i}.bias"].astype(mx.float32) if f"lm_heads.{i}.bias" in G else None))
                for i in range(n)]

    # -- pieces
    def _wte(self, x):
        return _linear(x.astype(self.compute_dtype), self.wte_w, self.wte_b)

    def _puc(self, x, i):
        if self.puc is None:
            return x
        w, b = self.puc[i]
        return x + _conv1d_same(x, w.astype(x.dtype), b.astype(x.dtype))

    def _final_ln(self, h):
        return mx.fast.layer_norm(h, self.G["transformer.ln_f.weight"], self.G.get("transformer.ln_f.bias"), 1e-6)

    def _mask(self, L_cur, pbias):
        P = self.P
        m = np.zeros((P + L_cur, P + L_cur), np.float32)
        m[:P, P:] = NEG_INF
        mask = mx.array(m)[None, None]
        if pbias is not None:
            mask = mask + mx.pad(pbias, [(0, 0), (0, 0), (0, 0), (0, L_cur)])
        return mask

    @staticmethod
    def _sample(logits, temperature, top_k, top_p, key=None):
        if temperature == 0 or top_k == 1:
            return mx.argmax(logits, axis=-1)
        lf = logits.astype(mx.float32) / temperature
        if top_k > 0:
            kth = mx.sort(lf, axis=-1)[..., -top_k][..., None]
            lf = mx.where(lf < kth, NEG_INF, lf)
        if top_p < 1.0:
            order = mx.argsort(-lf, axis=-1)
            sl = mx.take_along_axis(lf, order, axis=-1)
            pr = mx.softmax(sl, axis=-1)
            remove = (mx.cumsum(pr, axis=-1) - pr) > top_p
            sl = mx.where(remove, NEG_INF, sl)
            inv = mx.argsort(order, axis=-1)
            lf = mx.take_along_axis(sl, inv, axis=-1)
        return mx.random.categorical(lf, axis=-1, key=key)

    # -- main entry
    def generate(self, prefix, l0=None, *, temperature=1.0, top_k=0, top_p=1.0, cfg_scale=0.0,
                 cfg_null_eos=50256, seed=None, force_codes=None, trace=None):
        """prefix [B,P] raw token ids; l0 [B,levels[0]] optional level-0 codes.
        Returns codes [B, sum(levels)] (mx.int).  `force_codes` [B, n] pins the first n codes
        (a whole number of levels; all of them = teacher forcing) and the rest are sampled;
        `trace` collects per-level logits. Both serve parity checks and hierarchy probing."""
        if seed is not None:
            mx.random.seed(seed)
        prefix = mx.array(np.asarray(prefix), dtype=mx.int32)
        idx = None if l0 is None else mx.array(np.asarray(l0), dtype=mx.int32)
        Bc = prefix.shape[0]
        cfg = None
        if cfg_scale > 0:
            if self.prefix_pad_id is None:
                raise ValueError("cfg_scale needs prefix_pad_id")
            null = mx.full(prefix.shape, int(self.prefix_pad_id), dtype=mx.int32)
            null[:, 0] = cfg_null_eos
            prefix = mx.concatenate([prefix, null], 0)
            if idx is not None:
                idx = mx.concatenate([idx, idx], 0)
            cfg = {"scale": float(cfg_scale), "nlev": len(self.levels)}
        B = prefix.shape[0]
        P = self.P
        nlev = len(self.levels)
        levels = self.levels
        force = None if force_codes is None else mx.array(np.asarray(force_codes), dtype=mx.int32)

        def sample(logits, level, acc, L):
            if cfg is not None:
                t = cfg["scale"] * (level / max(cfg["nlev"] - 1, 1))
                logits = (1.0 + t) * logits[:Bc] - t * logits[Bc:]
            if trace is not None:
                trace.append(logits)
            if force is not None and acc + L <= force.shape[1]:   # forced (possibly only the first levels)
                pred = force[:, acc:acc + L]
            else:
                pred = self._sample(logits, temperature, top_k, top_p)
            return mx.concatenate([pred, pred], 0) if cfg is not None else pred

        prefix_emb = self._wte(self.enc_wte[prefix[:, :P]])

        pbias, pad_mask = None, None
        if self.prefix_pad_id is not None:
            pad_mask = prefix[:, :P] == int(self.prefix_pad_id)
            pbias = mx.where(pad_mask, NEG_INF, 0.0).astype(mx.float32)[:, None, None, :]

        if idx is None and self.query_level0:
            n0 = levels[0]
            q0 = self._puc(mx.broadcast_to(self.l0_queries, (B, n0, self.l0_queries.shape[-1])).astype(self.compute_dtype), 0)
            q = self._wte(q0) + self.level_pos[:, :n0]
            h = self.blocks(mx.concatenate([prefix_emb, q], 1), self._mask(n0, pbias), self.rope_gen[0])
            h = self._final_ln(h)
            idx = sample(self.heads[0](h[:, P:]), 0, 0, n0)
        elif idx is None:
            assert self.predict_level0, "need l0 codes, query_level0 or predict_level0"
            m = mx.zeros((1, 1, P, P)) if pbias is None else pbias
            cos, sin = self.rope_gen[0]
            h = self._final_ln(self.blocks(prefix_emb, m, (cos[:P], sin[:P])))
            n0 = levels[0]
            if pad_mask is None:
                pos = mx.broadcast_to(mx.arange(P - n0, P)[None], (B, n0))
            else:
                last = mx.maximum((~pad_mask).sum(axis=1) - 1, n0 - 1)
                pos = mx.clip(last[:, None] - mx.arange(n0 - 1, -1, -1)[None], 0, P - 1)
            sl = mx.take_along_axis(h, pos[:, :, None], axis=1)
            idx = sample(self.lm_head_l0(sl), 0, 0, n0)
        elif force is not None and cfg is None:
            idx = force[:, :levels[0]]

        out = [idx]
        z_hat = mx.zeros((B, levels[-1], self.vq.D))
        acc_pos = levels[0] if self.query_level0 else 0
        acc_code = levels[0]
        idx_next = idx
        for li in range(nlev - 1):
            zq = self.vq.emb[li][idx_next]
            z_hat, x = self.vq.next_input(li, z_hat, zq)
            hi = li + (1 - self.lvl_first)
            L_cur = levels[li + 1]
            x = self._puc(x.astype(self.compute_dtype), hi)
            x = self._wte(x) + self.level_pos[:, acc_pos:acc_pos + L_cur]
            h = self.blocks(mx.concatenate([prefix_emb, x], 1), self._mask(L_cur, pbias), self.rope_gen[hi])
            h = self._final_ln(h)
            idx_next = sample(self.heads[hi](h[:, P:]), li + 1, acc_code, L_cur)
            out.append(idx_next)
            acc_pos += L_cur
            acc_code += L_cur
            mx.eval(idx_next)   # one graph per level keeps memory flat and mirrors 1 NFE/level
        codes = mx.concatenate(out, 1)
        return codes[:Bc] if cfg is not None else codes

    def decode_logits(self, codes, all_levels=False):
        codes = mx.array(np.asarray(codes), dtype=mx.int32)
        zl = self.vq.codes_to_zlevels(codes)
        if not all_levels:
            r = self.decoder(zl[-1].astype(self.compute_dtype)).astype(mx.float32)
            return np.array(r)
        return [np.array(self.decoder(z.astype(self.compute_dtype)).astype(mx.float32)) for z in zl]

    def sync(self):
        mx.synchronize() if hasattr(mx, "synchronize") else None

    def to_numpy(self, codes):
        return np.array(codes)
