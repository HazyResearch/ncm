import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from utils.registry import level_schedulers


# Phi (quant_resi): per-level residual conv after upsample, blending (1-r)*h + r*Conv1d(h).
class Phi(nn.Conv1d):
    def __init__(self, embed_dim, quant_resi, learnable_ratio=False):
        ks = 3
        super().__init__(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=ks,
            stride=1,
            padding=ks // 2,
        )
        init_ratio = abs(quant_resi)
        self._learnable_ratio = bool(learnable_ratio)
        if self._learnable_ratio:
            r_clamped = min(max(init_ratio, 1e-4), 1 - 1e-4)
            init_logit = math.log(r_clamped / (1 - r_clamped))
            self.ratio_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        else:
            self._ratio_const = init_ratio

    @property
    def resi_ratio(self):
        if self._learnable_ratio:
            return torch.sigmoid(self.ratio_logit)
        return self._ratio_const

    def forward(self, h_BCL):
        r = self.resi_ratio
        return h_BCL * (1 - r) + super().forward(h_BCL) * r


class PhiConvStack(nn.Module):
    def __init__(self, embed_dim, kernel_size=3, hidden_mult=1):
        super().__init__()
        ks = kernel_size
        pad = ks // 2
        h = embed_dim * hidden_mult
        self.c1 = nn.Conv1d(embed_dim, h, kernel_size=ks, padding=pad)
        self.act1 = nn.GELU()
        self.c2 = nn.Conv1d(h, h, kernel_size=ks, padding=pad)
        self.act2 = nn.GELU()
        self.c3 = nn.Conv1d(h, embed_dim, kernel_size=ks, padding=pad)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)

    def forward(self, h_BCL):
        y = self.act1(self.c1(h_BCL))
        y = self.act2(self.c2(y))
        y = self.c3(y)
        return h_BCL + y


class PhiShared(nn.Module):
    def __init__(self, qresi):
        super().__init__()
        self.qresi = qresi

    def __getitem__(self, _):
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1):
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self):
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1):
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self):
        return f'ticks={self.ticks}'


def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key, default)
    if isinstance(cfg, dict) and key in cfg:
        return cfg[key]
    return default


def _build_quant_resi(embed_dim, num_levels, quant_resi_config):
    enabled = bool(_cfg_get(quant_resi_config, 'enabled', True))
    if not enabled:
        return None
    raw_ratio = _cfg_get(quant_resi_config, 'ratio', 0.5)
    share_mode = int(_cfg_get(quant_resi_config, 'share_mode', 0))
    learnable_ratio = bool(_cfg_get(quant_resi_config, 'learnable_ratio', False))
    arch = str(_cfg_get(quant_resi_config, 'arch', 'blend'))
    hidden_mult = int(_cfg_get(quant_resi_config, 'hidden_mult', 1))
    kernel_size = int(_cfg_get(quant_resi_config, 'kernel_size', 3))

    if isinstance(raw_ratio, (list, tuple)) or (hasattr(raw_ratio, '__len__') and not isinstance(raw_ratio, str)):
        ratios_per_level = [float(r) for r in raw_ratio]
        if share_mode != 0:
            raise ValueError(
                f"per-level ratio list requires share_mode=0 (non-shared); got share_mode={share_mode}"
            )
        if len(ratios_per_level) != num_levels:
            raise ValueError(
                f"ratio list must have length {num_levels} (one per level); got {len(ratios_per_level)}"
            )
    else:
        ratios_per_level = [float(raw_ratio)] * num_levels

    def make_phi(r):
        if arch == 'convstack':
            return PhiConvStack(embed_dim, kernel_size=kernel_size, hidden_mult=hidden_mult)
        if learnable_ratio:
            return Phi(embed_dim, r, learnable_ratio=True)
        return Phi(embed_dim, r) if abs(r) > 1e-6 else nn.Identity()

    if share_mode == 0:
        return PhiNonShared([make_phi(r) for r in ratios_per_level])
    if share_mode == 1:
        return PhiShared(make_phi(ratios_per_level[0]))
    return PhiPartiallyShared(nn.ModuleList([make_phi(ratios_per_level[0]) for _ in range(share_mode)]))

def _make_learned_down_l1(D, stride=4):
    conv = nn.Conv1d(D, D, kernel_size=stride, stride=stride, bias=True)
    with torch.no_grad():
        conv.weight.zero_()
        eye_idx = torch.arange(D)
        conv.weight[eye_idx, eye_idx, :] = 1.0 / stride
        conv.bias.zero_()
    return conv


def _make_learned_up_l1(D, stride=4):
    convT = nn.ConvTranspose1d(D, D, kernel_size=stride, stride=stride, bias=True)
    with torch.no_grad():
        convT.weight.zero_()
        eye_idx = torch.arange(D)
        convT.weight[eye_idx, eye_idx, :] = 1.0
        convT.bias.zero_()
    return convT


def _plan_cascade_strides(total_stride, base=4):
    S = int(total_stride)
    strides = []
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


def _cascade_norm(D, groups=8):
    return nn.GroupNorm(groups, D, affine=False)


def _make_cascaded_down_l1(D, total_stride, base=4, inter_norm=False, final_norm=False):
    strides = _plan_cascade_strides(total_stride, base)
    mods = []
    for i, s in enumerate(strides):
        if inter_norm and i > 0:
            mods.append(_cascade_norm(D))
        mods.append(_make_learned_down_l1(D, stride=s))
    if final_norm:
        mods.append(_cascade_norm(D))
    return nn.Sequential(*mods)


def _make_cascaded_up_l1(D, total_stride, base=4, inter_norm=False):
    strides = list(reversed(_plan_cascade_strides(total_stride, base)))
    mods = []
    for i, s in enumerate(strides):
        if inter_norm and i > 0:
            mods.append(_cascade_norm(D))
        mods.append(_make_learned_up_l1(D, stride=s))
    return nn.Sequential(*mods)


def _make_zeroinit_residual_conv(D, kernel_size=3):
    pad = kernel_size // 2
    conv = nn.Conv1d(D, D, kernel_size=kernel_size, stride=1, padding=pad, bias=True)
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
    return conv


class VectorQuantizerBaseClass(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        _ = inputs
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        return cls(**{k: v for k, v in config.items() if k != "clss"})

class MultiscaleResidualVectorQuantizer(VectorQuantizerBaseClass):
    def __init__(
        self,
        vqvae_vocab_size,
        embed_dim,
        levels,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
        quant_resi=None,
        learned_l1_sampling=False,
        learned_all_sampling=False,
        learned_l1_cascade=False,
        learned_l1_cascade_base=4,
        learned_l1_cascade_norm=False,
        learned_l1_cascade_norm_final=False,
        restart_threshold=0.0,
        ema_scale_by_world_size=False,
        kmeans_init=False,
        kmeans_iters=10,
        pre_quant_norm_type='group',
        aux=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        self.ema_scale_by_world_size = bool(ema_scale_by_world_size)
        if isinstance(decay, (list, tuple)):
            self.base_decay = float(decay[0])
            self.base_decay_per_level = [float(x) for x in decay]
        else:
            self.base_decay = float(decay)
            self.base_decay_per_level = None
        # Fine-level dropout: hand the decoder a z_hat truncated at a random level >= min_keep. Train-only.
        _fd = _cfg_get(aux, 'fine_drop', None)
        self.fine_drop_prob = float(_cfg_get(_fd, 'prob', 0.0))
        self.fine_drop_min_keep = int(_cfg_get(_fd, 'min_keep', 2))
        _ps = _cfg_get(_fd, 'prob_start', None)
        self.fine_drop_prob_start = float(_ps) if _ps is not None else None
        self.fine_drop_anneal_steps = int(_cfg_get(_fd, 'anneal_steps', 0) or 0)
        self.register_buffer('_fd_step', torch.zeros((), dtype=torch.long), persistent=False)
        self.learned_all_sampling = bool(learned_all_sampling)
        self.learned_l1_sampling = bool(learned_l1_sampling) or self.learned_all_sampling
        self.learned_l1_cascade = bool(learned_l1_cascade)
        self.learned_l1_cascade_base = int(learned_l1_cascade_base)
        self.learned_l1_cascade_norm = bool(learned_l1_cascade_norm)
        self.learned_l1_cascade_norm_final = bool(learned_l1_cascade_norm_final)
        self.restart_threshold = float(restart_threshold)
        self.kmeans_init = bool(kmeans_init)
        self.kmeans_iters = int(kmeans_iters)
        self._km_buf = [[] for _ in range(len(vqvae_vocab_size))] if self.kmeans_init else None
        if self.kmeans_init:
            self.register_buffer('_km_done',
                                 torch.zeros(len(vqvae_vocab_size), dtype=torch.bool),
                                 persistent=False)

        if levels.use_manual_levels:
            self.levels = levels.manual_levels
        else:
            self.levels = level_schedulers[levels.scheduler](levels.start, levels.end, levels.num_levels)

        assert len(vqvae_vocab_size) == len(self.levels), \
            f"vqvae_vocab_size length {len(vqvae_vocab_size)} must match number of levels {len(self.levels)}"
        self.vqvae_vocab_sizes = list(vqvae_vocab_size)

        self.embeddings = nn.ModuleList([
            nn.Embedding(vs, embed_dim) for vs in self.vqvae_vocab_sizes
        ])
        # Codebooks are updated by EMA, not gradients.
        for emb in self.embeddings:
            emb.weight.data.normal_()
            emb.weight.requires_grad_(False)
        for i, vs in enumerate(self.vqvae_vocab_sizes):
            self.register_buffer(f'ema_cluster_size_{i}', torch.zeros(vs))
            self.register_buffer(f'ema_w_{i}', self.embeddings[i].weight.data.clone())
            self.register_buffer(f'codebook_hit_{i}', torch.zeros(vs, dtype=torch.long))

        self.quant_resi = _build_quant_resi(embed_dim, len(self.levels), quant_resi)
        self.quant_resi_enabled = self.quant_resi is not None
        self._qr_num_levels = len(self.levels)

        # Learned down/upsampler at level 0 only, replacing F.interpolate; init matches area-pool / nearest-repeat.
        if self.learned_l1_sampling:
            L = self.levels[-1]
            ell_1 = self.levels[0]
            stride = L // ell_1
            assert L % ell_1 == 0 and stride * ell_1 == L, \
                f"learned_l1_sampling requires L={L} divisible by ell_1={ell_1}"
            if self.learned_l1_cascade:
                b = self.learned_l1_cascade_base
                nrm = self.learned_l1_cascade_norm
                self.learned_down_l1 = _make_cascaded_down_l1(embed_dim, stride, base=b, inter_norm=nrm,
                                                              final_norm=self.learned_l1_cascade_norm_final)
                self.learned_up_l1 = _make_cascaded_up_l1(embed_dim, stride, base=b, inter_norm=nrm)
            else:
                self.learned_down_l1 = _make_learned_down_l1(embed_dim, stride=stride)
                self.learned_up_l1 = _make_learned_up_l1(embed_dim, stride=stride)
        else:
            self.learned_down_l1 = None
            self.learned_up_l1 = None

        if self.learned_all_sampling:
            num_levels = len(self.levels)
            self.learned_down_refine = nn.ModuleList([
                _make_zeroinit_residual_conv(embed_dim, kernel_size=3) if 1 <= i <= num_levels - 2 else nn.Identity()
                for i in range(num_levels - 1)
            ])
            self.learned_up_refine = nn.ModuleList([
                _make_zeroinit_residual_conv(embed_dim, kernel_size=3) if 1 <= i <= num_levels - 2 else nn.Identity()
                for i in range(num_levels - 1)
            ])
        else:
            self.learned_down_refine = None
            self.learned_up_refine = None

    @torch.no_grad()
    def kmeans_init_level(self, level_idx, samples, n_iters=20):
        D = samples.shape[1]
        V = self.vqvae_vocab_sizes[level_idx]
        device = samples.device
        if samples.shape[0] >= V:
            perm = torch.randperm(samples.shape[0], device=device)[:V]
            centers = samples[perm].clone().contiguous().float()
        else:
            idx = torch.randint(0, samples.shape[0], (V,), device=device)
            centers = samples[idx].clone().contiguous().float()
        N = samples.shape[0]
        chunk = max(1, min(N, 8192))
        for it in range(n_iters):
            assigns = torch.empty(N, dtype=torch.long, device=device)
            for i in range(0, N, chunk):
                xs = samples[i:i+chunk].float()
                d2 = (xs**2).sum(1, keepdim=True) + (centers**2).sum(1) - 2 * (xs @ centers.T)
                assigns[i:i+chunk] = d2.argmin(dim=1)
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(V, device=device)
            new_centers.index_add_(0, assigns, samples.float())
            counts.index_add_(0, assigns, torch.ones_like(assigns, dtype=torch.float32))
            mask = counts > 0
            new_centers[mask] /= counts[mask].unsqueeze(1)
            empty = (~mask).nonzero(as_tuple=False).flatten()
            if empty.numel() > 0:
                rand_idx = torch.randint(0, N, (empty.numel(),), device=device)
                new_centers[empty] = samples[rand_idx].float()
            centers = new_centers
        self.embeddings[level_idx].weight.data.copy_(centers.to(self.embeddings[level_idx].weight.dtype))
        ema_cs = getattr(self, f'ema_cluster_size_{level_idx}')
        ema_w  = getattr(self, f'ema_w_{level_idx}')
        ema_cs.fill_(1.0)
        ema_w.copy_(centers)

    def _apply_quant_resi(self, h_BLC, level_idx):
        if not self.quant_resi_enabled:
            return h_BLC
        h_BCL = h_BLC.transpose(-1, -2).contiguous()
        at = level_idx / max(self._qr_num_levels - 1, 1)
        h_BCL = self.quant_resi[at](h_BCL)
        return h_BCL.transpose(-1, -2).contiguous()

    def _distances(self, flat_input, emb_w):
        return (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(emb_w**2, dim=1)
            - 2 * torch.matmul(flat_input, emb_w.t())
        )

    def _down_for_quant(self, z_BLC, level_idx, target_len):
        z_BCL = z_BLC.transpose(-1, -2)
        if level_idx == 0 and self.learned_l1_sampling:
            return self.learned_down_l1(z_BCL).transpose(-1, -2).contiguous()
        zd_BCL = F.interpolate(z_BCL, size=(target_len,), mode="area")
        if self.learned_all_sampling and 1 <= level_idx <= len(self.levels) - 2:
            zd_BCL = zd_BCL + self.learned_down_refine[level_idx](zd_BCL)
        return zd_BCL.transpose(-1, -2).contiguous()

    def _up_to_full(self, q_BLC, level_idx, target_len):
        q_BCL = q_BLC.transpose(-1, -2)
        if level_idx == 0 and self.learned_l1_sampling:
            return self.learned_up_l1(q_BCL).transpose(-1, -2).contiguous()
        qu_BCL = F.interpolate(q_BCL, size=(target_len,), mode="linear")
        if self.learned_all_sampling and 1 <= level_idx <= len(self.levels) - 2:
            qu_BCL = qu_BCL + self.learned_up_refine[level_idx](qu_BCL)
        return qu_BCL.transpose(-1, -2).contiguous()

    def _get_ema_decay(self):
        return self._scale_decay(self.base_decay)

    def _scale_decay(self, d):
        if not getattr(self, "ema_scale_by_world_size", False):
            return d
        world_size = (
            dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        )
        return 1.0 - (1.0 - d) / world_size

    def _get_ema_decay_level(self, i):
        if getattr(self, "base_decay_per_level", None) is None:
            return self._get_ema_decay()
        return self._scale_decay(self.base_decay_per_level[min(i, len(self.base_decay_per_level) - 1)])

    def _current_fine_drop_prob(self):
        if self.fine_drop_prob_start is None or self.fine_drop_anneal_steps <= 0:
            return self.fine_drop_prob
        t = float(self._fd_step.item()) / float(self.fine_drop_anneal_steps)
        t = min(max(t, 0.0), 1.0)
        return self.fine_drop_prob_start + t * (self.fine_drop_prob - self.fine_drop_prob_start)

    @torch.no_grad()
    def _restart_dead_codes(self, flat_input, ema_cs, ema_w, emb_i):
        dead = ema_cs < self.restart_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0 or flat_input.shape[0] == 0:
            return
        src = torch.randint(0, flat_input.shape[0], (n_dead,), device=flat_input.device)
        new_vecs = flat_input[src].to(emb_i.weight.dtype)
        emb_i.weight.data[dead] = new_vecs
        ema_w[dead] = new_vecs
        ema_cs[dead] = self.restart_threshold

    def forward(self, z):
        z = z.float()
        B, L, C = z.shape
        z_no_grad = z.detach()

        z_rest = z_no_grad.clone()
        z_hat = torch.zeros_like(z_rest)

        z_hat_keep = []
        _fd_p = self._current_fine_drop_prob()
        if self.training:
            self._fd_step += 1
        do_fine_drop = (self.training and _fd_p > 0.0
                        and float(torch.rand(())) < _fd_p)
        track_cum = do_fine_drop

        vq_loss = 0
        num_levels = len(self.levels)
        decay = self._get_ema_decay()

        for i, level in enumerate(self.levels):
            emb_i = self.embeddings[i]
            vs_i = self.vqvae_vocab_sizes[i]

            if i != num_levels - 1:
                flat_input_BLC = self._down_for_quant(z_rest, i, level)
            else:
                flat_input_BLC = z_rest
            flat_input = flat_input_BLC.reshape(-1, C)

            distances = self._distances(flat_input, emb_i.weight)

            indices = torch.argmin(distances, dim=1).view(B, level)

            encodings = torch.zeros(B * level, vs_i, device=z.device)
            encodings.scatter_(1, indices.view(-1, 1), 1)

            if self.training:
                decay = self._get_ema_decay_level(i)
                ema_cs = getattr(self, f'ema_cluster_size_{i}')
                ema_w = getattr(self, f'ema_w_{i}')
                encodings_sum = encodings.sum(0)
                dw = torch.matmul(encodings.t(), flat_input.detach())
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(encodings_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(dw, op=dist.ReduceOp.SUM)
                ema_cs.data.mul_(decay).add_(encodings_sum, alpha=1 - decay)
                n = ema_cs.data.sum()
                smoothed = (ema_cs.data + self.epsilon) / (n + vs_i * self.epsilon) * n
                ema_w.data.mul_(decay).add_(dw, alpha=1 - decay)
                emb_i.weight.data.copy_(ema_w.data / smoothed.unsqueeze(1))
                if self.restart_threshold > 0:
                    self._restart_dead_codes(flat_input.detach(), ema_cs.data, ema_w.data, emb_i)

            q_BLC = emb_i(indices)
            # Straight-through estimator at level 0 only; at higher levels it diverges within ~100 steps.
            if i == 0 and self.learned_l1_sampling:
                q_BLC = flat_input_BLC + (q_BLC - flat_input_BLC).detach()

            if i != num_levels - 1:
                quantized = self._up_to_full(q_BLC, i, L)
            else:
                quantized = q_BLC

            quantized = self._apply_quant_resi(quantized, i)

            z_hat = z_hat + quantized
            if track_cum:
                z_hat_keep.append(z_hat)
            z_rest = z_rest - quantized

            vq_loss = vq_loss + self.commitment_cost * F.mse_loss(z_hat.detach(), z)
            vq_loss = vq_loss + F.mse_loss(z_hat, z_no_grad)

            getattr(self, f'codebook_hit_{i}')[indices.view(B * level)] = 1

        vq_loss = vq_loss / num_levels
        if do_fine_drop and len(z_hat_keep) >= 2:
            kmin = max(self.fine_drop_min_keep - 1, 0)
            kmax = len(z_hat_keep) - 2
            if kmax >= kmin:
                K = int(torch.randint(kmin, kmax + 1, ()))
                z_hat = z_hat_keep[K]
        z_hat = z + (z_hat - z).detach()

        per_level_codebook_utilization = [
            torch.sum(getattr(self, f'codebook_hit_{i}')).item() / self.vqvae_vocab_sizes[i]
            for i in range(num_levels)
        ]
        total_slots = sum(self.vqvae_vocab_sizes)
        used_slots = sum(
            torch.sum(getattr(self, f'codebook_hit_{i}')).item() for i in range(num_levels)
        )
        global_codebook_utilization = used_slots / total_slots

        return (
            z_hat,
            vq_loss,
            per_level_codebook_utilization,
            global_codebook_utilization,
            0.1
        )

    @torch.no_grad()
    def set_stochastic_encode(self, enabled, temperature=0.5, top_k=10, top_p=0.95):
        self._stoch_encode = bool(enabled)
        self._stoch_temp = float(temperature)
        self._stoch_top_k = int(top_k)
        self._stoch_top_p = float(top_p)

    def _sample_codes(self, distances):
        logits = -distances / self._stoch_temp
        k = self._stoch_top_k
        if k and k > 0:
            kth = torch.topk(logits, min(k, logits.shape[-1]), dim=-1).values[:, -1, None]
            logits = torch.where(logits < kth, torch.full_like(logits, float('-inf')), logits)
        p = self._stoch_top_p
        if p and p < 1.0:
            sl, si = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sl, dim=-1)
            remove = (probs.cumsum(-1) - probs) > p
            sl = sl.masked_fill(remove, float('-inf'))
            logits = torch.full_like(logits, float('-inf')).scatter(-1, si, sl)
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)

    def z_to_idx(self, z, return_zhats=False, return_zlevels=False):
        z = z.float()
        B, L, C = z.shape
        z_no_grad = z.detach()

        z_rest = z_no_grad.clone()
        z_hat = torch.zeros_like(z_rest)

        z_hats = []
        z_levels = []
        idx = []

        num_levels = len(self.levels)

        for i, level in enumerate(self.levels):
            emb_i = self.embeddings[i]

            if i != num_levels - 1:
                flat_input = self._down_for_quant(z_rest, i, level).reshape(-1, C)
            else:
                flat_input = z_rest.reshape(-1, C)

            distances = self._distances(flat_input, emb_i.weight)

            if getattr(self, '_stoch_encode', False):
                indices = self._sample_codes(distances).view(B, level)
            else:
                indices = torch.argmin(distances, dim=1).view(B, level)
            idx.append(indices)

            if i != num_levels - 1:
                quantized = self._up_to_full(emb_i(indices), i, L)
            else:
                quantized = emb_i(indices)
            quantized = self._apply_quant_resi(quantized, i)

            z_hat = z_hat + quantized
            z_rest -= quantized
            z_hats.append(z_hat)
            z_levels.append(
                F.interpolate(
                    z_hat.transpose(-1, -2), size=(level,), mode="area"
                ).transpose(-1, -2)
                if i != num_levels - 1
                else z_hat
            )

        idx = torch.cat(idx, dim=-1)

        response = (idx,)

        if return_zhats:
            response = response + (z_hats,)
        if return_zlevels:
            response = response + (z_levels,)

        if not (return_zhats or return_zlevels):
            return idx
        else:
            return response

    @torch.no_grad()
    def idx_to_zhats(self, idx, return_zlevels=True):
        z_hats = []
        z_levels = []
        z_hat = torch.zeros(
            idx.shape[0], self.levels[-1], self.embed_dim, device=idx.device
        )

        level_accum = 0
        for level_idx in range(len(self.levels)):
            emb_i = self.embeddings[level_idx]
            emb_at_level = emb_i(idx[:, level_accum : level_accum + self.levels[level_idx]])
            if level_idx != len(self.levels) - 1:
                z_q_level = self._up_to_full(emb_at_level, level_idx, self.levels[-1])
            else:
                z_q_level = emb_at_level
            z_q_level = self._apply_quant_resi(z_q_level, level_idx)
            z_hat.add_(z_q_level)
            z_hats.append(z_hat)
            if return_zlevels:
                z_levels.append(
                    F.interpolate(
                        z_hat.transpose(-1, -2),
                        size=(self.levels[level_idx],),
                        mode="area",
                    ).transpose(-1, -2)
                    if level_idx != len(self.levels) - 1
                    else z_hat
                )
            level_accum += self.levels[level_idx]

        results = (z_hats,)
        if return_zlevels:
            results = results + (z_levels,)
        return results

    @torch.no_grad()
    def idx_to_ncm_input(self, idx):
        next_concepts = []
        z_hat = torch.zeros(
            idx.shape[0], self.levels[-1], self.embed_dim, device=idx.device
        )
        n_levels = len(self.levels)

        level_accum = 0
        for level_idx in range(n_levels - 1):
            emb_i = self.embeddings[level_idx]
            emb_at_level = emb_i(idx[:, level_accum : level_accum + self.levels[level_idx]])
            z_q_level = self._up_to_full(emb_at_level, level_idx, self.levels[-1])
            z_q_level = self._apply_quant_resi(z_q_level, level_idx)
            z_hat.add_(z_q_level)
            next_concepts.append(
                F.interpolate(
                    z_hat.transpose(-1, -2),
                    size=(self.levels[level_idx + 1],),
                    mode="area",
                ).transpose(-1, -2)
            )
            level_accum += self.levels[level_idx]

        return torch.cat(next_concepts, dim=1)

    @torch.no_grad()
    def get_next_autoregressive_input(self, level_idx, z_hat, z_q_level):
        n_levels = len(self.levels)
        if level_idx != n_levels - 1:
            upsampled = self._up_to_full(z_q_level, level_idx, self.levels[-1])
            upsampled = self._apply_quant_resi(upsampled, level_idx)
            z_hat.add_(upsampled)
            return z_hat, (F.interpolate(
                z_hat.transpose(-1, -2), size=(self.levels[level_idx + 1],), mode="area"
            ).transpose(-1, -2))
        else:
            z_q_level = self._apply_quant_resi(z_q_level, level_idx)
            z_hat.add_(z_q_level)
            return z_hat, z_hat

    @torch.no_grad()
    def idx_to_z_q(self, indices):
        """Reconstruct at full resolution from any complete prefix of levels."""
        endpoints = []
        total = 0
        for level in self.levels:
            total += level
            endpoints.append(total)
        if indices.ndim != 2 or indices.shape[1] not in endpoints:
            raise ValueError("indices must contain a complete prefix of concept levels")
        num_levels = endpoints.index(indices.shape[1]) + 1
        z_hat = torch.zeros(indices.shape[0], self.levels[-1], self.embed_dim, device=indices.device)

        level_accum = 0
        for level_idx in range(num_levels):
            emb_i = self.embeddings[level_idx]
            emb_at_level = emb_i(indices[:, level_accum : level_accum + self.levels[level_idx]])
            if level_idx != len(self.levels) - 1:
                z_q_level = self._up_to_full(emb_at_level, level_idx, self.levels[-1])
            else:
                z_q_level = emb_at_level
            z_q_level = self._apply_quant_resi(z_q_level, level_idx)
            z_hat.add_(z_q_level)
            level_accum += self.levels[level_idx]
        return z_hat


if __name__ == "__main__":
    vq = MultiscaleResidualVectorQuantizer(
        vqvae_vocab_size=512, embed_dim=256, levels=[16, 32, 64]
    )
    z = torch.randn(1, 64, 256)
    idx = vq.z_to_idx(z)
    print(idx.shape)
    z_q = vq.idx_to_z_q(idx)
    print(z_q.shape)
