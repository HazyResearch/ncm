"""Convert a released NCM checkpoint pair (tokenizer.pt + generator.pt) into one MLX-friendly
safetensors file + config.json. Only tensors needed for inference are kept:

    gen.*        next-concept generator (NextConceptTransformer state dict)
    vq.*         multiscale residual VQ: codebooks, quant_resi convs, learned level-0 upsampler
    dec.*        VQ-VAE decoder (latent -> raw-token logits)
    enc_wte      VQ-VAE encoder token embedding (used to embed the raw-token prefix)
    inference.*  token / level-0 unigram statistics shipped with the generator
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from .tasks import Task
from .torch_backend import _torch_load

DROP_VQ = ("ema_", "codebook_hit_", "learned_down", "_fd_step", "_km_done")


def convert(task: Task, dtype: str = "float32", force: bool = False) -> str:
    import mlx.core as mx

    out_dir = task.mlx_dir
    out_w = os.path.join(out_dir, "weights.safetensors")
    if os.path.exists(out_w) and not force:
        return out_dir
    os.makedirs(out_dir, exist_ok=True)

    tok_sd = _torch_load(task.cfg.tokenizer.vqvae.checkpoint_path)["model"]
    gen_ck = _torch_load(os.path.join(task.ckpt_dir, task.cfg.checkpoints.generator))
    gen_sd = gen_ck["model"]

    np_dtype = {"float32": np.float32, "float16": np.float16, "bfloat16": np.float32}[dtype]
    arrays: dict[str, mx.array] = {}

    def put(name, t, cast=True):
        a = t.detach().float().cpu().numpy() if t.is_floating_point() else t.cpu().numpy()
        arr = mx.array(a.astype(np_dtype) if (cast and a.dtype == np.float32) else a)
        if cast and dtype == "bfloat16" and arr.dtype == mx.float32:
            arr = arr.astype(mx.bfloat16)
        arrays[name] = arr

    for k, v in gen_sd.items():
        put("gen." + k, v)
    for k, v in tok_sd.items():
        if k.startswith("vq."):
            if any(s in k for s in DROP_VQ):
                continue
            put(k, v, cast=False)                       # keep VQ math in fp32
        elif k.startswith("decoder."):
            put("dec." + k[len("decoder."):], v)
        elif k == "encoder.wte.weight":
            put("enc_wte", v)
    for k, v in (gen_ck.get("inference") or {}).items():
        if torch.is_tensor(v):
            put("inference." + k, v, cast=False)

    mx.save_safetensors(out_w, arrays)
    meta = {"task": task.name, "dtype": dtype, "config": OmegaConf.to_container(task.cfg, resolve=True)}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return out_dir
