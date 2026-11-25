"""Task registry: where configs, checkpoints, converted MLX weights and eval data live.

Layout (relative to the project root, i.e. the folder holding explore.py):
    upstream/ncm/               unmodified clone of github.com/HazyResearch/ncm
    artifacts/<ckpt_subdir>/    hf download hazyresearch/ncm-checkpoints
    data/<data_subdir>/         hf download hazyresearch/ncm-datasets (eval subsets only)
    mlx_weights/<task>/         output of `explore.py convert`
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Either a standalone checkout with the upstream repo cloned into upstream/ncm, or this folder living
# inside the upstream repo itself (apple_silicon/ next to models/, config/, ...).
_NESTED = os.path.join(ROOT, "upstream", "ncm")
UPSTREAM = _NESTED if os.path.isdir(_NESTED) else os.path.dirname(ROOT)
ARTIFACTS = os.path.join(ROOT, "artifacts")
DATA = os.path.join(ROOT, "data")
MLX_WEIGHTS = os.path.join(ROOT, "mlx_weights")

# task name -> (checkpoint subdir, data subdir, kind)
#   kind: "uncond" = unconditional text, "cond" = prefix-conditioned text, "sudoku"
TASKS = {
    "tinystories":   ("tinystories",   "tinystories",   "uncond"),
    "lm1b-small":    ("lm1b/small",    "lm1b",          "uncond"),
    "lm1b-medium":   ("lm1b/medium",   "lm1b",          "uncond"),
    "lm1b-large":    ("lm1b/large",    "lm1b",          "uncond"),
    "xsum":          ("xsum",          "xsum",          "cond"),
    "wmt14":         ("wmt14",         "wmt14",         "cond"),
    "sudoku-easy":   ("sudoku/easy",   "sudoku/easy",   "sudoku"),
    "sudoku-medium": ("sudoku/medium", "sudoku/medium", "sudoku"),
    "sudoku-hard":   ("sudoku/hard",   "sudoku/hard",   "sudoku"),
}


def ensure_upstream_importable():
    if UPSTREAM not in sys.path:
        sys.path.insert(0, UPSTREAM)


@dataclass
class Task:
    name: str
    cfg: object          # OmegaConf DictConfig (paths resolved to absolute)
    ckpt_dir: str
    data_dir: str
    kind: str

    @property
    def mlx_dir(self) -> str:
        return os.path.join(MLX_WEIGHTS, self.name)

    @property
    def gen(self):
        return self.cfg.generation


def load_task(name: str) -> Task:
    if name not in TASKS:
        raise SystemExit(f"unknown task {name!r}; choose from {', '.join(TASKS)}")
    ck_sub, data_sub, kind = TASKS[name]
    ck = os.path.join(ARTIFACTS, ck_sub)
    dd = os.path.join(DATA, data_sub)
    cfg_path = os.path.join(ck, "config.yaml")
    if not os.path.exists(cfg_path):  # fall back to the identical copy in the repo
        cfg_path = os.path.join(UPSTREAM, "config", "eval", f"{name}.yaml")
    cfg = OmegaConf.load(cfg_path)
    # Resolve checkpoint-relative paths exactly like scripts/eval_shard.py does.
    cfg.tokenizer.vqvae.checkpoint_path = os.path.normpath(os.path.join(ck, cfg.checkpoints.tokenizer))
    for key, name_ in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if cfg.checkpoints.get(name_):
            cfg.tokenizer.pre_tokenizer[key] = os.path.join(ck, cfg.checkpoints[name_])
    return Task(name=name, cfg=cfg, ckpt_dir=ck, data_dir=dd, kind=kind)


def require_checkpoint(task: Task):
    p = os.path.join(task.ckpt_dir, task.cfg.checkpoints.generator)
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}\nrun: python explore.py download --tasks {task.name}")
