"""The paper's scoring, ported off CUDA (mirrors upstream scripts/eval_merge.py)."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch

from .tasks import ensure_upstream_importable


def _dev():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def bleu(hyps, refs):
    import sacrebleu
    b = sacrebleu.corpus_bleu(hyps, [refs])
    bl = sacrebleu.corpus_bleu(hyps, [refs], lowercase=True)
    c = sacrebleu.corpus_chrf(hyps, [refs])
    lr = sum(len(h.split()) for h in hyps) / max(sum(len(r.split()) for r in refs), 1)
    return {"BLEU": b.score, "BLEU(lc)": bl.score, "chrF": c.score, "len ratio": lr}


def rouge(hyps, refs, with_genppl=True):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    R = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
    for h, r in zip(hyps, refs):
        v = sc.score(r, h)
        for k in R:
            R[k].append(v[k].fmeasure)
    out = {"R-1": 100 * np.mean(R["rouge1"]), "R-2": 100 * np.mean(R["rouge2"]), "R-L": 100 * np.mean(R["rougeL"])}
    if with_genppl:
        out["genPPL(gpt2)"] = gen_ppl(hyps, "gpt2")
    out["words"] = float(np.mean([len(h.split()) for h in hyps]))
    return out


def sudoku_cells(t):
    return t.replace("|", "")


def sudoku_valid(t):
    g = sudoku_cells(t)
    if len(g) != 81 or not g.isdigit() or "0" in g:
        return False
    rows = [g[i * 9:(i + 1) * 9] for i in range(9)]
    cols = ["".join(r[j] for r in rows) for j in range(9)]
    boxes = ["".join(rows[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3))
             for br in (0, 3, 6) for bc in (0, 3, 6)]
    return all(sorted(x) == list("123456789") for x in rows + cols + boxes)


def exact(hyps, refs):
    n = len(hyps)
    ex = sum(h == r for h, r in zip(hyps, refs))
    cell = np.mean([np.mean([a == b for a, b in zip(sudoku_cells(h), sudoku_cells(r))]) for h, r in zip(hyps, refs)])
    val = sum(sudoku_valid(h) for h in hyps)
    return {"solved %": 100 * ex / n, "solved": f"{ex}/{n}", "cell %": 100 * cell, "valid %": 100 * val / n}


_LM_CACHE = {}


def _ref_lm(name):
    if name not in _LM_CACHE:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        lm = AutoModelForCausalLM.from_pretrained(name).to(_dev()).eval()
        _LM_CACHE[name] = (tok, lm)
    return _LM_CACHE[name]


def gen_ppl(texts, ref="gpt2-large", batch_size=16):
    ensure_upstream_importable()
    from evaluation.gen_ppl import generation_perplexity
    tok, lm = _ref_lm(ref)
    with torch.no_grad():
        return generation_perplexity(lm, tok, [""] * len(texts), texts, device=_dev(),
                                     score_only_generated=True, batch_size=batch_size).mean_ppl


def unigram_entropy(ids):
    c, m = Counter(ids), len(ids)
    return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0


def genppl_flm(texts, ref="gpt2-large"):
    """LM1B protocol (FLM): rescore first 128 GPT-2 tokens, corpus-level perplexity."""
    import torch.nn.functional as F
    tok, lm = _ref_lm(ref)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(texts, return_tensors="pt", return_token_type_ids=False, truncation=True, padding=True, max_length=128)
    ids, attn = enc["input_ids"].to(_dev()), enc["attention_mask"].to(_dev())
    nll_sum = tok_sum = 0.0
    with torch.no_grad():
        for i in range(0, ids.shape[0], 16):
            s, m = ids[i:i + 16], attn[i:i + 16]
            logits = lm(s, attention_mask=m)[0].transpose(-1, -2)
            nll = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none")
            first_eos = (s == tok.eos_token_id).cumsum(-1) == 1
            valid = (first_eos[..., 1:] + (s != tok.eos_token_id)[..., 1:]).to(nll.dtype)
            nll_sum += float((nll * valid).sum())
            tok_sum += float(valid.sum())
    return float(np.exp(nll_sum / tok_sum))
