import argparse, glob, os, pickle, sys
import numpy as np
from omegaconf import OmegaConf
sys.path.insert(0, os.environ.get("NCM_ROOT", os.getcwd()))
from evaluation.metrics import sudoku_metrics
from evaluation.shards import load_shards

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
inputs = ap.add_mutually_exclusive_group(required=True)
inputs.add_argument("--glob", help="legacy shard file pattern")
inputs.add_argument("--shards", nargs="+", help="explicit shard files from one run")
ap.add_argument("--run-id", default=None)
ap.add_argument("--score-bs", type=int, default=0, help="reference-model batch size (default: task protocol)")
a = ap.parse_args()
if a.score_bs < 0:
    ap.error("--score-bs must be positive")

CFG = OmegaConf.load(a.config)
paths = a.shards if a.shards is not None else sorted(glob.glob(a.glob))
parts = load_shards(paths, unconditional=str(CFG.eval.metric) in ("genppl", "genppl_flm"), run_id=a.run_id)
rows = sorted((i, h, r) for p in parts for i, h, r in zip(p["idxs"], p["hyps"], p["refs"]))
seeds = [s for p in parts for s in p.get("seeds", [])]
toks = [t for p in parts for t in p.get("toks", [])]
raw_ids = [r for p in parts for r in p.get("raw_ids", [])]
hyps_by_part = [h for p in parts for h in p["hyps"]]
hyps = [h for _, h, _ in rows]
refs = [r for _, _, r in rows]
print(f"\nmerged {len(parts)} shards -> {len(rows)} {CFG.task} outputs\n")


def bleu():
    import sacrebleu
    b = sacrebleu.corpus_bleu(hyps, [refs])
    bl = sacrebleu.corpus_bleu(hyps, [refs], lowercase=True)
    c = sacrebleu.corpus_chrf(hyps, [refs])
    lr = sum(len(h.split()) for h in hyps) / max(sum(len(r.split()) for r in refs), 1)
    print(f"{'BLEU':>8} {'BLEU(lc)':>9} {'chrF':>7} {'len ratio':>10}")
    print(f"{b.score:8.2f} {bl.score:9.2f} {c.score:7.2f} {lr:10.3f}")
    print(f"\n{b.format(signature=True) if hasattr(b, 'format') else ''}")


def rouge():
    import torch
    from rouge_score import rouge_scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluation.gen_ppl import generation_perplexity
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    g2 = AutoTokenizer.from_pretrained("gpt2")
    lm = AutoModelForCausalLM.from_pretrained("gpt2").cuda().eval()

    R = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
    for h, r in zip(hyps, refs):
        v = sc.score(r, h)
        for k in R:
            R[k].append(v[k].fmeasure)
    with torch.no_grad():
        result = generation_perplexity(lm, g2, [""] * len(hyps), hyps, device="cuda",
                                       score_only_generated=True, batch_size=a.score_bs or 32)
        gp = result.mean_ppl
    print(f"{'R-1':>7} {'R-2':>7} {'R-L':>7} {'genPPL':>8} {'words':>7}")
    print(f"{100 * np.mean(R['rouge1']):7.2f} {100 * np.mean(R['rouge2']):7.2f} {100 * np.mean(R['rougeL']):7.2f} "
          f"{gp:8.2f} {np.mean([len(h.split()) for h in hyps]):7.1f}")
    print(f"genPPL: {len(hyps) - result.unscored_examples}/{len(hyps)} examples scored; "
          f"{result.unscored_examples} have no scorable tokens")


def exact():
    result = sudoku_metrics(hyps, refs)
    n, solved = result["n"], result["solved"]
    print(f"{'solved %':>9} {'solved':>7} {'cell %':>7} {'valid %':>8} {'malformed %':>11}")
    print(f"{100 * solved / n:9.2f} {solved:4d}/{n} {100 * result['cell_accuracy']:7.2f} "
          f"{100 * result['valid'] / n:8.2f} {100 * result['malformed'] / n:11.2f}")


def genppl():
    import math
    import torch
    from collections import Counter
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluation.gen_ppl import generation_perplexity
    ref = str(CFG.eval.ref_model)
    rtok = AutoTokenizer.from_pretrained(ref)
    rlm = AutoModelForCausalLM.from_pretrained(ref).cuda().eval()

    def entropy(ids):
        c, m = Counter(ids), len(ids)
        return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0

    per = []
    for seed in dict.fromkeys(seeds):
        sel = [k for k, s_ in enumerate(seeds) if s_ == seed]
        gen = [hyps_by_part[k] for k in sel]
        with torch.no_grad():
            result = generation_perplexity(rlm, rtok, [""] * len(gen), gen, device="cuda", batch_size=a.score_bs or 32)
        ent = float(np.mean([entropy(toks[k]) for k in sel if toks[k]]))
        per.append((seed, result.mean_ppl, ent, result.unscored_examples))
    print(f"{'seed':>5} {'genPPL':>8} {'entropy':>8} {'unscored':>8}")
    for seed, ppl, ent, unscored in per:
        print(f"{seed:5d} {ppl:8.2f} {ent:8.4f} {unscored:8d}")
    g = np.array([p[1] for p in per]); e = np.array([p[2] for p in per])
    sd = g.std(ddof=1) if len(g) > 1 else 0.0
    print(f"\n{'mean':>5} {g.mean():8.2f} {e.mean():8.4f}   (+/- {sd:.2f} over {len(g)} seeds, {len(hyps_by_part)} samples)")


def genppl_flm():
    # FLM protocol: decode with specials kept, rescore the first 128 GPT-2 tokens, corpus-level perplexity
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    gen_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id))
    ref = str(CFG.eval.ref_model)
    rtok = AutoTokenizer.from_pretrained(ref)
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token
    rlm = AutoModelForCausalLM.from_pretrained(ref).cuda().eval()

    def corpus_ppl(texts):
        enc = rtok(texts, return_tensors="pt", return_token_type_ids=False, truncation=True, padding=True, max_length=128)
        ids, attn = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
        nll_sum = tok_sum = 0.0
        with torch.no_grad():
            bs = a.score_bs or 16
            for i in range(0, ids.shape[0], bs):
                s_, m_ = ids[i:i + bs], attn[i:i + bs]
                logits = rlm(s_, attention_mask=m_, use_cache=False)[0].transpose(-1, -2)
                nll = F.cross_entropy(logits[..., :-1], s_[..., 1:], reduction="none")
                first_eos = (s_ == rtok.eos_token_id).cumsum(-1) == 1
                valid = (first_eos[..., 1:] + (s_ != rtok.eos_token_id)[..., 1:]).to(nll.dtype)
                nll_sum += float((nll * valid).sum()); tok_sum += float(valid.sum())
        return float(np.exp(nll_sum / tok_sum))

    def entropy(ids):
        t = torch.tensor(ids)
        return float(torch.special.entr(torch.unique(t, return_counts=True)[1].float() / t.numel()).sum())

    per = []
    for seed in dict.fromkeys(seeds):
        sel = [k for k, s_ in enumerate(seeds) if s_ == seed]
        texts = gen_tok.batch_decode([raw_ids[k] for k in sel])
        per.append((seed, corpus_ppl(texts), float(np.mean([entropy(raw_ids[k]) for k in sel]))))
    print(f"{'seed':>5} {'genPPL':>8} {'entropy':>8}")
    for seed, ppl, ent in per:
        print(f"{seed:5d} {ppl:8.2f} {ent:8.4f}")
    g = np.array([p[1] for p in per]); e = np.array([p[2] for p in per])
    sd = g.std(ddof=1) if len(g) > 1 else 0.0
    print(f"\n{'mean':>5} {g.mean():8.2f} {e.mean():8.4f}   (+/- {sd:.2f} over {len(g)} seeds, {len(raw_ids)} samples)")


{"bleu": bleu, "rouge": rouge, "exact": exact, "genppl": genppl, "genppl_flm": genppl_flm}[str(CFG.eval.metric)]()
