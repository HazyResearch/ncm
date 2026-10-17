"""Time generation with and without the prefix KV cache and check that outputs match.

    python scripts/bench_kv.py --config config/eval/wmt14.yaml                    # 1024 inputs, batch 16
    python scripts/bench_kv.py --config config/eval/xsum.yaml --n 256 --bs 64 --cfg 0.5
    python scripts/bench_kv.py --config config/eval/sudoku-easy.yaml --checkpoint-dir artifacts/sudoku/easy \\
        --data-dir data/sudoku/easy --n 2000 --bs 128

Each mode (eager: generate(use_kv_cache=False), kv: use_kv_cache=True) runs the same inputs once after a
warmup batch. Timing covers generation plus decoding to token ids (no detokenization or scoring). Prints
throughput, peak device memory, the share of outputs identical to the first mode, and the task metric
(BLEU for wmt14, ROUGE-L for xsum, solved % for sudoku).
"""
import argparse, json, os, pickle, sys, time
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint-dir", default="")
ap.add_argument("--data-dir", default="")
ap.add_argument("--n", type=int, default=1024)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--cfg", type=float, default=None)
ap.add_argument("--modes", default="eager,kv", help="comma-separated, each eager or kv")
ap.add_argument("--json", default="", help="also write results to this file")
a = ap.parse_args()

CFG = OmegaConf.load(a.config)
G = CFG.generation
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if DEV == "cuda":
    # TF32 changes results with the matmul shapes, which differ between the two paths
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
ck = a.checkpoint_dir or os.path.join(ROOT, "artifacts", CFG.task)
dd = a.data_dir or os.path.join(ROOT, "data", CFG.task)
cfg = a.cfg if a.cfg is not None else float(G.get("cfg_scale", 0.0))
temp, top_k, top_p = float(G.get("temperature", 1.0)), int(G.get("top_k", 1)), float(G.get("top_p", 0.0))
if not (top_k == 1 or temp == 0):
    print(f"note: {CFG.task} samples stochastically (temperature={temp}, top_k={top_k}); outputs will differ between modes", file=sys.stderr)


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()
    elif DEV == "mps":
        torch.mps.synchronize()


def peak_gb():
    if DEV == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if DEV == "mps":  # no peak counter on MPS: memory the driver holds at the end
        return torch.mps.driver_allocated_memory() / 1e9
    return float("nan")


CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.to(DEV).eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(DEV)
net.load_state_dict(torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)["model"], strict=True)
net.eval()
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
P = int(CFG.model.prefix_len)

items = pickle.load(open(os.path.join(dd, CFG.data.test), "rb"))[:a.n]
refs = pickle.load(open(os.path.join(dd, CFG.data.test_refs), "rb"))[:a.n]
src_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id)) if isinstance(items[0], str) else None
X = torch.tensor([((src_tok(i)["input_ids"] if src_tok else list(i))[:P] + [PAD] * P)[:P] for i in items], device=DEV)
n = X.shape[0]


def dec(ids):
    ids = ids[:ids.index(EOS)] if EOS in ids else ids
    return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)


@torch.no_grad()
def run(x, kv):
    torch.manual_seed(0)
    g = net.generate(None, x, temperature=temp, top_k=top_k, top_p=top_p, cfg_scale=cfg, cfg_null_eos=EOS, use_kv_cache=kv)
    _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
    return tk.vqvae.decoder(zl[-1]).argmax(-1)


def metric(hyps):
    ref_txt = [r if isinstance(r, str) else dec(list(r)) for r in refs]
    if CFG.eval.metric == "bleu":
        import sacrebleu
        return "BLEU", sacrebleu.corpus_bleu(hyps, [ref_txt]).score
    if CFG.eval.metric == "rouge":
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return "ROUGE-L", 100 * sum(sc.score(r, h)["rougeL"].fmeasure for r, h in zip(ref_txt, hyps)) / len(hyps)
    if CFG.eval.metric == "exact":
        return "solved%", 100 * sum(h == r for h, r in zip(hyps, ref_txt)) / len(hyps)
    return "-", float("nan")


name = torch.cuda.get_device_name() if DEV == "cuda" else DEV
print(f"[{CFG.task}] device={name} torch={torch.__version__} n={n} bs={a.bs} cfg={cfg}", flush=True)
results, base = [], None
for mode in a.modes.split(","):
    assert mode in ("eager", "kv"), f"bad mode {mode}"
    kv = mode == "kv"
    run(X[:a.bs], kv)  # warmup
    sync()
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    outs, t = [], time.perf_counter()
    for s in range(0, n, a.bs):
        outs.append(run(X[s:s + a.bs], kv))
    sync(); dt = time.perf_counter() - t
    out = torch.cat(outs).cpu()
    base = out if base is None else base
    same = (out == base).all(1).float().mean().item()
    mname, mval = metric([dec(r) for r in out.tolist()])
    r = dict(task=CFG.task, mode=mode, device=name, n=n, bs=a.bs, cfg=cfg, seconds=dt, seq_per_s=n / dt,
             peak_gb=peak_gb(), identical_to_first=same, metric=mname, score=mval)
    results.append(r)
    print(f"{mode:6s} {dt:8.2f}s  {n / dt:8.1f} seq/s  peak {r['peak_gb']:5.2f} GB  identical {same:6.1%}  {mname} {mval:.2f}", flush=True)
if len(results) == 2:
    print(f"speedup {results[1]['mode']} vs {results[0]['mode']}: {results[0]['seconds'] / results[1]['seconds']:.2f}x")
if a.json:
    json.dump(results, open(a.json, "w"), indent=1)
