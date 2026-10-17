import argparse, os, pickle, sys
import torch
from omegaconf import OmegaConf
sys.path.insert(0, os.environ.get("NCM_ROOT", os.getcwd()))
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint-dir", default="")
ap.add_argument("--data-dir", default="")
ap.add_argument("--split", default="")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--bs", type=int, default=0)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--temp", type=float, default=None)
ap.add_argument("--top_k", type=int, default=None)
ap.add_argument("--top_p", type=float, default=None)
ap.add_argument("--cfg", type=float, default=None)
ap.add_argument("--freq", type=float, default=None)
ap.add_argument("--mu", type=float, default=None)
ap.add_argument("--adj", default=None)
ap.add_argument("--l0", default="", help="unigram (default), active, uniform, or smooth:<eps> (mix eps of uniform into the unigram)")
ap.add_argument("--out", required=True)
ap.add_argument("--kv-cache", action="store_true", help="encode the prefix once and reuse its keys/values at every level")
a = ap.parse_args()

CFG = OmegaConf.load(a.config)
G = CFG.generation
ck = a.checkpoint_dir or os.path.join("artifacts", CFG.task)
dd = a.data_dir or os.path.join(os.environ.get("NCM_DATA_ROOT", "data"), CFG.task)
pick = lambda cli, key, cast, default: cast(cli if cli is not None else G.get(key, default))
temp = pick(a.temp, "temperature", float, 1.0)
top_k = pick(a.top_k, "top_k", int, 1)
top_p = pick(a.top_p, "top_p", float, 0.0)
cfg = pick(a.cfg, "cfg_scale", float, 0.0)
freq = pick(a.freq, "frequency_penalty", float, 0.0)
mu = pick(a.mu, "frequency_deficit", float, 0.0)
adj = str(a.adj if a.adj is not None else G.get("adjacency_penalty", "") or "")
split = a.split or CFG.eval.get("split", "test")
n = a.n or int(CFG.eval.get("n_docs", 0))
bs = a.bs or int(CFG.eval.batch_size)

CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
for key, name in (('vocab_file_path', 'bpe_vocab'), ('merges_file_path', 'bpe_merges')):
    if CFG.checkpoints.get(name):
        CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.cuda().eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).cuda()
sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()
P = int(CFG.model.prefix_len)
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id


def load_allowance(spec):
    if ":" in spec:
        path, ref = spec.split(":", 1)
        obj = torch.load(os.path.join(ck, path), map_location="cuda", weights_only=False)
        for k in ref.split("."):
            obj = obj[k]
        return obj.double()
    return torch.load(os.path.join(ck, spec), map_location="cuda").double()


ALLOW = load_allowance(str(G.frequency_allowance)) if G.get("frequency_allowance") else None
L0_MODE = a.l0 or "unigram"
L0_P = load_allowance(str(G.level0_unigram)).float() if G.get("level0_unigram") else None
if L0_P is not None and L0_MODE == "active":
    L0_P = (L0_P > 0).float() / (L0_P > 0).sum()
elif L0_P is not None and L0_MODE == "uniform":
    L0_P = torch.ones_like(L0_P) / L0_P.numel()
elif L0_P is not None and L0_MODE.startswith("smooth:"):
    eps = float(L0_MODE.split(":", 1)[1])
    L0_P = (1.0 - eps) * L0_P + eps * torch.ones_like(L0_P) / L0_P.numel()
UNCOND = not CFG.data or not CFG.data.get(split)
ADJ = [float(v) for v in adj.split(",")] if adj else None

if UNCOND:
    seeds = [int(x) for x in CFG.eval.seeds][a.shard::a.nshards]
    n = a.n or int(CFG.eval.n_samples)
    items, refs, idxs, src_tok = [], [], [], None
else:
    items = pickle.load(open(os.path.join(dd, CFG.data[split]), "rb"))[:n]
    refs = pickle.load(open(os.path.join(dd, CFG.data[f"{split}_refs"]), "rb"))[:n]
    assert len(items) == len(refs), f"{len(items)} inputs vs {len(refs)} references"
    idxs = list(range(a.shard, len(items), a.nshards))
    src_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id)) if isinstance(items[0], str) else None


def prefix(item):
    ids = src_tok(item)["input_ids"] if src_tok is not None else list(item)
    return (ids[:P] + [PAD] * P)[:P]


def pick_tokens(logits):
    if freq <= 0 and not ADJ:
        return logits.argmax(-1)
    B, T, V = logits.shape
    lg = logits.double()
    allow = (ALLOW[:V] * T)[None, :] if ALLOW is not None else None
    al = ADJ[0] if ADJ else 0.0
    aw = int(ADJ[1]) if ADJ and len(ADJ) > 1 else 1
    ad = ADJ[2] if ADJ and len(ADJ) > 2 else 1.0
    cnt = torch.zeros(B, V, dtype=torch.float64, device=lg.device)
    out = torch.empty(B, T, dtype=torch.long, device=lg.device)
    hist = []
    for t in range(T):
        step = lg[:, t, :].clone()
        if freq > 0:
            excess = (cnt - allow).clamp_min(0.0) if allow is not None else cnt
            step -= freq * excess
            if mu > 0 and allow is not None:
                step += mu * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=torch.float64, device=lg.device))
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=torch.float64, device=lg.device))
    return out


CUT_AT_EOS = bool(G.get("cut_at_eos", True))


def dec(ids):
    ids = list(ids)
    if CUT_AT_EOS and EOS in ids:
        ids = ids[:ids.index(EOS)]
    return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)


RAW = []


def run(x, l0=None):
    with torch.no_grad():
        g = net.generate(l0, x, temperature=temp, top_k=top_k, top_p=top_p, cfg_scale=cfg, cfg_null_eos=EOS, use_kv_cache=a.kv_cache)
        _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        ids = pick_tokens(tk.vqvae.decoder(zl[-1])).tolist()
        RAW.extend(ids)
        return [dec(r) for r in ids]


hyps, sample_seed = [], []
if UNCOND:
    L0 = tk.vqvae.vq.levels[0]
    for seed in seeds:
        torch.manual_seed(seed)
        for s in range(0, n, bs):
            B = min(bs, n - s)
            l0 = torch.multinomial(L0_P.cpu(), B * L0, replacement=True).view(B, L0).cuda() if L0_P is not None else None
            hyps += run(torch.full((B, 1), EOS, device="cuda"), l0)
            sample_seed += [seed] * B
        print(f"  shard{a.shard} seed {seed}: {n} samples", flush=True)
    idxs = list(range(len(hyps)))
    toks = [tk.pre_tokenizer.encode(h) for h in hyps]
    out = {"idxs": idxs, "hyps": hyps, "refs": [""] * len(hyps), "seeds": sample_seed, "toks": toks, "raw_ids": RAW}
else:
    for s in range(0, len(idxs), bs):
        batch = idxs[s:s + bs]
        hyps += run(torch.tensor([prefix(items[i]) for i in batch], device="cuda"))
        if a.shard == 0 and (s // bs) % 10 == 0:
            print(f"  shard0 {s}/{len(idxs)}", flush=True)
    out = {"idxs": idxs, "hyps": hyps, "refs": [r if isinstance(r, str) else dec(r) for r in (refs[i] for i in idxs)]}

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
pickle.dump(out, open(a.out, "wb"))
print(f"shard {a.shard}: {len(hyps)} -> {a.out}")
