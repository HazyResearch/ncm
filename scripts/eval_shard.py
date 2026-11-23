import argparse, os, pickle, sys
import torch
from omegaconf import OmegaConf
sys.path.insert(0, os.environ.get("NCM_ROOT", os.getcwd()))
from utils.generation import Generator
from utils.inference import add_soft_decision_args, add_speed_args, positive_temperature
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
add_soft_decision_args(ap)
add_speed_args(ap)
ap.add_argument("--freq", type=float, default=None)
ap.add_argument("--mu", type=float, default=None)
ap.add_argument("--adj", default=None)
ap.add_argument("--l0", default="", help="unigram (default), active, uniform, or smooth:<eps> (mix eps of uniform into the unigram)")
ap.add_argument("--out", required=True)
ap.add_argument("--benchmark-config", help=argparse.SUPPRESS)
a = ap.parse_args()

if a.benchmark_config:
    benchmark_cfg = OmegaConf.load(a.benchmark_config)
    torch.set_num_threads(int(benchmark_cfg.benchmark.cpu_threads))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

CFG = OmegaConf.load(a.config)
G = CFG.generation
ck = a.checkpoint_dir or os.path.join("artifacts", CFG.task)
dd = a.data_dir or os.path.join(os.environ.get("NCM_DATA_ROOT", "data"), CFG.task)
pick = lambda cli, key, cast, default: cast(cli if cli is not None else G.get(key, default))
temp = pick(a.temp, "temperature", float, 1.0)
top_k = pick(a.top_k, "top_k", int, 1)
top_p = pick(a.top_p, "top_p", float, 0.0)
cfg = pick(a.cfg, "cfg_scale", float, 0.0)
soft_decisions = pick(a.soft_decisions, "soft_decisions", bool, False)
soft_temperature = positive_temperature(pick(a.soft_temperature, "soft_temperature", float, 1.0))
freq = pick(a.freq, "frequency_penalty", float, 0.0)
mu = pick(a.mu, "frequency_deficit", float, 0.0)
adj = str(a.adj if a.adj is not None else G.get("adjacency_penalty", "") or "")
split = a.split or CFG.eval.get("split", "test")
n = a.n or int(CFG.eval.get("n_docs", 0))
bs = a.bs or int(CFG.eval.batch_size)

G.update(temperature=temp, top_k=top_k, top_p=top_p, cfg_scale=cfg,
         frequency_penalty=freq, frequency_deficit=mu, adjacency_penalty=adj,
         soft_decisions=soft_decisions, soft_temperature=soft_temperature)
runtime = Generator(CFG, ck, a)
tk, net = runtime.tokenizer, runtime.model
P = int(CFG.model.prefix_len)
EOS, PAD = runtime.eos, runtime.pad
ALLOW = runtime.allowance

L0_MODE = a.l0 or "unigram"
L0_P = runtime.initial_probabilities
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
    items = pickle.load(open(os.path.join(dd, CFG.data[split]), "rb"))
    refs = pickle.load(open(os.path.join(dd, CFG.data[f"{split}_refs"]), "rb"))
    assert len(items) == len(refs), f"{len(items)} inputs vs {len(refs)} references"
    if not a.benchmark_config:
        items, refs = items[:n], refs[:n]
    idxs = list(range(a.shard, len(items), a.nshards))
    src_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id)) if isinstance(items[0], str) else None


def prefix(item):
    ids = src_tok(item)["input_ids"] if src_tok is not None else list(item)
    return (ids[:P] + [PAD] * P)[:P]


CUT_AT_EOS = runtime.cut_at_eos
dec = runtime.decode
generate_tokens = runtime.generate_tokens
RAW = []


def run(x, l0=None):
    ids = generate_tokens(x, l0).tolist()
    RAW.extend(ids)
    return [dec(r) for r in ids]


if a.benchmark_config:
    from utils.benchmarking import benchmark_inference
    benchmark_inference(globals())
    sys.exit(0)


print(f"[inference] soft_decisions={soft_decisions} soft_temperature={soft_temperature} "
      f"cfg={cfg} kv_cache={a.kv_cache} compile={a.compile} resize_matmul={a.resize_matmul} "
      f"cuda_graphs={a.cuda_graphs} precision={a.precision}", flush=True)
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

out["generation"] = {"soft_decisions": soft_decisions, "soft_temperature": soft_temperature,
                     "cfg_scale": cfg, "use_kv_cache": a.kv_cache,
                     "compile": a.compile, "resize_matmul": a.resize_matmul,
                     "cuda_graphs": a.cuda_graphs, "precision": a.precision,
                     "graph_captures": len(generate_tokens.graphs) if a.cuda_graphs else 0,
                     "graph_replays": generate_tokens.replay_count if a.cuda_graphs else 0}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
pickle.dump(out, open(a.out, "wb"))
print(f"shard {a.shard}: {len(hyps)} -> {a.out}")
