#!/usr/bin/env python
"""Explore Next-Concept Modeling (HazyResearch/ncm) on Apple Silicon.

    python explore.py download                 # fetch checkpoints + eval data from Hugging Face
    python explore.py convert                  # .pt -> MLX safetensors (done lazily otherwise)
    python explore.py sample  --task tinystories --n 3 --levels
    python explore.py summarize --index 0 --levels
    python explore.py translate --text "Das Wetter ist heute schön."
    python explore.py sudoku --difficulty hard --index 3 --levels
    python explore.py encode --task tinystories --text "Once upon a time ..."
    python explore.py bench --task tinystories --batch 1 8 64
    python explore.py eval --task sudoku-hard --n 500
    python explore.py parity --task xsum

Every generation command takes --backend mlx|torch (default mlx).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ncm_mac.tasks import ARTIFACTS, DATA, TASKS, load_task

con = Console()


# ============================================================================ helpers
def session(args, task=None):
    from ncm_mac.session import NCMSession
    t = task or args.task
    with con.status(f"loading {t} on {args.backend}…"):
        return NCMSession(t, backend=args.backend, dtype=getattr(args, "dtype", None))


def sampling_overrides(args):
    return dict(temperature=args.temp, top_k=args.top_k, top_p=args.top_p, cfg_scale=getattr(args, "cfg", None))


def penalty_overrides(args):
    if getattr(args, "no_penalty", False):
        return dict(freq=0.0, mu=0.0, adj="")
    return {}


def show_levels(sess, level_texts, title="coarse → fine"):
    tb = Table(title=title, show_lines=True, expand=True)
    tb.add_column("lvl", justify="right", style="cyan", no_wrap=True)
    tb.add_column("len", justify="right", style="dim", no_wrap=True)
    tb.add_column("decoded text at this level")
    for k, (l, txt) in enumerate(zip(sess.levels, level_texts)):
        tb.add_row(str(k), str(l), txt.strip() or "[dim](empty)[/]")
    con.print(tb)


def timing_line(r, n):
    return (f"[dim]{n} sample(s): generate {r.t_generate*1e3:.0f} ms · decode {r.t_decode*1e3:.0f} ms · "
            f"post {r.t_post*1e3:.0f} ms · total {r.t_total*1e3:.0f} ms[/]")


def add_sampling_args(p, cfg=False):
    p.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None,
                   help="MLX compute dtype for generator/decoder (default: as converted, fp32)")
    p.add_argument("--temp", type=float, default=None, help="temperature (default: task config)")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--levels", action="store_true", help="show the coarse-to-fine decode of every level")
    p.add_argument("--no-penalty", action="store_true",
                   help="plain argmax decoding of the final level (disable the frequency/adjacency penalties)")
    if cfg:
        p.add_argument("--cfg", type=float, default=None, help="classifier-free guidance scale")
        p.add_argument("--mbr", type=int, default=0, metavar="K",
                       help="test-time scaling: sample K candidates, keep the MBR choice "
                            "(sampling defaults to --temp 0.5 --top-k 0 --top-p 1.0 unless given; on 200 XSum docs K=16 gave ROUGE-L 29.7 vs 27.5 greedy)")


def mbr_sampling(args):
    ov = sampling_overrides(args)
    if getattr(args, "mbr", 0):
        ov["temperature"] = 0.5 if args.temp is None else args.temp
        ov["top_k"] = 0 if args.top_k is None else args.top_k
        ov["top_p"] = 1.0 if args.top_p is None else args.top_p
    return ov


# ============================================================================ commands
def cmd_download(args):
    from huggingface_hub import snapshot_download
    tasks = args.tasks or list(TASKS)
    ck_pat, data_pat = ["README.md"], ["README.md"]
    for t in tasks:
        sub, dsub, kind = TASKS[t]
        ck_pat += [f"{sub}/*"]
        if t.startswith("lm1b"):
            ck_pat += ["lm1b/tokenizer.pt"]
        if kind != "uncond":
            data_pat += [f"{dsub}/*"]
    con.print(f"[bold]checkpoints[/] → {ARTIFACTS}")
    snapshot_download("hazyresearch/ncm-checkpoints", local_dir=ARTIFACTS, allow_patterns=ck_pat)
    con.print(f"[bold]eval data[/] → {DATA}")
    snapshot_download("hazyresearch/ncm-datasets", repo_type="dataset", local_dir=DATA, allow_patterns=data_pat)


def cmd_convert(args):
    from ncm_mac.convert import convert
    for t in args.tasks or list(TASKS):
        task = load_task(t)
        if not os.path.exists(os.path.join(task.ckpt_dir, task.cfg.checkpoints.generator)):
            con.print(f"[yellow]skip {t}: checkpoint not downloaded[/]")
            continue
        t0 = time.time()
        out = convert(task, dtype=args.dtype, force=args.force)
        sz = os.path.getsize(os.path.join(out, "weights.safetensors")) / 1e6
        con.print(f"[green]✓[/] {t:14s} → {out}  ({sz:.0f} MB, {time.time()-t0:.1f}s)")


def cmd_sample(args):
    s = session(args)
    if s.task.kind != "uncond":
        sys.exit(f"{args.task} is conditional; use summarize/translate/sudoku")
    r = s.unconditional(args.n, seed=args.seed, l0_mode=args.l0_mode, show_levels=args.levels,
                        **sampling_overrides(args), **penalty_overrides(args))
    for i, txt in enumerate(r.texts):
        if args.levels:
            show_levels(s, r.level_texts[i], title=f"sample {i} — {s.task.name}")
        else:
            con.print(Panel(txt.strip(), title=f"sample {i}", border_style="cyan"))
    con.print(f"[dim]{len(s.levels)} levels (level 0 seeded from the unigram) → {len(s.levels) - 1} generator "
              f"forward passes + 1 decoder pass for {s.levels[-1]} tokens/sample[/]")
    con.print(timing_line(r, args.n))


def _conditional(args, task, label_src, label_ref):
    s = session(args, task)
    ref = None
    if args.text is not None:
        src = args.text
    elif args.file:
        src = open(args.file).read()
    else:
        items, refs = s.load_split("test")
        if items is None:
            sys.exit("test data missing; run `python explore.py download`")
        src, ref = items[args.index], s.ref_text(refs[args.index])
    if args.mbr:
        r = s.mbr([src], args.mbr, seed=args.seed, **mbr_sampling(args), **penalty_overrides(args))
    else:
        r = s.conditional([src], seed=args.seed, show_levels=args.levels,
                          **sampling_overrides(args), **penalty_overrides(args))
    shown = src if isinstance(src, str) else s.text._src_tok.decode(src[: s.text.P])
    shown = shown if len(shown) < 1500 else shown[:1500] + " …"
    con.print(Panel(shown.strip(), title=label_src, border_style="dim"))
    if args.levels and r.level_texts:
        show_levels(s, r.level_texts[0])
    if args.mbr:
        tb = Table(title=f"{args.mbr} sampled candidates (MBR utility = mean similarity to the others)")
        tb.add_column("score", justify="right", style="cyan"); tb.add_column("candidate")
        best = max(sc for _, sc in r.extra["candidates"][0])
        for txt, sc in r.extra["candidates"][0]:
            tb.add_row(f"{sc:.3f}", f"[bold green]{txt.strip()}[/]" if sc == best else txt.strip())
        con.print(tb)
    con.print(Panel(r.texts[0].strip(), title=f"NCM ({s.backend})", border_style="green"))
    if ref is not None:
        con.print(Panel(ref.strip(), title=label_ref, border_style="yellow"))
    con.print(timing_line(r, 1))
    return s, r


def cmd_summarize(args):
    _conditional(args, "xsum", "source document", "reference summary")


def cmd_translate(args):
    _conditional(args, "wmt14", "German source", "reference translation")


def _grid(s81, givens=None, title=""):
    tb = Table(show_header=False, show_lines=False, box=None, padding=(0, 1), title=title)
    for _ in range(11):
        tb.add_column(justify="center")
    for r in range(9):
        if r in (3, 6):
            tb.add_row(*(["─"] * 3 + ["┼"] + ["─"] * 3 + ["┼"] + ["─"] * 3))
        row = []
        for c in range(9):
            if c in (3, 6):
                row.append("│")
            ch = s81[r * 9 + c] if r * 9 + c < len(s81) else "?"
            if ch in "0_.":
                row.append("[dim]·[/]")
            elif givens is not None and givens[r * 9 + c] not in "0_.":
                row.append(f"[bold]{ch}[/]")
            elif givens is not None:
                row.append(f"[green]{ch}[/]")
            else:
                row.append(ch)
        tb.add_row(*row)
    return tb


def sudoku_prompt(puzzle: str, text) -> list[int]:
    """81-char puzzle (digits, 0/./_ for blanks) -> prompt token ids in the training format."""
    cells = [c for c in puzzle if c in "0123456789._"]
    if len(cells) != 81:
        sys.exit(f"puzzle must have 81 cells, got {len(cells)}")
    cells = ["_" if c in "0." else c for c in cells]
    rows = ["".join(cells[i * 9:(i + 1) * 9]) for i in range(9)]
    return text.pre.encode("<BOS>" + "|".join(rows) + "<BOS>")   # same layout as the test set


def cmd_sudoku(args):
    from ncm_mac import metrics
    s = session(args, f"sudoku-{args.difficulty}")
    ref = None
    if args.puzzle:
        prompt = sudoku_prompt(args.puzzle, s.text)
    else:
        items, refs = s.load_split("test")
        prompt, ref = items[args.index], s.ref_text(refs[args.index])
    prompt_txt = s.text.pre.decode(prompt, skip_special_tokens=True)
    givens = metrics.sudoku_cells(prompt_txt).replace("_", "0")[:81]
    r = s.conditional([prompt], seed=args.seed, show_levels=args.levels, **sampling_overrides(args))
    out = r.texts[0]
    cells = metrics.sudoku_cells(out)
    from rich.columns import Columns
    panels = [Panel(_grid(givens), title="puzzle"), Panel(_grid(cells, givens), title=f"NCM ({s.backend})")]
    if ref is not None:
        panels.append(Panel(_grid(metrics.sudoku_cells(ref)), title="solution"))
    con.print(Columns(panels))
    ok = metrics.sudoku_valid(out)
    keeps = all(g in "0" or g == c for g, c in zip(givens, cells))
    msg = f"valid grid: {'[green]yes[/]' if ok else '[red]no[/]'} · respects givens: {'[green]yes[/]' if keeps else '[red]no[/]'}"
    if ref is not None:
        acc = np.mean([a == b for a, b in zip(cells, metrics.sudoku_cells(ref))])
        msg += f" · exact match: {'[green]yes[/]' if out == ref else '[red]no[/]'} · cell acc {100*acc:.1f}%"
    con.print(msg)
    if args.levels:
        tb = Table(title="coarse → fine (decoded board after each level)")
        tb.add_column("lvl", style="cyan"); tb.add_column("codes"); tb.add_column("board (81 cells)")
        for k, txt in enumerate(r.level_texts[0]):
            c = metrics.sudoku_cells(txt)
            tb.add_row(str(k), str(s.levels[k]), c if len(c) <= 81 else c[:81])
        con.print(tb)
    con.print(timing_line(r, 1))


def cmd_encode(args):
    """Round-trip text through the frozen multiscale tokenizer: what does each level keep?"""
    s = session(args)
    n_tok = len(s.text.pre.encode(args.text))
    codes = s.encode([args.text])
    per = s.model.decode_logits(codes, all_levels=True)
    texts = [s.text.decode(lg[0].argmax(-1)) for lg in per]
    show_levels(s, texts, title="tokenizer reconstruction from levels 0..k")
    con.print(f"[dim]{n_tok} text tokens (padded to {s.levels[-1]}) → {sum(s.levels)} codes in "
              f"{len(s.levels)} levels (level sizes {s.levels})[/]")


def cmd_probe(args):
    """Paper Appendix H: do the coarse levels carry the 'concept' of a text?"""
    import pickle
    args.backend = "mlx"
    s = session(args)
    if s.task.kind != "uncond":
        sys.exit("probe works on unconditional tasks (tinystories, lm1b-*)")

    def story(text, index):
        if text:
            return text
        vp = os.path.join(s.task.data_dir, "valid_bpe_8192.pickle")
        if not os.path.exists(vp):
            sys.exit("pass --text (validation stories only ship for tinystories)")
        return pickle.load(open(vp, "rb"))[index]

    T = int(s.task.cfg.tokenizer.vqvae.context_length)

    def show(t):   # what the model sees: the first T tokens
        ids = s.text.pre.encode(t) if isinstance(t, str) else list(t)
        return s.text.decode(ids[:T])

    K = args.keep
    n_keep = s.level_offset(K)
    ov = sampling_overrides(args)
    pen = penalty_overrides(args)
    A = story(args.text, args.index)
    codes_a = s.encode([A])
    tb = Table(show_lines=True, expand=True)
    tb.add_column("", style="cyan", no_wrap=True)
    tb.add_column("text")
    tb.add_row("source", show(A).strip())
    tb.add_row("tokenizer\nround-trip", s.decode_codes(codes_a, **pen)[0].strip())

    if args.mode == "resample":
        title = f"keep levels 0..{K-1} ({n_keep} codes) of the source, resample levels {K}..{len(s.levels)-1}"
        pref = np.repeat(codes_a[:, :n_keep], args.samples, 0)
        out = s.decode_codes(s.regenerate(pref, seed=args.seed, **ov), **pen)
        for i, t in enumerate(out):
            tb.add_row(f"sample {i}", t.strip())
    elif args.mode == "mix":
        B = story(args.text2, args.index2)
        codes_b = s.encode([B])
        tb.add_row("source B", show(B).strip())
        mix = codes_a[:, :n_keep].copy()
        for k in range(K):            # first half of every kept level from A, second half from B
            o, l = s.level_offset(k), s.levels[k]
            mix[:, o + l // 2:o + l] = codes_b[:, o + l // 2:o + l]
        title = f"levels 0..{K-1}: left half of each level from A, right half from B; finer levels generated"
        pref = np.repeat(mix, args.samples, 0)
        for i, t in enumerate(s.decode_codes(s.regenerate(pref, seed=args.seed, **ov), **pen)):
            tb.add_row(f"mix {i}", t.strip())
    else:  # perturb
        rng = np.random.default_rng(args.seed)
        o, l = s.level_offset(1), s.levels[1]
        pos = rng.choice(l, size=min(args.n_codes, l), replace=False)
        pert = codes_a[:, :n_keep].copy()
        V1 = int(s.task.cfg.tokenizer.vqvae.vqvae_vocab_size[1])
        changes = []
        for p in pos:
            new = int(rng.integers(V1))
            changes.append(f"pos {p}: {pert[0, o + p]}→{new}")
            pert[0, o + p] = new
        title = f"level-1 codes replaced ({', '.join(changes)}); levels {K}.. regenerated with the same seed"
        clean = s.decode_codes(s.regenerate(codes_a[:, :n_keep], seed=args.seed, **ov), **pen)[0]
        dirty = s.decode_codes(s.regenerate(pert, seed=args.seed, **ov), **pen)[0]
        tb.add_row("clean", clean.strip())
        tb.add_row("perturbed", dirty.strip())
    tb.title = title
    con.print(tb)


def cmd_bench(args):
    import torch
    from ncm_mac.session import NCMSession
    rows = []
    for backend in args.backends:
        with con.status(f"loading {args.task} on {backend}"):
            s = NCMSession(args.task, backend=backend, dtype=args.dtype if backend == "mlx" else None)
        prompts = None
        if s.task.kind != "uncond":
            items, _ = s.load_split("test")
            prompts = items
        for B in args.batch:
            def once(seed):
                if prompts is None:
                    return s.unconditional(B, seed=seed)
                return s.conditional([prompts[(seed * B + i) % len(prompts)] for i in range(B)], seed=seed)
            once(0)  # warm-up / compile
            res = [once(i + 1) for i in range(args.repeats)]
            g = np.median([r.t_generate for r in res]); d = np.median([r.t_decode for r in res])
            p = np.median([r.t_post for r in res])
            toks = B * s.levels[-1]
            rows.append((backend, B, g, d, p, toks / (g + d + p), toks / (g + d)))
        del s
    tb = Table(title=f"{args.task}: {len(load_task(args.task).cfg.tokenizer.vqvae.vector_quantizer_config.levels.manual_levels)} levels → "
                     f"{load_task(args.task).cfg.tokenizer.vqvae.context_length} tokens / sample")
    for c in ("backend", "batch", "generate ms", "decode ms", "post ms", "tok/s (all)", "tok/s (model only)"):
        tb.add_column(c, justify="right")
    for b, B, g, d, p, tps, tpm in rows:
        tb.add_row(b, str(B), f"{g*1e3:.1f}", f"{d*1e3:.1f}", f"{p*1e3:.1f}", f"{tps:,.0f}", f"{tpm:,.0f}")
    con.print(tb)
    con.print("[dim]'post' = the paper's sequential frequency/adjacency-penalty decoding on CPU (task-dependent).[/]")
    if args.ar_baseline:
        ar_baseline(args)


def ar_baseline(args):
    """Rough autoregressive yardstick: GPT-2 (124M, close in size to the 12x768 NCM generators)
    sampling the same number of tokens with a KV cache on MPS. Not the paper's diffusion baselines."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    T = int(load_task(args.task).cfg.tokenizer.vqvae.context_length)
    tok = AutoTokenizer.from_pretrained("gpt2")
    lm = AutoModelForCausalLM.from_pretrained("gpt2").to("mps").eval()
    tb = Table(title=f"AR yardstick: GPT-2 124M, {T} new tokens, KV cache, MPS")
    for c in ("batch", "ms", "tok/s"):
        tb.add_column(c, justify="right")
    for B in args.batch:
        x = torch.full((B, 1), tok.eos_token_id, device="mps")
        kw = dict(max_new_tokens=T, min_new_tokens=T, do_sample=True, pad_token_id=tok.eos_token_id,
                  attention_mask=torch.ones_like(x))
        with torch.no_grad():
            lm.generate(x, **kw); torch.mps.synchronize()
            ts = []
            for _ in range(args.repeats):
                t0 = time.perf_counter(); lm.generate(x, **kw); torch.mps.synchronize(); ts.append(time.perf_counter() - t0)
        t = float(np.median(ts))
        tb.add_row(str(B), f"{t*1e3:.0f}", f"{B*T/t:,.0f}")
    con.print(tb)


def cmd_eval(args):
    from ncm_mac import metrics
    s = session(args)
    metric = str(s.task.cfg.eval.metric)
    bs = args.batch or int(s.task.cfg.eval.batch_size)
    t0 = time.time()
    gen_time = 0.0
    if s.task.kind == "uncond":
        seeds = args.seeds or [42]
        per = []
        for seed in seeds:
            hyps, raw = [], []
            for st in range(0, args.n, bs):
                b = min(bs, args.n - st)
                r = s.unconditional(b, seed=seed * 100003 + st, **sampling_overrides(args), **penalty_overrides(args))
                gen_time += r.t_total
                hyps += r.texts; raw += [list(x) for x in r.extra["raw_ids"]]
            with con.status("scoring with gpt2-large on MPS…"):
                if metric == "genppl_flm":
                    ppl = metrics.genppl_flm(s.text._src_tok.batch_decode(raw), str(s.task.cfg.eval.ref_model))
                    ent = float(np.mean([metrics.unigram_entropy(x) for x in raw]))
                else:
                    ppl = metrics.gen_ppl(hyps, str(s.task.cfg.eval.ref_model))
                    ent = float(np.mean([metrics.unigram_entropy(s.text.pre.encode(h)) for h in hyps if h]))
            per.append((seed, ppl, ent))
            con.print(f"seed {seed}: genPPL {ppl:.2f}  entropy {ent:.3f}")
        res = {"genPPL": np.mean([p[1] for p in per]), "entropy": np.mean([p[2] for p in per])}
        n = args.n * len(seeds)
    else:
        items, refs = s.load_split("test")
        n = min(args.n, len(items))
        hyps = []
        if args.mbr:
            bs = max(1, (args.batch or 256) // args.mbr)   # prompts per batch; rows = prompts * K
        for st in range(0, n, bs):
            if args.mbr:
                r = s.mbr(items[st:min(st + bs, n)], args.mbr, seed=args.seed + st, **mbr_sampling(args), **penalty_overrides(args))
            else:
                r = s.conditional(items[st:min(st + bs, n)], seed=args.seed, **sampling_overrides(args), **penalty_overrides(args))
            gen_time += r.t_total
            hyps += r.texts
            con.print(f"[dim]  {min(st + bs, n)}/{n}[/]", end="\r")
        refs = [s.ref_text(x) for x in refs[:n]]
        with con.status("scoring…"):
            res = {"bleu": metrics.bleu, "rouge": metrics.rouge, "exact": metrics.exact}[metric](hyps, refs)
    tb = Table(title=f"{s.task.name} · {n} items · backend {s.backend}")
    for k in res:
        tb.add_column(k, justify="right")
    tb.add_row(*[f"{v:.2f}" if isinstance(v, float) else str(v) for v in res.values()])
    con.print(tb)
    con.print(f"[dim]generation wall-clock {gen_time:.1f}s ({n/gen_time:.1f} items/s, batch {bs}); total {time.time()-t0:.1f}s[/]")


def cmd_parity(args):
    """Teacher-forced comparison of MLX vs the upstream PyTorch model (same codes, compare logits)."""
    from ncm_mac.session import NCMSession
    if args.mlx_cpu:
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
    tasks = args.tasks or [t for t in TASKS if os.path.exists(os.path.join(load_task(t).ckpt_dir, load_task(t).cfg.checkpoints.generator))]
    tb = Table(title="MLX vs PyTorch (teacher-forced, greedy)")
    for c in ("task", "ref", "max|Δ|", "relΔ", "argmax=", "dec argmax=", "free-run=", "same text"):
        tb.add_column(c, justify="right")
    for t in tasks:
        with con.status(f"{t}: loading both backends"):
            ts = NCMSession(t, backend="torch", device=args.device)
            ms = NCMSession(t, backend="mlx")
        if ts.task.kind == "uncond":
            import torch
            l0 = ts.text.sample_level0(args.n, torch.Generator().manual_seed(1))
            pre = ts.text.uncond_prefix(args.n)
        else:
            items, _ = ts.load_split("test")
            pre = np.array([ts.text.prefix(x) for x in items[:args.n]])
            l0 = None
        kw = ts.sampling()
        kw.pop("temperature")
        codes_t, logs_t = ts.model.generate_traced(pre, l0, **kw)
        trace = []
        ms.model.generate(pre, l0, temperature=0, force_codes=codes_t, trace=trace, **{k: v for k, v in kw.items()})
        logs_m = [np.array(x) for x in trace]
        mx_abs, rel, agree, tot = 0.0, 0.0, 0, 0
        for a, b in zip(logs_t, logs_m):
            a = a.reshape(-1, a.shape[-1]); b = b.reshape(-1, b.shape[-1])[:, :a.shape[-1]]
            mx_abs = max(mx_abs, float(np.abs(a - b).max()))
            rel = max(rel, float(np.abs(a - b).max() / (np.abs(a).max() + 1e-9)))
            agree += int((a.argmax(-1) == b.argmax(-1)).sum()); tot += a.shape[0]
        dt = ts.model.decode_logits(codes_t); dm = ms.model.decode_logits(codes_t)
        dagree = float((dt.argmax(-1) == dm.argmax(-1)).mean())
        free_m = np.array(ms.model.generate(pre, l0, temperature=0, **kw))
        free_agree = float((free_m == codes_t).mean())
        txt_t = [ts.text.decode(x) for x in ts.text.pick_tokens(dt)]
        txt_m = [ms.text.decode(x) for x in ms.text.pick_tokens(ms.model.decode_logits(free_m))]
        same = sum(a == b for a, b in zip(txt_t, txt_m))
        tb.add_row(t, ts.model.device, f"{mx_abs:.2e}", f"{rel:.1e}", f"{100*agree/tot:.2f}%", f"{100*dagree:.2f}%",
                   f"{100*free_agree:.1f}%", f"{same}/{len(txt_t)}")
        del ts, ms
    con.print(tb)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="download checkpoints + eval data")
    p.add_argument("--tasks", nargs="*", choices=list(TASKS))
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("convert", help="convert .pt checkpoints to MLX safetensors")
    p.add_argument("--tasks", nargs="*", choices=list(TASKS))
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("sample", help="unconditional generation (tinystories, lm1b-*)")
    p.add_argument("--task", default="tinystories", choices=[t for t, v in TASKS.items() if v[2] == "uncond"])
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--l0-mode", default="unigram", help="unigram | uniform | active | smooth:<eps>")
    add_sampling_args(p)
    p.set_defaults(fn=cmd_sample)

    for name, fn, hlp in (("summarize", cmd_summarize, "XSum one-sentence summarisation"),
                          ("translate", cmd_translate, "WMT14 German→English translation")):
        p = sub.add_parser(name, help=hlp)
        g = p.add_mutually_exclusive_group()
        g.add_argument("--text"); g.add_argument("--file"); g.add_argument("--index", type=int, default=0)
        add_sampling_args(p, cfg=True)
        p.set_defaults(fn=fn)

    p = sub.add_parser("sudoku", help="solve a Sudoku in one shot (one forward pass per level)")
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--index", type=int, default=0)
    g.add_argument("--puzzle", help="81 chars, 0 . or _ for blanks (whitespace / separators ignored)")
    add_sampling_args(p)
    p.set_defaults(fn=cmd_sudoku)

    p = sub.add_parser("encode", help="tokenizer round-trip: what each level of the code hierarchy keeps")
    p.add_argument("--task", default="tinystories", choices=list(TASKS))
    p.add_argument("--text", required=True)
    p.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    p.set_defaults(fn=cmd_encode)

    p = sub.add_parser("probe", help="paper App. H: keep/mix/perturb coarse levels of a story, regenerate the rest")
    p.add_argument("mode", choices=["resample", "mix", "perturb"])
    p.add_argument("--task", default="tinystories", choices=[t for t, v in TASKS.items() if v[2] == "uncond"])
    p.add_argument("--text"); p.add_argument("--index", type=int, default=0, help="validation story index")
    p.add_argument("--text2"); p.add_argument("--index2", type=int, default=1, help="second story (mix)")
    p.add_argument("--keep", type=int, default=3, help="number of coarse levels taken from the source(s)")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--n-codes", type=int, default=2, help="perturb: level-1 codes to replace")
    add_sampling_args(p)
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("bench", help="wall-clock benchmark, MLX vs PyTorch-MPS (+ optional AR yardstick)")
    p.add_argument("--task", default="tinystories", choices=list(TASKS))
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--backends", nargs="+", default=["mlx", "torch"], choices=["mlx", "torch"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    p.add_argument("--ar-baseline", action="store_true", help="also time GPT-2 124M autoregressive sampling")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("eval", help="score a slice of the paper's evaluation")
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="*", help="unconditional tasks: seeds (paper: 42 0 1 2 3)")
    add_sampling_args(p, cfg=True)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("parity", help="check the MLX port against upstream PyTorch")
    p.add_argument("--tasks", nargs="*", choices=list(TASKS))
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--device", default="cpu", help="torch device for the reference (cpu is the strictest)")
    p.add_argument("--mlx-cpu", action="store_true",
                   help="run MLX on CPU (full fp32 matmuls) to isolate port correctness from GPU matmul precision")
    p.set_defaults(fn=cmd_parity)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
