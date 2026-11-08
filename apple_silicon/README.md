# NCM on Apple Silicon

Scripts to run Next-Concept Modeling (NCM, arXiv 2610.14162) on a Mac. The main README lists an NVIDIA
(CUDA) GPU as a prerequisite; this folder runs every released model on an M-series Mac instead, and
checks the paper's headline numbers. It uses the code in the rest of this repository unchanged.

## What NCM is, in plain terms

An ordinary language model such as GPT writes **one token at a time**: to produce 128 tokens it runs
the whole network 128 times, each run seeing everything written so far.

NCM writes text **coarse to fine** in a fixed, small number of steps (about 9–12, regardless of length):

1. **A frozen "tokenizer" network** (a *multiscale residual VQ-VAE*) was trained beforehand to compress
   a piece of text into a ladder of **levels**. For TinyStories the ladder has 12 levels holding
   1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108 and 128 *codes* — 641 codes in all.
   - A **code** is just a number that picks one entry from a learned table (the *codebook*) of 4,096
     vectors. Each vector is a point in a 768-dimensional space, i.e. a learned "meaning fragment".
   - Level 0 is a one-code gist of the whole text. Each later level is longer and records only **what
     the levels before it got wrong** (that is what *residual* means), so detail is added level by level.
   - Adding up the vectors from all levels gives a 128-position signal, and a **decoder** network turns
     that into 128 word tokens.
2. **The generator** (the model you actually sample from) is a transformer that predicts **every code of
   a level at once**, in one pass of the network, given the levels already chosen (and, for translation,
   summaries or Sudoku, the input text or puzzle). After the last level, the decoder produces the words.

So 128 tokens need 12 generator passes instead of 128. The paper calls the coarse codes "concepts", and
claims (a) quality comparable to diffusion language models (another family of parallel text generators),
(b) 9–31× faster generation than them, and (c) that the coarse levels really do carry the gist.

You can watch this happen. `encode` shows what the frozen tokenizer keeps at each level of a real
sentence (each row decodes only levels 0..n):

```
lvl  codes  text decoded from levels 0..lvl
  6     52  a time a a red in
  7     64  upon time a little a red in park
  9     92  Once a time a little little found red ball in park
 11    128  Once upon a time, a little dog found a red ball in the park.
```

and `translate --levels` shows the generator building a translation of *"Das Wetter ist heute schön."*:

```
lvl  codes  text decoded so far
  0      1  keyword
  3     14  nice
  6     32  The very nice.
  8     48  weather weather very nice today.
 10     64  The weather is very nice today.
```

## Released models

| task name | what it does | levels (codes per level) | generator size |
|---|---|---|---|
| `tinystories` | writes short children's stories from nothing | 12 (1 → 128) | 12 layers, 768 wide |
| `lm1b-small` / `-medium` / `-large` | writes news-style sentences from nothing (One Billion Word benchmark); 97M / 317M / 725M parameters | 12 (1 → 128) | 12 / 24 / 36 layers |
| `xsum` | one-sentence summary of a BBC article (up to 512 tokens in) | 9 (4 → 48) | 12 layers |
| `wmt14` | German → English translation (up to 96 tokens in) | 11 (1 → 64) | 12 layers |
| `sudoku-easy` / `-medium` / `-hard` | fills in a Sudoku grid; the codebook has only 16 entries | 12 (1 → 89) | 8 layers, 512 wide |

There is no OpenWebText model: its checkpoint hasn't been released (`owt/TODO.md` on the Hub).

## Two ways to run it: `mlx` and `torch`

| backend | what it is | why it exists |
|---|---|---|
| `mlx` (default) | my rewrite of the generator, tokenizer and decoder in **MLX**, Apple's own array library for M-series chips (`ncm_mac/mlx_ncm.py`) | fastest: 1.6–2× the speed of the PyTorch route |
| `torch` | the **original upstream PyTorch code, unchanged**, run on the Mac GPU through PyTorch's Apple backend (**MPS**, "Metal Performance Shaders") (`ncm_mac/torch_backend.py`) | the reference: if MLX and this disagree, MLX is wrong |

**Why the CUDA requirement didn't matter.** The upstream code only uses CUDA-specific pieces in two
optional places — *flex-attention* (a fast attention kernel) and *Triton* (a compiler for NVIDIA GPU
kernels) — and every released model has them switched off. Everything else is ordinary PyTorch.

**The one thing I had to patch.** Between levels, the code has to shrink the 128-position signal down
to the next level's length, by averaging neighbouring positions. Going from 128 to 64 is easy (average
each pair), but going from 128 to 52 means each output averages about 2.5 inputs, with overlapping
windows. PyTorch's function for this (`adaptive_avg_pool1d`, reached via `interpolate(mode="area")`)
works on Apple GPUs only when the lengths divide evenly, and crashes otherwise. The torch backend
swaps that one call, at run time, for a precomputed 52×128 table of averaging weights and a matrix
multiply. The result is numerically identical; no upstream file is edited.

**Converting the weights.** The checkpoints come as PyTorch `.pt` files (about 8 GB in total).
`explore.py convert` copies the tensors into `.safetensors` files that MLX can load directly, plus a
`config.json` per task, in `mlx_weights/<task>/`. Nothing is retrained or quantised; it takes seconds.

## Setup

```bash
cd apple_silicon                         # from the root of this repository
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python explore.py download     # ~8 GB of checkpoints + small test sets → artifacts/, data/
.venv/bin/python explore.py convert      # .pt → mlx_weights/<task>/weights.safetensors
```

All commands run from `apple_silicon/`. The rest of the repository is needed even with the MLX backend:
it supplies the tokenizers, the reference PyTorch model and the evaluation metrics. (The scripts also
work outside the repo if it is cloned into `upstream/ncm` next to `explore.py`.)

## Commands

All commands are `.venv/bin/python explore.py <command> …`; below, `ncm` stands for that.

### Generate text

```bash
ncm sample --task tinystories --n 3 --levels       # 3 stories from scratch, showing every level
ncm sample --task lm1b-large --n 2 --temp 0.9
ncm summarize --index 7 --levels                   # summarise test article no. 7 (shows the reference too)
ncm translate --text "Die Katze sitzt auf der Matte."
ncm translate --index 3 --cfg 1.0                  # with classifier-free guidance (see below)
ncm summarize --index 7 --mbr 16                   # 16 candidates, keep the consensus one (see below)
```

- `--levels` prints the text decoded after each level, as in the examples above.
- `--temp`, `--top-k`, `--top-p` control randomness when the generator picks codes. Lower temperature =
  safer, more repetitive choices; `--temp 0` always takes the most likely code. Defaults come from each
  task's released config.
- `--seed` makes a run repeatable.
- LM1B samples keep BERT's `[CLS]` / `[SEP]` markers, because the model was trained on several sentences
  packed into one 128-token window and the paper scores them that way.
- `sample --l0-mode` picks how the one-code gist (level 0) is chosen for unconditional tasks: by default
  it is drawn at random with the frequencies seen in training (`unigram`); `uniform`, `active` (any code
  that ever occurred) and `smooth:<eps>` are alternatives to experiment with.

**Two extras from the paper:**

- **Classifier-free guidance** (`--cfg S`, WMT14): at each level the generator runs twice, once with
  the German source and once with it hidden, and the prediction is pushed away from the "no source"
  version by a factor that grows with the level. It costs two passes per level and gained about one
  BLEU point here.
- **Minimum Bayes risk selection** (`--mbr K`): sample K different outputs, then keep the one that
  overlaps most with all the others (ROUGE-L overlap for summaries, character n-gram overlap (chrF) for
  translation). It's a way of buying quality with more compute. With `--mbr`, sampling defaults to
  temperature 0.5 so the candidates differ.

**How the final words are chosen.** The decoder outputs a score for every vocabulary word at every
position, and the word with the highest score is taken. For LM1B and XSum the paper adds two penalties
while doing this: a word that has already appeared more often than usual is marked down, and so is
repeating the word just written. This runs as a loop over the 128 positions on the CPU. `--no-penalty`
switches it off (faster, but more repetition).

### Sudoku

```bash
ncm sudoku --difficulty hard --index 3 --levels
ncm sudoku --puzzle "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
```

`--puzzle` takes 81 characters, row by row, with `.` or `0` for blanks. The model fills the grid in 12
passes with no search or backtracking. The output shows the puzzle next to the model's grid and says
whether the grid is a valid Sudoku and whether it kept the given digits.

### Look inside the tokenizer and the hierarchy

```bash
ncm encode --task tinystories --text "Once upon a time, a little dog found a red ball in the park."
ncm probe resample --task tinystories --index 0 --keep 3     # keep the gist, regenerate the detail
ncm probe mix --task tinystories --index 0 --index2 5 --keep 2
ncm probe perturb --task tinystories --index 0 --n-codes 2
```

- `encode` runs your text through the frozen tokenizer and shows what survives at each level.
- `probe` repeats the paper's Appendix H experiments on a real story (`--index` picks one from the
  TinyStories validation set, or pass `--text`):
  - `resample` keeps the first `--keep` levels of the story's codes and lets the generator invent the rest.
  - `mix` takes coarse levels from two stories and lets the generator fill in the remainder.
  - `perturb` replaces `--n-codes` of the 8 level-1 codes with random ones and regenerates.

### Measure

```bash
ncm bench --task lm1b-small --batch 1 8 64 --ar-baseline
ncm eval --task sudoku-hard --n 2000
ncm eval --task wmt14 --n 3003 [--cfg 1.0] [--mbr 8]
ncm eval --task xsum --n 2000
ncm eval --task tinystories --n 256 --seeds 42 0 1 2 3
ncm parity --n 8 [--mlx-cpu]
```

- `bench` times generation at several **batch sizes** (how many samples are produced together), on both
  backends; `--ar-baseline` also times GPT-2 small writing the same number of tokens one at a time.
- `eval` scores the model on the first `--n` items of a test set, using the paper's metrics (explained
  under Findings).
- `parity` feeds the same inputs to the MLX rewrite and to the original PyTorch code and reports how
  far apart their outputs are. `--mlx-cpu` runs MLX on the CPU, which rules out GPU rounding as the cause
  of any difference (see the precision note below).

Common flags: `--backend mlx|torch`, `--dtype bfloat16` (run MLX at half precision; faster, slightly
less exact).

## Findings on an M5 Max (8 Nov 2026)

All runs used the MLX backend with each task's released settings. Raw outputs are in `results/`.

### What the metrics mean

- **Sudoku solved %**: share of puzzles where every cell is correct.
- **BLEU** (0–100): how many 1- to 4-word sequences of the translation also appear in the human
  reference translation. "Cased" counts *The* and *the* as different words; "lowercased" doesn't.
- **ROUGE-1 / 2 / L** (0–100): overlap with the human-written summary in single words, word pairs, and
  the longest common in-order word sequence.
- **gen-PPL** (generative perplexity, lower is better): how surprised a large independent model
  (GPT-2-large) is by the generated text. Fluent text scores low. Very repetitive text also scores low,
  which is why it's paired with:
- **entropy** (higher = more varied wording): of the word frequencies within each sample. Real text sits
  around 4; repetitive text falls well below.

### Quality: the paper's claims reproduce

| task | metric | this Mac | paper | notes |
|---|---|---|---|---|
| Sudoku easy / medium / hard | solved % (2,000 puzzles each) | **99.40 / 96.35 / 86.20** | 99.5 / 96.5 / 86.1 | every solved grid is also a valid Sudoku |
| WMT14 German → English | BLEU (all 3,003 test sentences) | 21.73 cased, **22.63 lowercased** | 22.6 | the paper's figure matches the lowercased score |
| WMT14 with guidance (`--cfg 1.0`) | BLEU | 22.95 cased, **23.86 lowercased** | 23.9 | |
| XSum | ROUGE-1 / 2 / L (first 2,000 of 11,334 articles) | **34.73 / 12.51 / 27.75** | 35.0 / 12.8 / 28.0 | |
| XSum, best of 16 (`--mbr 16`) | ROUGE-L (first 200 articles) | **29.74** (single greedy output on the same 200: 27.47) | 29.6 (greedy 28.0) | at temperature 0.5; 0.8 gave 29.55. The paper doesn't say which it used |
| TinyStories | gen-PPL / entropy (256 samples, 1 seed) | **15.58 / 3.96** | 15.41 / 3.98 | |
| LM1B-small | gen-PPL / entropy (256 samples, 1 seed) | **95.36 / 4.31** | 93.05 / 4.31 | the paper averages 5 seeds |

**The "concept" claim also holds.** On a TinyStories story:
- Keeping only the first 3 levels (25 of 641 codes) and regenerating the rest gives new stories with the
  same characters and theme.
- Replacing two level-1 codes changes details ("shiny silver pot" became "big black pot from grandma")
  while the plot survives.
- Mixing two stories' coarse codes gives a story whose first half follows one and second half the other.

### Speed: fast for single samples; the paper's comparison is to diffusion models, not GPT-style models

The paper's 9–31× speed-ups are against diffusion language models on an NVIDIA H100. Those models aren't
available here, so `bench --ar-baseline` compares against GPT-2 small (124M parameters, similar in size
to LM1B-small) writing the same 128 tokens one at a time, which is how most chat models work.

| LM1B-small, 128 tokens, time per batch | 1 sample | 8 samples | 64 samples |
|---|---|---|---|
| NCM on MLX (generate + decode + penalties) | 48 ms | 177 ms | 902 ms |
| NCM on PyTorch-MPS | 93 ms | 298 ms | 1,634 ms |
| GPT-2 small, one token at a time | 237 ms | 364 ms | 855 ms |

- **One sample:** NCM is about 5× faster, because 128 tokens take 12 passes instead of 128.
- **64 samples at once:** GPT-2 catches up. GPT-2 caches its earlier work, so each of its 128 steps is
  cheap and batches well. Each NCM pass, by contrast, processes all the codes chosen so far.
- **Penalties:** the repetition penalties (above) are a one-position-at-a-time loop on the CPU and
  account for about 28% of NCM's time at 64 samples. `--no-penalty` removes that cost.
- **Sudoku-hard:** 23 ms per puzzle one at a time, or about 95 puzzles per second in batches of 128.

### Accuracy of the MLX rewrite, and one precision caveat

Run on the CPU, the MLX rewrite agrees with the original PyTorch code to within 0.001% on every model,
and produces **exactly the same codes and text** on every task (`parity --mlx-cpu`).

On the GPU there is a small difference. The model uses 32-bit floating-point numbers, which are good to
about 7 significant digits, but MLX's matrix multiplications on the Apple GPU come out accurate to only
about 1 part in 1,000 (about 3 digits). That is the kind of speed-for-accuracy trade NVIDIA GPUs make in
their "TF32" mode. PyTorch's Apple backend keeps the full 7 digits. The effect:
- The two backends still pick the same code at 99.5–100% of positions, and give identical Sudoku and
  translation outputs.
- With `--temp 0` on LM1B, though, where the most likely code is always taken, two almost equally likely
  codes occasionally swap, and that change carries through the later levels, so the text differs.
- Normal random sampling isn't affected in any meaningful way. All results above came from the GPU.

## Files

```
explore.py                CLI: all the commands above
ncm_mac/mlx_ncm.py        MLX rewrite: generator, multi-level tokenizer, decoder, sampling, guidance
ncm_mac/torch_backend.py  runs the upstream PyTorch code on MPS (with the pooling patch, and logging for parity)
ncm_mac/convert.py        .pt checkpoints → safetensors + config.json for MLX
ncm_mac/textio.py         text ↔ token ids, input formatting, the repetition-penalty word picker
ncm_mac/session.py        high-level Python API (unconditional / conditional / MBR), with timings
ncm_mac/metrics.py        ROUGE, BLEU/chrF, Sudoku scoring, gen-PPL with GPT-2-large
ncm_mac/tasks.py          where each task's checkpoint, config and test data live
results/                  saved eval outputs behind the Findings tables
artifacts/ data/ mlx_weights/   downloaded checkpoints, test data, converted weights (git-ignored, ~16 GB)
```

Using it from Python:

```python
from ncm_mac.session import NCMSession
s = NCMSession("wmt14")                      # MLX backend by default
r = s.conditional(["Guten Morgen, wie geht es dir?"], show_levels=True)
print(r.texts[0])        # the final translation
print(r.level_texts[0])  # the text after each level
```
