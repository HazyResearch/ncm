<h1 align="center"><img src="assets/ncm_logo.png" height="28" alt="NCM logo" /> Next-Concept Modeling</h1>

<p align="center"><strong>Fast, Parallel Coarse-to-Fine Text Generation</strong></p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.14162"><img src="https://img.shields.io/badge/Paper-arxiv-b31b1b.svg" alt="Paper" /></a>
  <a href="https://huggingface.co/collections/hazyresearch/ncm"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Collection-NCM-FFD21E" alt="Hugging Face collection" /></a>
  <a href="https://hazyresearch.stanford.edu/blog/2026-10-12-ncm"><img src="https://img.shields.io/badge/Blog-Hazy-b31b1b.svg?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAHKADAAQAAAABAAAAHAAAAABkvfSiAAAETklEQVRIDa1Wz2skRRSuqq6ZTmd%2BJlmTqNFVyOjGgwhevCvIQnBRhGxAZcGDBw8iCB68LuIlF0%2F%2BC6KIV1dUxL0oXrLGxciuht242bibzHSnp3v6R1X5veqZnsQfkKiVoru66r2vvvfeV1Phl555qjo1bbRm%2F0fjQqTdfQlEzthUs16tSGb%2BAzBnaZb3wgiAUms906xPuFU%2FjDhn9AfWnBt8nGQTY1irPtlmbC8IJVyrUgIxjGLOeTJITLNtkoGbp6JaZbA9XjPWsuFNAFAWLsSR8yzL71u9sLh8Lu5119feMzevC1k5HiZZUWy2ieEbL63E7Pxjqy97rdb06YcWz7%2BEPcarJxmJEh%2F8ecUdbcbcRoMJ5yRQI1vOhU4SJkq%2ByOAwiUWORobHfgsCFMG1zWx%2FD%2Bjox3b9O0OLACgASpPn6c525FSyqvdvokV9oT%2Bt8zCI7txiKgOgpJyi7oF%2F0LvhtU8dSgXD8bApRkLIphQXUAqqRikEm4VB1utl%2FVDUPFavwbKUlEAyTZaVgek8D37b9mbnheeZNOFRXwoO5RJQnoOOGsQqjnSa4pOYkaSGCRyCDrFGFPDZuH%2FhiYtr02eWpOclfrB%2FdWPr04%2B6l79wUVUhCtYFZSElqnq4sEdBS56M1efm68%2BeLSYas3OnOp3Fc89vfvzhjxffcbOED9VmwYtTpzXSVWTvH0EP4Q%2BHiG3pxRVZq11563UX2UZYhEmQ3D6oBtCAGCv0ryA0o5Tqd7tFsIVF5%2Bzy3AurKstAljug5hCWfYJmMT6qzfI8WYCtr778bOW5S8tPf37h%2FO7GDwUonmdeeVXVGuBEiM4YkWK3%2FSho6cfY3Z83v3%2FzNXV1vdr3B99%2B893bb8S9XrE%2B0%2Bm0Hn8SMig52gHRZEUCxjhU%2BjHVLAwdlUFPTErZaOZb17Yvf10Yw6jx6JLGb4VTxC4Y8bX9TzklfYwxUQZOPtYIQyB3N9ZLBvXTD5tiCTJAEqBQ0Bw9R9UH3FGm5I8Zuz8GEGO0u1uCuu0p5kgrLMM0jg3xoSwTl9GJIgBMY3bUKEGwIvpoFBqSOFpkDi4FWyWjwZiET6cYTIlHcbBoF5rScVy6JYEPQRU7k53j4IyXqwPfpx2LOGhrWyLaHk04KzP1Vr2WZipRKu91s8lGa7ET3r59Ze1dc2cH1wlRhp6lTPf2VHumvvCAf%2FPGTx%2B8L6MDCp8KQfGBrVut4Eru9Xz%2BySPzD87fE%2FRjP4rhm2nj3Lugowg%2B0p2wEqHYKbmcp0rzmTkT96u4Fh1ptGKIX1PXSjU9rzk58ev1X%2BiuT9JsugUxC1IyAol91J1NtWxQI0RKMUyQ2Vh4FcNd4mY07nP6RwQjpaYatdAP8KMlAXW3F8ABt3YpC5CiMbEDqB0XA1sKWkUnUNQdHZIlDfSDg99v7cCHJIWp3b0uTJEjLqG7AsuCYkwcLbR9EuIwjdaTkI0eDPKor7OMDijnfwAueMlZ7E250AAAAABJRU5ErkJggg%3D%3D" alt="Blog post" /></a>
  <a href="https://scalingintelligence.stanford.edu/blogs/ncm/"><img src="https://img.shields.io/badge/Blog-Scaling-2ea043.svg?logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAIAAAD9b0jDAAAABGdBTUEAALGPC%2FxhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAUGVYSWZNTQAqAAAACAACARIAAwAAAAEAAQAAh2kABAAAAAEAAAAmAAAAAAADoAEAAwAAAAEAAQAAoAIABAAAAAEAAAAcoAMABAAAAAEAAAAcAAAAAHZg2rwAAAIyaVRYdFhNTDpjb20uYWRvYmUueG1wAAAAAAA8eDp4bXBtZXRhIHhtbG5zOng9ImFkb2JlOm5zOm1ldGEvIiB4OnhtcHRrPSJYTVAgQ29yZSA2LjAuMCI%2BCiAgIDxyZGY6UkRGIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyI%2BCiAgICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiCiAgICAgICAgICAgIHhtbG5zOmV4aWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vZXhpZi8xLjAvIgogICAgICAgICAgICB4bWxuczp0aWZmPSJodHRwOi8vbnMuYWRvYmUuY29tL3RpZmYvMS4wLyI%2BCiAgICAgICAgIDxleGlmOlBpeGVsWURpbWVuc2lvbj43MjA8L2V4aWY6UGl4ZWxZRGltZW5zaW9uPgogICAgICAgICA8ZXhpZjpQaXhlbFhEaW1lbnNpb24%2BNzIwPC9leGlmOlBpeGVsWERpbWVuc2lvbj4KICAgICAgICAgPGV4aWY6Q29sb3JTcGFjZT4xPC9leGlmOkNvbG9yU3BhY2U%2BCiAgICAgICAgIDx0aWZmOk9yaWVudGF0aW9uPjE8L3RpZmY6T3JpZW50YXRpb24%2BCiAgICAgIDwvcmRmOkRlc2NyaXB0aW9uPgogICA8L3JkZjpSREY%2BCjwveDp4bXBtZXRhPgqLa6CqAAAHN0lEQVRIDY2We2xT1x3Hz3352k5ix%2FEjLyd2kiYh7xePgCALCYMi0ReBAmOig3Zsk1pt0tC6TevEP92E0KRuZdMo1QoCIqBrCRQCIUADIW9I4iROsJ3EdpyH7di%2Btq9jX1%2Ffxy5hDS4Maeev87uPzzm%2F1%2FcciOdD4KUDBgAZemw99snxMMVcPf0ZALGXfhv3Ao2bPzdFliJU85VrYmJ0M28Mr34bABEALABPVgIAAoADgH7un6fmy6Ainz%2F44bFPsxSkhpu7schmBYiWW7drygpdXo%2FVMUd6nZnarG319QAwL3Kh%2F%2BU%2B2na%2Fb2CiXasPFK4Sz4sGGJifdoeunAsWcRuaGhozeeuty618xb7fvv8BAJH%2FB4o1X7lBI8M1jdMAJ2RAT7FhgrUH2ZCHosJhbM18qeui8Ywr%2B%2FSZZgwVgiBEgH%2BOixw9%2BvvvHgmhQB8aJoY9X6x5bTKB0wQZLwrhMTgYjDBcTMyizCyzcN4wfqGFH0ZJPkqnJKakyGUwLET5e2PFfezh6KhhYmrMbMkqteduMinESQpE7SUjhq4EmufDMZpiKSjf3u2w9l%2BWZBRFE0M8eFx2%2B7NmtVK%2BnLRn3KeLoCOPTaZp%2B66tW2Dt1EBwuPdiIcTJhonR1is40NmMRPfV9sEH9xct%2F9anJaYp9RQxLmWS2ep1hWpl6nNEgf0EyrJsZ%2F%2BjkoK81o57gwYbbSy900Z0meenDBlJxQttPWP08JtN5b82DbDzkzBryNGVodoyLE%2BfSMQc%2FoD%2FKeTZPpdtiKIj1tmFroHhcDRCzsimDMQC4R5qSTN1yRml62E75vX67C67xzuXmacImRXwg0rukW7oVKrFavzruc8BwOKJwlxIDp8gkVUUFTS92iCVJI1YLLeHW9MyJLLcEBeS%2BJ3S8tLkwW%2F72m%2BRsD5asUFpGUUfBDuDypiqPKKYrTnw%2Bu4X2%2Bxp8XPVpYVf3byzd8d2VSq2vb46P5%2BI4D6PH3LblAcPFfnrmLa%2Bu6vL16%2FLrba4zkcDFB%2FEihqhKm1dTpYegKUXdyo8YUry84UW%2BsfZS3aX2elzyjbZbeNsWgkHDxSbRhbqKjZUF1eFfKI7Pd8uZRvJcZ5zJY%2BeydDUCAEVOkpwX2hZoYP%2FO1ZKSrAxfyD4i%2BMfTSXeEMMcgsCYBMhVSFqSlnHLMUQUhgJ%2BjbXT7Ekeq%2Fr7no%2F1WVqZHNeoNOOTxgSpRJeRsVIG8VCBKzVMGA988TNSZUyWIWIRpsljIZyjaSRMwh4PbzOixbHaH2prOYWVoulcRXlD1XZcDOMiRJv2rLbiofD9%2Foery8q8XvLkzUv3pjocoVky4sMAjNNogITDgEcckiNvH%2FKqWjpG7a5JJDMLr0jV7ax9961tbwBAfec9iIdCZttMdrpGjCfE6CgEJH85eVJHdmfJ8LB36YjzkUMZFg9lltUCi9lrp6AkDGYhOlHOV4nTDzT8au%2BOPSvceOnjC%2FT65bhw8%2B5FKibo18Sug1sgGO3pH5E6O3Jw3kO4%2BjtwCmXzXo3QkwlJajZ9baj36yVH859X5RVXFhU%2FLa%2FntEDIoJBKkU6b1933YGulcmnG5Z%2B0DplMukJIq4UWJyT%2BKVFVkX6VVIUidMwhq3G%2Bc2TjH7LlVRdbby5X%2FZMYxO9UMOEL174JeF0Ihnd33QUlyR5a1ViSwosgrUq8EI5kb%2BXmr4N1urof1%2B4%2BfOfDD%2Fa895OmXbbZ6T2b33IvEtFoGMdxQQy%2FF1OO4%2Fb%2F8kimjCt6JUeKi3xk1DDhSA9YqGSKPeyYI2KDvXACoYqaUgznvglRkWSZ8nL7jd98enRzXbZEzsSCyn31hzbVrI3XU%2BSxdcbrtqVJIPGdL0cHx64iw7qNhCMmERVDCbnE0AjkmuBTq8POR6gYFv1gTd3fLvzzxNkzew%2BrEcaKTC15Ft1n29oa1m6Jd1%2F0ZXv7hG38tXxdmit6j%2BHE5ZHwK16gQecYrqMdTF5XSdVBf4yQF6a23R1IkSlPXvq66T0kGp5JNaQq1%2B8v5EXDZ0%2F98cSJeChbXVQ0a%2B5lqIifhOohzvwv5FaSZH4JxRjRO2%2Bs%2FtNHqnnCf6q7xycju3sG7FxfeaU%2BnGlM%2F0rLKgsO7Nv%2F%2Bvs%2Fd8fI3OzseCivlitsNk8WFCvgoBwcOurFjjsh%2FW6RFCA1TRYL2htN5bIZ2nyL15TjdTsTMN2E6RpW4of52kohy7%2F76cEh0%2BN339wZD4UgDKEsc5a5heoMDIc5KQDKCOSuDCjUsTZ7zB8Ebh8gI3DYleLL9N112iWd6mPqikEJ%2BqPGbULlr6%2BoXl%2BxBoDoChQKhvwiGM2oWWfu7bSkQ4vLF4YMkrt%2BDk%2FYGEMRNMaAaASlpqS5s2JNRJJP6CtxcYvVV77jcLoyZVmlBK16coVZgcIuD%2FnJ56cv9%2FTSPN43xkOccPJywkEJOaGQVQQJtxMYAgwEApwbY4IEPG2NnKdImuM%2F3uQnl0iFPHFFpf4DgiiBxpi6DYIAAAAASUVORK5CYII%3D" alt="Scaling Intelligence blog post" /></a>
  <a href="https://x.com/KumbongHermann"><img src="https://img.shields.io/badge/Thread-X-000000.svg?logo=x&logoColor=white" alt="X thread" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT license" /></a>
</p>

Next-Concept Modeling (NCM) is a two-stage framework for parallel text generation. A multiscale residual VQ tokenizer represents a sequence as a hierarchy of discrete codes, and a transformer predicts that hierarchy from coarse to fine. Positions within a level are generated in parallel, so generation requires one forward evaluation per level.

<p align="center">
  <img src="assets/overview.png" width="98%" alt="Comparison of autoregressive, diffusion, and next-concept generation" />
</p>

## Highlights

### Coarse-to-fine generation

NCM generates coarse to fine: each level predicts a longer sequence in parallel, and decoding every level shows a sample being refined from a rough sketch to the full output in a few steps.

<p align="center"><img src="assets/coarse_to_fine.gif" width="100%" alt="TinyStories, XSum and WMT14 samples refined level by level"/></p>

### Quality at a fraction of the cost

NCM beats or matches strong diffusion baselines on different tasks & datasets while generating 9-31× faster.

<p align="center">
  <img src="assets/quality_speed.png" width="98%" alt="Generation quality against wall-clock time on LM1B and XSum, and hard Sudoku accuracy by method" />
</p>

## Released models

Weights and tokenized datasets are hosted in the [NCM Hugging Face collection](https://huggingface.co/collections/hazyresearch/ncm). Each released model consists of a frozen tokenizer and a next-concept generator.

<table>
  <thead>
    <tr><th>Task</th><th>Model</th><th>Tokenizer</th><th>Generator</th><th>Config</th><th>Data</th></tr>
  </thead>
  <tbody>
    <tr><td rowspan="1"><b>XSum</b></td><td>&ndash;</td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/xsum/tokenizer.pt">tokenizer.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/xsum/generator.pt">generator.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/xsum/config.yaml">config.yaml</a></td><td rowspan="1"><a href="https://huggingface.co/datasets/hazyresearch/ncm-datasets/tree/main/xsum">test set</a></td></tr>
    <tr><td rowspan="1"><b>WMT14 De&ndash;En</b></td><td>&ndash;</td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/wmt14/tokenizer.pt">tokenizer.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/wmt14/generator.pt">generator.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/wmt14/config.yaml">config.yaml</a></td><td rowspan="1"><a href="https://huggingface.co/datasets/hazyresearch/ncm-datasets/tree/main/wmt14">test set</a></td></tr>
    <tr><td rowspan="3"><b>Sudoku</b></td><td>easy</td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/sudoku/easy/tokenizer.pt">tokenizer.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/sudoku/easy/generator.pt">generator.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/sudoku/easy/config.yaml">config.yaml</a></td><td rowspan="3"><a href="https://huggingface.co/datasets/hazyresearch/ncm-datasets/tree/main/sudoku">test boards</a></td></tr>
    <tr><td>medium</td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/sudoku/medium/tokenizer.pt">tokenizer.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/sudoku/medium/generator.pt">generator.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/sudoku/medium/config.yaml">config.yaml</a></td></tr>
    <tr><td>hard</td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/sudoku/hard/tokenizer.pt">tokenizer.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/sudoku/hard/generator.pt">generator.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/sudoku/hard/config.yaml">config.yaml</a></td></tr>
    <tr><td rowspan="3"><b>LM1B</b></td><td>small</td><td rowspan="3"><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/lm1b/tokenizer.pt">tokenizer.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/lm1b/small/generator.pt">generator.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/lm1b/small/config.yaml">config.yaml</a></td><td rowspan="3"><a href="https://huggingface.co/datasets/hazyresearch/ncm-datasets/tree/main/lm1b">tokenized text</a></td></tr>
    <tr><td>medium</td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/lm1b/medium/generator.pt">generator.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/lm1b/medium/config.yaml">config.yaml</a></td></tr>
    <tr><td>large</td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/lm1b/large/generator.pt">generator.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/lm1b/large/config.yaml">config.yaml</a></td></tr>
    <tr><td rowspan="1"><b>TinyStories</b></td><td>&ndash;</td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/tinystories/tokenizer.pt">tokenizer.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/tinystories/generator.pt">generator.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/tinystories/config.yaml">config.yaml</a></td><td rowspan="1"><a href="https://huggingface.co/datasets/hazyresearch/ncm-datasets/tree/main/tinystories">tokenized text</a></td></tr>
    <tr><td rowspan="1"><b>OpenWebText</b></td><td>&ndash;</td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/owt/tokenizer.pt">tokenizer.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/owt/generator.pt">generator.pt</a></td><td><a href="https://huggingface.co/hazyresearch/ncm-checkpoints/blob/main/owt/config.yaml">config.yaml</a></td><td rowspan="1"><a href="https://huggingface.co/datasets/hazyresearch/ncm-datasets/tree/main/owt">tokenized text</a></td></tr>
  </tbody>
</table>

## Installation

```bash
git clone https://github.com/HazyResearch/ncm.git
cd ncm
pip install -r requirements.txt
```

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended.

## Evaluation

Download the checkpoints and evaluation inputs:

```bash
hf download hazyresearch/ncm-checkpoints --local-dir artifacts
hf download hazyresearch/ncm-datasets --repo-type dataset --local-dir data
```

Then run:

```bash
python evaluate.py --config config/eval/<task>.yaml --checkpoint-dir artifacts/<task> --data-dir data/<task>
# <task> is one of xsum, wmt14, sudoku-easy, sudoku-medium or sudoku-hard

python evaluate.py --config config/eval/<dataset>.yaml --checkpoint-dir artifacts/<dataset>
# <dataset> is tinystories, owt or lm1b-<size> with <size> one of small, medium or large
```

This runs one generation shard per visible GPU, merges them, and prints ROUGE-1/2/L and generative perplexity for XSum; cased and lowercased sacreBLEU, chrF and length ratio for WMT14; solved, cell and valid-grid rates for Sudoku; and generative perplexity under GPT-2 large with unigram entropy, over 5 seeds of 256 unconditional samples, for TinyStories, OpenWebText and LM1B, the latter under the FLM scoring protocol. Add `--cfg <scale>` for classifier-free guidance on XSum and WMT14; the guided WMT14 row uses `--cfg 1.0`. Add `--kv-cache` to encode the prefix once and reuse its keys and values at every level; outputs are the same and generation is 2-5x faster.

## Repository layout

- `evaluate.py`: evaluation entry point
- `scripts/`: per-GPU generation shard and the merge that scores it
- `config/eval/`: one config per released checkpoint, identical to the `config.yaml` shipped with it
- `models/`, `tokenization/`: generator and tokenizer implementations

## Citation

```bibtex
@misc{kumbong2026nextconcept,
      title={Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation},
      author={Hermann Kumbong and Ishan S. Khare and Iddah Mlauzi and Andrew Woen and Charles Ding and E. Kelly Buchanan and Simran Arora and Azalia Mirhoseini and Christopher Ré},
      year={2026},
      eprint={2610.14162},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.14162},
}
```

Released under the [MIT license](LICENSE).
