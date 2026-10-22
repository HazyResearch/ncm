# Evaluation integrity

Run the offline regression suite with:

```bash
python -m unittest discover -s tests -v
```

`evaluate.py` merges an explicit list of files written by its current workers and
checks their run identifier. Stale files matching `shard_*.pkl` in the output
directory are excluded. The merger validates field lengths, sample coverage,
duplicate conditional indices, and configuration/policy consistency. Unconditional
shards use local indices and disjoint seed groups, which are checked separately.
Legacy `--glob` input is supported, but old shards lack provenance metadata.
Checkpoint paths in metadata do not prove immutable checkpoint contents.

Sudoku cell accuracy uses 81 cells per reference, including missing predicted
positions as errors. Malformed output is reported separately. Exact string match
and valid-grid rates keep their existing definitions.

`--score-bs` sets reference-model batch size independently from generation `--bs`.
Omitting it preserves the previous task-specific batch sizes. Scoring no longer
requests unused KV caches. For left-padded text it excludes pad-to-first-token
transitions and uses positions counted from the first real token. This makes
scores consistent with scoring each example without padding.

An example with no scorable tokens has NaN per-example perplexity and is counted
as unscored. An entirely unscorable corpus raises `ValueError`. Corpus perplexity
retains token weighting; the FLM EOS convention is unchanged.
