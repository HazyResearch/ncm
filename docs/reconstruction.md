# Reconstruction contracts

The neural tokenizer is lossy. Transport fixes do not guarantee preservation of
source bytes, syntax, identifiers, or program behaviour after quantization.

## Intermediate outputs

`vq.z_to_idx(z, return_zhats=True)` and `vq.idx_to_zhats(indices)[0]` return
independent cumulative snapshots of shape `[batch, final_length, embed_dim]`.
The second output of `idx_to_zhats` contains the same cumulative representations
resampled to each level's native length. It does not contain residual codebook
vectors. Intermediate reconstruction accuracy need not increase monotonically.

`VQVAE.reconstruct_multiscales` and the corresponding `VQVAETokenizer` methods
retain native-resolution decoding by default. To compare the same positions
throughout refinement, explicitly request final-resolution decoding:

```python
model.eval()
reconstructions = model.reconstruct_multiscales(token_ids, full_resolution=True)
# reconstructions[example][level][position]

drafts = tokenizer.decode_multiscales(concept_ids, full_resolution=True)
# drafts[example][level] is text decoded at the final sequence length.
```

## Text blocks and UTF-8

`VQVAETokenizer.encode` concatenates complete concept hierarchies for text blocks.
`decode` consumes every complete hierarchy and joins the reconstructed text token
ids before detokenization, including when a UTF-8 character spans model blocks.
An incomplete hierarchy raises `ValueError` instead of silently ignoring codes.

The flat concept stream does not store the original text length. Consequently,
`decode` can include padding; use the pre-tokenizer's special-token handling where
appropriate. The `reconstruct` helpers know the original token length and remove
only the extra positions introduced by block padding. These helpers still return
the neural reconstruction, not a lossless copy of their input.

`Tokenizer.encode_chunk(path, start, end)` partitions a UTF-8 file by byte offsets.
Both boundaries round forward to character boundaries, assigning a split
character to the preceding partition. Adjacent partitions preserve characters
exactly once. Invalid UTF-8 raises an error rather than silently dropping bytes.
BPE ids can differ from encoding the entire file because merges do not cross
partition boundaries.

## Regression tests

```bash
python -m unittest discover -s tests -v
```

The tests use small seeded models without network access or downloaded weights.
They verify independent snapshots, final-output parity, native and final lengths,
parallel/incremental conditioning equivalence, block ordering, and UTF-8 transport.
The device-routing test also runs on CUDA when available.
