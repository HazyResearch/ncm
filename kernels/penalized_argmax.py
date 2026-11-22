"""Triton kernels for frequency- and adjacency-penalized argmax."""

import triton
import triton.language as tl


@triton.jit
def _argmax_pair(a, ai, b, bi):
    # Match torch.argmax: first index wins ties, including multiple NaNs.
    a_nan, b_nan = a != a, b != b
    left = (a > b) | ((a == b) & (ai < bi))
    left = left | (a_nan & (~b_nan | (ai < bi)))
    return tl.where(left, a, b), tl.where(left, ai, bi)


@triton.jit(do_not_specialize=['position'])
def _penalize_tiles(
    Logits, Allowance, Deficit, Adjacency, Counts, Tokens, Values, Indices,
    position, LENGTH: tl.constexpr, VOCAB: tl.constexpr,
    BATCH_STRIDE: tl.constexpr, TIME_STRIDE: tl.constexpr, VOCAB_STRIDE: tl.constexpr,
    FREQUENCY: tl.constexpr, HAS_ALLOWANCE: tl.constexpr,
    HAS_DEFICIT: tl.constexpr, WINDOW: tl.constexpr,
    TILE: tl.constexpr, TILES: tl.constexpr,
):
    batch, tile = tl.program_id(0), tl.program_id(1)
    ids = tile * TILE + tl.arange(0, TILE)
    scores = tl.load(
        Logits + batch * BATCH_STRIDE + position * TIME_STRIDE + ids * VOCAB_STRIDE,
        ids < VOCAB, float('-inf'),
    ).to(tl.float64)

    if FREQUENCY > 0:
        counts = tl.load(Counts + batch * VOCAB + ids, ids < VOCAB, 0).to(tl.float64)
        if HAS_ALLOWANCE:
            allowance = tl.load(Allowance + ids, ids < VOCAB, 0) * LENGTH
            excess = tl.maximum(counts - allowance, 0.)
        else:
            excess = counts
        scores = scores - tl.full((), FREQUENCY, tl.float64) * excess
        if HAS_DEFICIT:
            scores = scores + tl.load(Deficit + position) * tl.maximum(allowance - counts, 0.)

    for distance in tl.static_range(WINDOW):
        previous = tl.load(Tokens + batch * LENGTH + position - distance - 1,
                           position > distance, -1)
        scores = scores + tl.where(ids == previous, tl.load(Adjacency + distance), 0.)

    value, token = tl.reduce((scores, ids), 0, _argmax_pair)
    tl.store(Values + batch * TILES + tile, value)
    tl.store(Indices + batch * TILES + tile, token)


@triton.jit(do_not_specialize=['position'])
def _select_token(Values, Indices, Counts, Tokens, position,
                  LENGTH: tl.constexpr, VOCAB: tl.constexpr,
                  TILES: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    tiles = tl.arange(0, BLOCK)
    values = tl.load(Values + batch * TILES + tiles, tiles < TILES, float('-inf'))
    indices = tl.load(Indices + batch * TILES + tiles, tiles < TILES, 2147483647)
    _, token = tl.reduce((values, indices), 0, _argmax_pair)
    tl.store(Tokens + batch * LENGTH + position, token)
    count = tl.load(Counts + batch * VOCAB + token)
    tl.store(Counts + batch * VOCAB + token, count + 1)
