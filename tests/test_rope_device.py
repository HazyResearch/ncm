import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.common import precompute_rope_cos_sin  # noqa: E402


def test_precompute_rope_cos_sin_cpu_values_and_shapes():
    positions = torch.tensor([0, 2, 5], dtype=torch.long)
    cos, sin = precompute_rope_cos_sin(positions, head_dim=4, base=10000)

    assert cos.shape == (3, 4)
    assert sin.shape == (3, 4)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32
    assert cos.device == positions.device
    assert sin.device == positions.device
    assert torch.isfinite(cos).all()
    assert torch.isfinite(sin).all()

    torch.testing.assert_close(cos[0], torch.ones(4, dtype=torch.float32))
    torch.testing.assert_close(sin[0], torch.zeros(4, dtype=torch.float32))
    torch.testing.assert_close(cos.pow(2) + sin.pow(2), torch.ones_like(cos))
    torch.testing.assert_close(cos[:, :2], cos[:, 2:])
    torch.testing.assert_close(sin[:, :2], sin[:, 2:])

    expected_cos_row1 = torch.tensor(
        [math.cos(2.0), math.cos(0.02), math.cos(2.0), math.cos(0.02)],
        dtype=torch.float32,
    )
    expected_sin_row1 = torch.tensor(
        [math.sin(2.0), math.sin(0.02), math.sin(2.0), math.sin(0.02)],
        dtype=torch.float32,
    )
    torch.testing.assert_close(cos[1], expected_cos_row1)
    torch.testing.assert_close(sin[1], expected_sin_row1)


@pytest.mark.parametrize("seq_len, head_dim, base", [(4, 8, 10000), (7, 64, 500000)])
def test_precompute_rope_cos_sin_meta_device(seq_len, head_dim, base):
    positions = torch.empty(seq_len, dtype=torch.long, device="meta")
    cos, sin = precompute_rope_cos_sin(positions, head_dim=head_dim, base=base)

    assert cos.device == torch.device("meta")
    assert sin.device == torch.device("meta")
    assert cos.shape == (seq_len, head_dim)
    assert sin.shape == (seq_len, head_dim)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32


@pytest.mark.parametrize("head_dim", [2, 4, 8])
def test_precompute_rope_cos_sin_accepts_even_head_dim(head_dim):
    positions = torch.arange(4, dtype=torch.long)
    cos, sin = precompute_rope_cos_sin(positions, head_dim)
    assert cos.shape == (4, head_dim)
    assert sin.shape == (4, head_dim)


@pytest.mark.parametrize("head_dim", [1, 3, 7])
def test_precompute_rope_cos_sin_rejects_odd_head_dim(head_dim):
    positions = torch.arange(4, dtype=torch.long)
    with pytest.raises(AssertionError, match="head_dim must be even"):
        precompute_rope_cos_sin(positions, head_dim)
