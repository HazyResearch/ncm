"""Python orchestration for fused token selection."""

import torch
import triton

from kernels.penalized_argmax import _penalize_tiles, _select_token


class FusedTokenPicker:
    """Two kernels per position; preserve FP64 penalty arithmetic and token order.

    Coefficients are prepared during warmup, so steady-state CUDA graph replay
    needs no CPU transfers. Unsupported inputs retain the original picker.
    """

    def __init__(self, fallback, frequency, deficit, adjacency, allowance):
        self.fallback = fallback
        self.frequency = frequency
        self.deficit = deficit
        self.allowance = allowance
        self.strength = adjacency[0] if adjacency else 0.
        self.window = int(adjacency[1]) if adjacency and len(adjacency) > 1 else 1
        self.decay = adjacency[2] if adjacency and len(adjacency) > 2 else 1.
        if not adjacency:
            self.window = 0
        # A nonpositive configured window has special Python slicing semantics.
        self.supported_window = not adjacency or 0 < self.window <= 16
        self.coefficients = {}

    def __call__(self, logits):
        if (logits.device.type != 'cuda' or not self.supported_window
                or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)):
            return self.fallback(logits)
        batch, length, vocab = logits.shape
        allowance = self.allowance
        if allowance is not None and (
            allowance.device != logits.device or allowance.dtype != torch.float64
            or allowance.ndim != 1 or not allowance.is_contiguous() or allowance.numel() < vocab
        ):
            return self.fallback(logits)
        if batch == 0 or length == 0 or vocab == 0:
            return self.fallback(logits)

        key = (length, logits.device)
        if key not in self.coefficients:
            # Compute the same Python-double constants as the original loop.
            deficit = [self.deficit * ((length - t) / length) for t in range(length)]
            adjacency = [-self.strength * self.decay ** j for j in range(self.window)]
            self.coefficients[key] = (
                torch.tensor(deficit, dtype=torch.float64, device=logits.device),
                torch.tensor(adjacency, dtype=torch.float64, device=logits.device),
            )
        deficit, adjacency = self.coefficients[key]
        tokens = torch.empty((batch, length), dtype=torch.long, device=logits.device)
        counts = torch.zeros((batch, vocab), dtype=torch.int32, device=logits.device)
        tile = 512
        tiles = triton.cdiv(vocab, tile)
        values = torch.empty((batch, tiles), dtype=torch.float64, device=logits.device)
        indices = torch.empty((batch, tiles), dtype=torch.int32, device=logits.device)

        for position in range(length):
            _penalize_tiles[(batch, tiles)](
                logits, allowance, deficit, adjacency, counts, tokens, values, indices,
                position, length, vocab, *logits.stride(), self.frequency,
                allowance is not None, allowance is not None and self.deficit > 0,
                self.window, tile, tiles, num_warps=4, enable_fp_fusion=False,
            )
            _select_token[(batch,)](
                values, indices, counts, tokens, position, length, vocab,
                tiles, triton.next_power_of_2(tiles), num_warps=4,
            )
        return tokens
