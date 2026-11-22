"""Inference-only resize matrices and fixed-shape CUDA graph replay."""
import torch
import torch.nn.functional as F


def fuse_token_picker(picker, frequency, deficit, adjacency, allowance):
    """Use fused CUDA penalties when available; keep the original fallback."""
    if (frequency <= 0 and not adjacency) or not torch.cuda.is_available() or torch.version.hip:
        return picker
    try:
        from utils.token_selection import FusedTokenPicker
    except ImportError:
        return picker
    return FusedTokenPicker(picker, frequency, deficit, adjacency, allowance)


def enable_resize_matmuls(vq):
    if vq.training:
        raise ValueError("Resize matrices are for an eval-mode codec")
    if getattr(vq, '_fast_resize_enabled', False):
        return
    device = vq.embeddings[0].weight.device
    length = int(vq.levels[-1])
    last = len(vq.levels) - 1
    original_up = vq._up_to_full
    original_next = vq.get_next_autoregressive_input

    def matrix(nin, nout, mode):
        # CPU construction preserves the benchmark's interpolation coefficients.
        basis = torch.eye(nin, dtype=torch.float32)[None]
        return F.interpolate(basis, size=nout, mode=mode)[0].T.contiguous().to(device)

    up = {i: matrix(int(n), length, 'linear')
          for i, n in enumerate(vq.levels[:-1]) if i > 0}
    down = {i: matrix(length, int(n), 'area')
            for i, n in enumerate(vq.levels) if i > 0}

    def up_to_full(q, level, target_length):
        # Preserve the learned first-level upsampler and its normalization.
        if level == 0 or level not in up or target_length != length:
            return original_up(q, level, target_length)
        with torch.autocast(device_type=q.device.type, enabled=False):
            resized = up[level].to(q.dtype) @ q
        if vq.learned_all_sampling:
            # Replace only interpolation, preserving the learned refinement.
            channels = resized.transpose(-1, -2)
            resized = (channels + vq.learned_up_refine[level](channels)).transpose(-1, -2)
        return resized.contiguous()

    def next_input(level, z, q):
        if level == last:
            return original_next(level, z, q)
        z.add_(vq._apply_quant_resi(vq._up_to_full(q, level, length), level))
        # Preserve the interpolation input dtype even under BF16 autocast.
        with torch.autocast(device_type=z.device.type, enabled=False):
            return z, down[level + 1].to(z.dtype) @ z

    vq._up_to_full = up_to_full
    vq.get_next_autoregressive_input = next_input
    vq._fast_resize_enabled = True


class CUDAGraphRunner:
    """Sequential inference; consume returned tokens before the next replay."""

    def __init__(self, gpu_generate):
        self.gpu_generate = gpu_generate
        self.graphs = {}
        self.replay_count = 0

    @torch.inference_mode()
    def __call__(self, x, l0=None):
        if x.device.type != 'cuda':
            raise ValueError("CUDA graph inference requires CUDA inputs")
        key = (x.device, tuple(x.shape), x.dtype,
               None if l0 is None else (tuple(l0.shape), l0.dtype))
        if key not in self.graphs:
            static_x = x.clone()
            static_l0 = None if l0 is None else l0.clone()
            # Warmup/capture must not consume the caller's sampling RNG stream.
            with torch.cuda.device(x.device), torch.random.fork_rng(devices=[x.device.index]):
                for _ in range(25):
                    self.gpu_generate(static_x, static_l0)
                stream = torch.cuda.Stream(device=x.device)
                stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.gpu_generate(static_x, static_l0)
                torch.cuda.current_stream(x.device).wait_stream(stream)
                torch.cuda.synchronize(x.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    static_tokens = self.gpu_generate(static_x, static_l0)
            self.graphs[key] = graph, static_x, static_l0, static_tokens
        graph, static_x, static_l0, static_tokens = self.graphs[key]
        static_x.copy_(x)
        if static_l0 is not None:
            static_l0.copy_(l0)
        graph.replay()
        self.replay_count += 1
        # The prefix KV cache is recomputed inside each replay for the new input.
        return static_tokens
