"""Shared model loading, sampling and decoding for generation and evaluation."""

from pathlib import Path

import torch
from omegaconf import OmegaConf

from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from utils.fast_generation import CUDAGraphRunner, enable_resize_matmuls, fuse_token_picker
from utils.inference import positive_temperature


def make_token_picker(frequency, deficit, adjacency, allowance):
    """Build the reference decoder, also used as the fused kernel's fallback."""
    def pick(logits):
        if frequency <= 0 and not adjacency:
            return logits.argmax(-1)
        batch, length, vocab = logits.shape
        logits = logits.double()
        allow = (allowance[:vocab] * length)[None, :] if allowance is not None else None
        strength = adjacency[0] if adjacency else 0.0
        window = int(adjacency[1]) if adjacency and len(adjacency) > 1 else 1
        decay = adjacency[2] if adjacency and len(adjacency) > 2 else 1.0
        counts = torch.zeros(batch, vocab, dtype=torch.float64, device=logits.device)
        output = torch.empty(batch, length, dtype=torch.long, device=logits.device)
        history = []
        for position in range(length):
            step = logits[:, position, :].clone()
            if frequency > 0:
                excess = (counts - allow).clamp_min(0.0) if allow is not None else counts
                step -= frequency * excess
                if deficit > 0 and allow is not None:
                    step += deficit * ((length - position) / length) * (allow - counts).clamp_min(0.0)
            for distance, token in enumerate(reversed(history[-window:])):
                step.scatter_add_(1, token[:, None], torch.full(
                    (batch, 1), -strength * decay ** distance,
                    dtype=torch.float64, device=logits.device))
            token = step.argmax(-1)
            output[:, position] = token
            history.append(token)
            counts.scatter_add_(1, token[:, None], torch.ones(
                batch, 1, dtype=torch.float64, device=logits.device))
        return output
    return pick


class Generator:
    """One frozen codec and generator; no scoring models or dataset dependencies."""

    def __init__(self, config, checkpoint_dir, options, *, trajectory=False):
        self.config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        self.options = options
        self.trajectory = trajectory
        self.checkpoint_dir = Path(checkpoint_dir)
        config = self.config
        config.tokenizer.vqvae.checkpoint_path = str(self.checkpoint_dir / config.checkpoints.tokenizer)
        for key, name in (('vocab_file_path', 'bpe_vocab'), ('merges_file_path', 'bpe_merges')):
            if config.checkpoints.get(name):
                config.tokenizer.pre_tokenizer[key] = str(self.checkpoint_dir / config.checkpoints[name])
        self.tokenizer = VQVAETokenizer.from_config(config.tokenizer)
        codec = self.tokenizer.vqvae.cuda().eval()
        self.model = NextConceptTransformer(vqvae_local=codec, config=config.model).cuda().eval()
        checkpoint = torch.load(self.checkpoint_dir / config.checkpoints.generator,
                                map_location='cpu', weights_only=True)
        self.model.load_state_dict(checkpoint['model'], strict=True)

        def tensor(spec):
            path, _, reference = str(spec).partition(':')
            value = checkpoint if path == config.checkpoints.generator else torch.load(
                self.checkpoint_dir / path, map_location='cpu', weights_only=True)
            for key in reference.split('.') if reference else []:
                value = value[key]
            return value.cuda().double()

        settings = config.generation
        self.allowance = tensor(settings.frequency_allowance) if settings.get('frequency_allowance') else None
        self.initial_probabilities = tensor(settings.level0_unigram).float() if settings.get('level0_unigram') else None
        self.eos = self.tokenizer.pre_tokenizer.eos_token_id
        self.pad = self.tokenizer.pre_tokenizer.pad_token_id
        # Existing evaluation configs retain their FLM compatibility default.
        self.cut_at_eos = bool(settings.get('cut_at_eos',
            OmegaConf.select(config, 'eval.metric') != 'genppl_flm'))
        self.sampling = dict(
            temperature=float(settings.get('temperature', 1.0)),
            top_k=int(settings.get('top_k', 1)), top_p=float(settings.get('top_p', 0.0)),
            cfg_scale=float(settings.get('cfg_scale', 0.0)), cfg_null_eos=self.eos,
            soft_decisions=bool(settings.get('soft_decisions', False)),
            soft_temperature=positive_temperature(settings.get('soft_temperature', 1.0)),
            use_kv_cache=options.kv_cache, return_latent=True, return_trajectory=trajectory,
        )
        frequency = float(settings.get('frequency_penalty', 0.0))
        deficit = float(settings.get('frequency_deficit', 0.0))
        adjacency = [float(v) for v in str(settings.adjacency_penalty).split(',')] if settings.get('adjacency_penalty') else None
        picker = make_token_picker(frequency, deficit, adjacency, self.allowance)
        self.decoder = codec.decoder
        if options.resize_matmul:
            enable_resize_matmuls(codec.vq)
        if options.compile:
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 128)
            torch._dynamo.config.accumulated_cache_size_limit = max(
                torch._dynamo.config.accumulated_cache_size_limit, 512)
            self.model.transformer.blocks = torch.nn.ModuleList([
                torch.compile(block, fullgraph=True, dynamic=True)
                for block in self.model.transformer.blocks
            ])
            self.decoder = torch.compile(self.decoder, fullgraph=True, dynamic=False)
            picker = torch.compile(picker, fullgraph=True, dynamic=False)
        self.pick_tokens = fuse_token_picker(picker, frequency, deficit, adjacency, self.allowance)
        self.generate_tokens = CUDAGraphRunner(self._generate) if options.cuda_graphs else self._generate

    def autocast(self):
        return torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.options.precision == 'bf16')

    @torch.inference_mode()
    def _generate(self, prefix, initial_codes=None):
        with self.autocast():
            _, latent = self.model.generate(initial_codes, prefix, **self.sampling)
            final_latent = latent[:, -1] if self.trajectory else latent
            tokens = self.pick_tokens(self.decoder(final_latent))
        return (tokens, latent) if self.trajectory else tokens

    def decode(self, ids, *, cut_at_eos=None):
        ids = list(ids)
        cut = self.cut_at_eos if cut_at_eos is None else cut_at_eos
        if cut and self.eos in ids:
            ids = ids[:ids.index(self.eos)]
        return self.tokenizer.pre_tokenizer.decode(
            [token for token in ids if token not in (self.eos, self.pad)], skip_special_tokens=True)

    @torch.inference_mode()
    def decode_trajectory(self, latents, final_tokens):
        """Decode the actual hard or soft latent saved after each level."""
        codec = self.tokenizer.vqvae
        with self.autocast():
            for level, length in enumerate(codec.vq.levels):
                if level == len(codec.vq.levels) - 1:
                    # Reuse the captured output to avoid BF16 differences from a second decode.
                    yield level + 1, int(length), final_tokens.tolist()
                    break
                ids = self.pick_tokens(self.decoder(latents[:, level])).tolist()
                yield level + 1, int(length), ids
