"""Configuration and sample selection for the generation-only CLI."""

from pathlib import Path
import math
import pickle
import random

from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_generation_config(path, args):
    path = Path(path).resolve()
    config = OmegaConf.load(path)
    allowed = {'model_config', 'checkpoint_dir', 'data_dir', 'split', 'samples',
               'batch_size', 'seed', 'generation'}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f'Unknown generation config fields: {sorted(unknown)}')
    model_config = OmegaConf.load(path.parent / config.model_config)
    model_config.generation = OmegaConf.merge(model_config.generation, config.get('generation', {}))
    for name in ('samples', 'batch_size', 'seed'):
        value = getattr(args, name, None)
        if value is not None:
            config[name] = value
        minimum = 0 if name == 'seed' else 1
        if type(config.get(name)) is not int or config[name] < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    for name in ('checkpoint_dir', 'data_dir'):
        value = getattr(args, name, None) or config.get(name)
        if value:
            config[name] = str((ROOT / value).resolve())
    for name, key in [('temperature', 'temperature'), ('top_k', 'top_k'),
                      ('top_p', 'top_p'), ('cfg', 'cfg_scale')]:
        value = getattr(args, name, None)
        if value is not None:
            model_config.generation[key] = value
    settings = model_config.generation
    finite = all(math.isfinite(float(settings[key])) for key in ('temperature', 'top_p', 'cfg_scale'))
    if not finite or settings.temperature < 0 or settings.top_k < 0 or not 0 <= settings.top_p <= 1 or settings.cfg_scale < 0:
        raise ValueError('temperature, top-k and cfg must be nonnegative; top-p must be in [0, 1]')
    for filename in model_config.checkpoints.values():
        target = Path(config.checkpoint_dir) / filename
        if not target.is_file():
            raise ValueError(f'Missing checkpoint: {target}')
    return config, model_config


def load_inputs(config, model_config):
    split = config.get('split', 'test')
    if not model_config.data or not model_config.data.get(split):
        if model_config.data:
            raise ValueError(f'No data configured for split {split!r}')
        return None
    with (Path(config.data_dir) / model_config.data[split]).open('rb') as stream:
        items = pickle.load(stream)
    with (Path(config.data_dir) / model_config.data[split + '_refs']).open('rb') as stream:
        references = pickle.load(stream)
    if len(items) != len(references):
        raise ValueError('Input and reference counts differ')
    if config.samples > len(items):
        raise ValueError(f'Requested {config.samples} samples from {len(items)} inputs')
    indices = random.Random(config.seed).sample(range(len(items)), config.samples)
    return [(index, items[index], references[index]) for index in indices]


def generate_samples(runtime, config, inputs):
    """Yield unedited samples and, optionally, their generated hierarchy prefixes."""
    torch.manual_seed(config.seed)
    pre_tokenizer = runtime.tokenizer.pre_tokenizer
    prefix_length = int(runtime.config.model.prefix_len)
    initial_length = int(runtime.tokenizer.vqvae.vq.levels[0])
    probabilities = runtime.initial_probabilities
    probabilities = probabilities.cpu() if probabilities is not None else None
    for start in range(0, config.samples, config.batch_size):
        size = min(config.batch_size, config.samples - start)
        records, prefixes = [], []
        for offset in range(size):
            record = dict(task=str(runtime.config.task), seed=int(config.seed),
                          batch_size=int(config.batch_size), precision=runtime.options.precision,
                          generation=OmegaConf.to_container(runtime.config.generation, resolve=True))
            if inputs is not None:
                index, item, reference = inputs[start + offset]
                ids = pre_tokenizer.encode(item) if isinstance(item, str) else list(item)
                prefix = ids[:prefix_length]
                record.update(dataset_index=index, prompt_ids=prefix,
                              prompt=runtime.decode(prefix, cut_at_eos=False),
                              prompt_truncated=len(ids) > prefix_length)
                prefix = prefix + [runtime.pad] * (prefix_length - len(prefix))
                reference_ids = pre_tokenizer.encode(reference) if isinstance(reference, str) else list(reference)
                record.update(reference=reference if isinstance(reference, str) else runtime.decode(reference_ids),
                              reference_ids=reference_ids)
            else:
                prefix = [runtime.eos]
            prefixes.append(prefix)
            records.append(record)
        initial = None
        if inputs is None and probabilities is not None:
            initial = torch.multinomial(probabilities, size * initial_length, replacement=True).view(size, initial_length).cuda()
        prefix = torch.tensor(prefixes, dtype=torch.long, device='cuda')
        generated = runtime.generate_tokens(prefix, initial)
        if runtime.trajectory:
            tokens, latents = generated
            for record in records:
                record['trajectory'] = []
            for level, length, rows in runtime.decode_trajectory(latents, tokens):
                for record, row in zip(records, rows):
                    record['trajectory'].append(dict(level=level, codes=length,
                                                     text=runtime.decode(row), raw_ids=row))
        else:
            tokens = generated
        for record, row in zip(records, tokens.tolist()):
            record.update(output=runtime.decode(row), raw_ids=row)
            yield record
