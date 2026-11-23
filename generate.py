"""Generate readable examples without running evaluation metrics."""

import argparse
from contextlib import redirect_stdout, nullcontext
import io
import json
from pathlib import Path
import shutil
import sys

from utils.inference import add_speed_args


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='configs/generation/<task>.yaml')
    parser.add_argument('--checkpoint-dir', help='override the checkpoint directory')
    parser.add_argument('--data-dir', help='override the conditional input directory')
    parser.add_argument('--samples', type=int, help='number of examples')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--trajectory', action='store_true', help='also show decoded hierarchy prefixes (default: off)')
    parser.add_argument('--out', type=Path, help='also save samples, token IDs and settings as JSONL')
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--top-p', type=float)
    parser.add_argument('--cfg', type=float, help='classifier-free guidance scale')
    add_speed_args(parser)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    import torch
    from utils.generation import Generator
    from utils.sample_display import format_sample
    from utils.sample_generation import load_generation_config, load_inputs, generate_samples

    try:
        config, model_config = load_generation_config(args.config, args)
        inputs = load_inputs(config, model_config)
        if not torch.cuda.is_available():
            raise ValueError('Generation requires a CUDA GPU')
        if args.out and args.out.exists():
            raise ValueError(f'Output already exists: {args.out}')
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    torch.set_num_threads(4)
    print(f'Loading {model_config.task}; preparing {config.samples} example(s)…', file=sys.stderr, flush=True)
    messages = io.StringIO()
    try:
        with redirect_stdout(messages):
            runtime = Generator(model_config, config.checkpoint_dir, args, trajectory=args.trajectory)
    except Exception:
        print(messages.getvalue(), file=sys.stderr)
        raise
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    width = max(40, min(100, shutil.get_terminal_size((96, 24)).columns))
    with (args.out.open('x') if args.out else nullcontext()) as output:
        for number, sample in enumerate(generate_samples(runtime, config, inputs), 1):
            print(format_sample(sample, number, width), end='\n\n', flush=True)
            if output:
                output.write(json.dumps(sample, ensure_ascii=False) + '\n')
                output.flush()


if __name__ == '__main__':
    main()
