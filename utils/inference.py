"""Shared inference flags for the evaluation entry point and its workers."""
import argparse
import math


def positive_temperature(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("soft temperature must be finite and greater than zero")
    return value


def add_soft_decision_args(parser):
    parser.add_argument("--soft-decisions", action=argparse.BooleanOptionalAction, default=None,
                        help="override generation.soft_decisions (otherwise use the config)")
    parser.add_argument("--soft-temperature", type=positive_temperature, default=None,
                        help="soft code-mixture temperature, independent of --temp (config value or 1.0)")


def add_speed_args(parser):
    for flag, help_text in [
        ('compile', 'compile generator blocks, codec decoder and token picker'),
        ('resize-matmul', 'use fixed matrices for parameter-free resizing'),
        ('cuda-graphs', 'capture and replay inference for each input batch shape'),
        ('kv-cache', 'reuse the request-local prefix KV cache'),
    ]:
        parser.add_argument('--' + flag, action=argparse.BooleanOptionalAction, default=True,
                            help=help_text + ' (default: on)')
    parser.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16',
                        help='inference autocast precision (default: bf16; use fp32 to disable autocast)')


def speed_cli_flags(args):
    flags = ['--' + ('' if getattr(args, name.replace('-', '_')) else 'no-') + name
             for name in ['compile', 'resize-matmul', 'cuda-graphs', 'kv-cache']]
    return flags + ['--precision', args.precision]
