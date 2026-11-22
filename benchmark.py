"""Benchmark the released inference path using a config from config/benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parent
SPEED_OPTIONS = ("compile", "resize_matmul", "cuda_graphs", "kv_cache")
BENCHMARK_OVERRIDES = ("batch_size", "samples", "warmup", "repeats", "seed")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="e.g. config/benchmark/<task>.yaml")
    parser.add_argument("--checkpoint-dir", help="override the checkpoint directory")
    parser.add_argument("--data-dir", help="override the conditional test data directory")
    parser.add_argument("--out", help="output directory; defaults to outputs/benchmark/<task>/<timestamp>")
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value for one GPU; otherwise use the first visible GPU")
    parser.add_argument("--precision", choices=["bf16", "fp32"])
    for flag in ("batch-size", "samples", "warmup", "repeats"):
        parser.add_argument("--" + flag, type=positive_int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true", help="validate and print config without loading weights")
    return parser


def validate_settings(config: DictConfig, conditional: bool) -> None:
    """Reject invalid timing settings before loading a model or starting a worker."""
    required = set(BENCHMARK_OVERRIDES) | {"cpu_threads"}
    settings = config.benchmark
    if set(settings) != required:
        raise ValueError(f"benchmark must contain exactly {sorted(required)}")
    for key in required - {"samples"}:
        minimum = 0 if key == "seed" else 1
        if type(settings[key]) is not int or settings[key] < minimum:
            raise ValueError(f"benchmark.{key} must be an integer >= {minimum}")
    if settings.samples is not None and (type(settings.samples) is not int or settings.samples < 1):
        raise ValueError("benchmark.samples must be positive, or null for the full conditional split")
    if not conditional and settings.samples is None:
        raise ValueError("Unconditional benchmarks require benchmark.samples")

    expected = set(SPEED_OPTIONS) | {"precision"}
    if set(config.inference) != expected:
        raise ValueError(f"inference must contain exactly {sorted(expected)}")
    if config.inference.precision not in {"bf16", "fp32"}:
        raise ValueError("inference.precision must be bf16 or fp32")
    for key in SPEED_OPTIONS:
        if type(config.inference[key]) is not bool:
            raise ValueError(f"inference.{key} must be a boolean")


def resolve_asset_path(path: str | Path) -> str:
    """Resolve checkpoint and data paths relative to the repository."""
    return str((ROOT / path).resolve())


def load_config(path: str | Path, args: argparse.Namespace) -> tuple[DictConfig, DictConfig, bool]:
    config_path = Path(path).resolve()
    config = OmegaConf.load(config_path)
    allowed = {"eval_config", "checkpoint_dir", "data_dir", "generation", "benchmark", "inference"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown benchmark config fields: {sorted(unknown)}")
    for key in ("eval_config", "checkpoint_dir", "benchmark", "inference"):
        if key not in config:
            raise ValueError(f"Missing benchmark config field: {key}")

    eval_path = (config_path.parent / config.eval_config).resolve()
    model_config = OmegaConf.load(eval_path)
    if config.get("generation"):
        model_config.generation = OmegaConf.merge(model_config.generation, config.generation)
    for key in BENCHMARK_OVERRIDES:
        override = getattr(args, key, None)
        if override is not None:
            config.benchmark[key] = override
    if args.precision:
        config.inference.precision = args.precision

    split = str(model_config.eval.get("split", "test"))
    conditional = bool(model_config.data and model_config.data.get(split))
    validate_settings(config, conditional)

    config.eval_config = str(eval_path)
    config.checkpoint_dir = resolve_asset_path(args.checkpoint_dir or config.checkpoint_dir)
    config.data_dir = resolve_asset_path(
        args.data_dir or config.get("data_dir", "data/" + str(model_config.task))
    )
    return config, model_config, conditional


def validate_assets(config: DictConfig, model_config: DictConfig, conditional: bool) -> None:
    for filename in model_config.checkpoints.values():
        path = Path(config.checkpoint_dir) / filename
        if not path.is_file():
            raise ValueError(f"Missing checkpoint asset: {path}")
    if conditional:
        split = str(model_config.eval.get("split", "test"))
        for key in (split, split + "_refs"):
            path = Path(config.data_dir) / model_config.data[key]
            if not path.is_file():
                raise ValueError(f"Missing dataset: {path}")


def prepare_output(config: DictConfig, model_config: DictConfig, output: str | None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = Path(output).resolve() if output else ROOT / "outputs/benchmark" / str(model_config.task) / stamp
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}; choose a new --out")
    OmegaConf.save(config, out / "benchmark.yaml", resolve=True)
    OmegaConf.save(model_config, out / "eval.yaml", resolve=True)
    return out


def build_worker_command(config: DictConfig, out: Path) -> list[str]:
    command = [
        sys.executable, str(ROOT / "scripts/eval_shard.py"),
        "--config", str(out / "eval.yaml"),
        "--checkpoint-dir", str(config.checkpoint_dir),
        "--data-dir", str(config.data_dir),
        "--benchmark-config", str(out / "benchmark.yaml"),
        "--bs", str(config.benchmark.batch_size),
        "--out", str(out / "samples.pkl"),
        "--precision", str(config.inference.precision),
    ]
    for key in SPEED_OPTIONS:
        prefix = "--" if config.inference[key] else "--no-"
        command.append(prefix + key.replace("_", "-"))
    return command


def worker_environment(config: DictConfig, gpu: str | None) -> dict[str, str]:
    env = dict(
        os.environ,
        NCM_ROOT=str(ROOT),
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS=str(config.benchmark.cpu_threads),
        MKL_NUM_THREADS=str(config.benchmark.cpu_threads),
    )
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def run_worker(command: list[str], env: dict[str, str], out: Path) -> None:
    """Stream worker output to the terminal and log; clean up on interruption."""
    (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (out / "run.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise SystemExit(f"Benchmark failed (exit {code}); see {out / 'run.log'}")


def print_summary(out: Path) -> None:
    result = json.loads((out / "results.json").read_text())
    print(
        f"{result['samples']} samples, batch {result['batch_size']}: "
        f"{result['median_seconds']:.3f} s, {result['samples_per_second']:.2f} samples/s, "
        f"{result['amortized_batch_ms']:.2f} ms/batch"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        config, model_config, conditional = load_config(args.config, args)
        if args.dry_run:
            print(OmegaConf.to_yaml(config, resolve=True))
            return
        if args.gpu is not None and (not args.gpu.strip() or "," in args.gpu):
            raise ValueError("--gpu must select one GPU; run separate benchmarks on separate GPUs")
        validate_assets(config, model_config, conditional)
        out = prepare_output(config, model_config, args.out)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    print(f"Benchmarking {model_config.task} -> {out}", flush=True)
    run_worker(build_worker_command(config, out), worker_environment(config, args.gpu), out)
    print_summary(out)


if __name__ == "__main__":
    main()
