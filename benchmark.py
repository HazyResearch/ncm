"""Benchmark the released inference path using a config from config/benchmark."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def load_config(path, args):
    path = Path(path).resolve()
    cfg = OmegaConf.load(path)
    allowed = {"eval_config", "checkpoint_dir", "data_dir", "generation", "benchmark", "inference"}
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Unknown benchmark config fields: {sorted(unknown)}")
    for key in ["eval_config", "checkpoint_dir", "benchmark", "inference"]:
        if key not in cfg:
            raise ValueError(f"Missing benchmark config field: {key}")
    eval_path = (path.parent / cfg.eval_config).resolve()
    model = OmegaConf.load(eval_path)
    if cfg.get("generation"):
        model.generation = OmegaConf.merge(model.generation, cfg.generation)
    for key in ["batch_size", "samples", "warmup", "repeats", "seed"]:
        override = getattr(args, key, None)
        if override is not None:
            cfg.benchmark[key] = override
    if args.precision:
        cfg.inference.precision = args.precision
    required = {"batch_size", "samples", "warmup", "repeats", "seed", "cpu_threads"}
    if set(cfg.benchmark) != required:
        raise ValueError(f"benchmark must contain exactly {sorted(required)}")
    for key in required - {"samples"}:
        value = cfg.benchmark[key]
        minimum = 0 if key == "seed" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"benchmark.{key} must be an integer >= {minimum}")
    samples = cfg.benchmark.samples
    if samples is not None and (type(samples) is not int or samples < 1):
        raise ValueError("benchmark.samples must be positive, or null for the full conditional split")
    split = str(model.eval.get("split", "test"))
    conditional = bool(model.data and model.data.get(split))
    if not conditional and samples is None:
        raise ValueError("Unconditional benchmarks require benchmark.samples")
    expected = {"precision", "compile", "resize_matmul", "cuda_graphs", "kv_cache"}
    if set(cfg.inference) != expected:
        raise ValueError(f"inference must contain exactly {sorted(expected)}")
    if cfg.inference.precision not in {"bf16", "fp32"}:
        raise ValueError("inference.precision must be bf16 or fp32")
    for key in expected - {"precision"}:
        if type(cfg.inference[key]) is not bool:
            raise ValueError(f"inference.{key} must be a boolean")
    checkpoint_dir = Path(args.checkpoint_dir or cfg.checkpoint_dir)
    data_dir = Path(args.data_dir or cfg.get("data_dir", "data/" + str(model.task)))
    # Asset paths in configs are relative to the repository, not the caller's cwd.
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = ROOT / checkpoint_dir
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    cfg.eval_config = str(eval_path)
    cfg.checkpoint_dir = str(checkpoint_dir.resolve())
    cfg.data_dir = str(data_dir.resolve())
    return cfg, model, conditional


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="e.g. config/benchmark/xsum.yaml")
    parser.add_argument("--checkpoint-dir", help="override the checkpoint directory")
    parser.add_argument("--data-dir", help="override the conditional test data directory")
    parser.add_argument("--out", help="output directory; defaults to outputs/benchmark/<task>/<timestamp>")
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES value for one GPU; otherwise use the first visible GPU")
    parser.add_argument("--precision", choices=["bf16", "fp32"])
    for flag in ["batch-size", "samples", "warmup", "repeats"]:
        parser.add_argument("--" + flag, type=positive_int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true", help="validate and print config without loading weights")
    args = parser.parse_args()
    try:
        cfg, model, conditional = load_config(args.config, args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    if args.gpu is not None and (not args.gpu.strip() or "," in args.gpu):
        parser.error("--gpu must select one GPU; run separate benchmarks on separate GPUs")
    checkpoint_dir, data_dir = Path(cfg.checkpoint_dir), Path(cfg.data_dir)
    for filename in model.checkpoints.values():
        if not (checkpoint_dir / filename).is_file():
            parser.error(f"Missing checkpoint asset: {checkpoint_dir / filename}")
    if conditional:
        split = str(model.eval.get("split", "test"))
        for key in [split, split + "_refs"]:
            if not (data_dir / model.data[key]).is_file():
                parser.error(f"Missing dataset: {data_dir / model.data[key]}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = Path(args.out).resolve() if args.out else ROOT / "outputs/benchmark" / str(model.task) / stamp
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error(f"Output directory is not empty: {out}; choose a new --out")
    OmegaConf.save(cfg, out / "benchmark.yaml", resolve=True)
    OmegaConf.save(model, out / "eval.yaml", resolve=True)
    command = [sys.executable, str(ROOT / "scripts/eval_shard.py"),
               "--config", str(out / "eval.yaml"), "--checkpoint-dir", str(checkpoint_dir),
               "--data-dir", str(data_dir), "--benchmark-config", str(out / "benchmark.yaml"),
               "--bs", str(cfg.benchmark.batch_size), "--out", str(out / "samples.pkl"),
               "--precision", str(cfg.inference.precision)]
    for key in ["compile", "resize_matmul", "cuda_graphs", "kv_cache"]:
        command.append("--" + ("" if cfg.inference[key] else "no-") + key.replace("_", "-"))
    env = dict(os.environ, NCM_ROOT=str(ROOT), TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS=str(cfg.benchmark.cpu_threads), MKL_NUM_THREADS=str(cfg.benchmark.cpu_threads))
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    print(f"Benchmarking {model.task} -> {out}", flush=True)
    with (out / "run.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
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
    result = json.loads((out / "results.json").read_text())
    print(f"{result['samples']} samples, batch {result['batch_size']}: "
          f"{result['median_seconds']:.3f} s, {result['samples_per_second']:.2f} samples/s, "
          f"{result['amortized_batch_ms']:.2f} ms/batch")


if __name__ == "__main__":
    main()
