"""Warm end-to-end timing for scripts/eval_shard.py's actual inference callable."""
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import time

import torch
from omegaconf import OmegaConf


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_state():
    from torch._dynamo.utils import counters
    return {"unique_graphs": int(counters["stats"]["unique_graphs"]),
            "graph_breaks": sum(counters["graph_break"].values())}


def telemetry():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used",
             "--format=csv"], capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


@torch.inference_mode()
def benchmark_inference(namespace):
    """Reuse the worker's model, preprocessing, token selection and text decoding."""
    ns = namespace
    args, model_cfg = ns["a"], ns["CFG"]
    cfg = OmegaConf.load(args.benchmark_config)
    settings = cfg.benchmark
    out = Path(args.out).parent
    device = torch.device("cuda", torch.cuda.current_device())
    unconditional = ns["UNCOND"]
    items = ns["items"]
    available = None if unconditional else len(items)
    count = int(settings.samples) if settings.samples is not None else available
    if not count or (available is not None and count > available):
        raise ValueError(f"Requested {count} samples from {available} inputs")
    batch_size = int(settings.batch_size)
    batches = [(start, min(batch_size, count - start)) for start in range(0, count, batch_size)]
    seed = int(settings.seed)
    l0_prob = None if ns["L0_P"] is None else ns["L0_P"].cpu()
    level0 = int(ns["tk"].vqvae.vq.levels[0])

    def request(start, size):
        if unconditional:
            # Include CPU sampling and H2D in the same scope as released generation.
            l0 = (torch.multinomial(l0_prob, size * level0, replacement=True)
                  .view(size, level0).to(device)) if l0_prob is not None else None
            prefix = torch.full((size, 1), ns["EOS"], dtype=torch.long, device=device)
        else:
            prefix = torch.tensor([ns["prefix"](item) for item in items[start:start + size]],
                                  dtype=torch.long, device=device)
            l0 = None
        return ns["run"](prefix, l0)

    props = torch.cuda.get_device_properties(device)
    paths = {key: str((Path(ns["ck"]) / value).resolve())
             for key, value in model_cfg.checkpoints.items()}
    metadata = {
        "task": str(model_cfg.task), "samples": count, "batch_size": batch_size,
        "batches": len(batches), "seed": seed, "repeats": int(settings.repeats),
        "precision": args.precision, "soft_decisions": False, "vcycles": 0,
        "inference": OmegaConf.to_container(cfg.inference, resolve=True),
        "generation": OmegaConf.to_container(model_cfg.generation, resolve=True),
        "cut_at_eos": ns["CUT_AT_EOS"],
        "full_split_size": available, "selection": "unconditional" if unconditional else "first N in stored order",
        "full_split": count == available if available is not None else None,
        "length_filtering": False, "partial_batch_size": count % batch_size,
        "device": props.name, "device_memory_bytes": props.total_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(), "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "checkpoint_sha256": {key: file_hash(path) for key, path in paths.items()},
        "checkpoint_paths": paths,
        "timing_includes": ["input preparation/tokenization", "initial-code sampling", "H2D",
                            "all generator levels", "codec decoder", "token selection", "D2H", "detokenization"],
        "timing_excludes": ["checkpoint/data loading", "compilation", "CUDA graph capture", "warmup",
                            "RNG reset", "scoring", "file I/O"],
        "timer": "perf_counter with CUDA synchronization before and after each complete job",
        "telemetry_before": telemetry(),
    }
    if not unconditional:
        split = str(model_cfg.eval.get("split", "test"))
        metadata["data_sha256"] = {key: file_hash(Path(ns["dd"]) / model_cfg.data[key])
                                   for key in [split, split + "_refs"]}
    root = Path(__file__).resolve().parents[1]
    files = [root / "benchmark.py", Path(__file__), root / "scripts/eval_shard.py",
             *sorted((root / "models").glob("*.py")), root / "utils/fast_generation.py"]
    metadata["source_sha256"] = {str(p.relative_to(root)): file_hash(p) for p in files}
    save(out / "metadata.json", metadata)

    # Warm every actual batch shape, including a short final batch. Compile/capture
    # work may be much slower than steady state and must not enter the timed jobs.
    torch.manual_seed(seed)
    shapes = {}
    for start, size in batches:
        shapes.setdefault(size, start)
    warmups = []
    for size, start in shapes.items():
        torch.cuda.synchronize(device)
        began = time.perf_counter()
        for _ in range(int(settings.warmup)):
            request(start, size)
            ns["RAW"].clear()
        torch.cuda.synchronize(device)
        warmups.append({"batch_size": size, "requests": int(settings.warmup),
                        "seconds": time.perf_counter() - began})
        print(f"Warmup complete: batch size {size}", flush=True)
    save(out / "warmup.json", warmups)
    before = compile_state()
    graph_count = len(ns["generate_tokens"].graphs) if args.cuda_graphs else 0
    runs, first_texts, first_ids = [], None, None
    for repeat in range(int(settings.repeats)):
        torch.manual_seed(seed)
        ns["RAW"].clear()
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        texts = []
        began = time.perf_counter()
        for start, size in batches:
            texts.extend(request(start, size))
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - began
        if len(texts) != count or len(ns["RAW"]) != count:
            raise RuntimeError("Inference did not return every requested sample")
        row = {"repeat": repeat, "seconds": seconds, "samples_per_second": count / seconds,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
               "text_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()}
        runs.append(row)
        if first_texts is None:
            first_texts, first_ids = texts, list(ns["RAW"])
        save(out / "runs.json", runs)
        print(f"Repeat {repeat + 1}/{settings.repeats}: {seconds:.3f} s, "
              f"{count / seconds:.2f} samples/s", flush=True)
    after = compile_state()
    captures_after = len(ns["generate_tokens"].graphs) if args.cuda_graphs else 0
    stable = before == after and graph_count == captures_after
    seconds = [r["seconds"] for r in runs]
    median = statistics.median(seconds)
    result = {
        "completed": stable, "samples": count, "batch_size": batch_size, "batches": len(batches),
        "median_seconds": median, "min_seconds": min(seconds), "max_seconds": max(seconds),
        "mean_seconds": statistics.mean(seconds),
        "stdev_seconds": statistics.stdev(seconds) if len(seconds) > 1 else 0.0,
        "samples_per_second": count / median, "amortized_batch_ms": median / len(batches) * 1000,
        "output_positions_per_second": count * int(ns["tk"].vqvae.vq.levels[-1]) / median,
        "peak_allocated_gib": max(r["peak_allocated_bytes"] for r in runs) / 2**30,
        "peak_reserved_gib": max(r["peak_reserved_bytes"] for r in runs) / 2**30,
        "runs": runs, "compile_state_before": before, "compile_state_after": after,
        "graph_captures_before": graph_count, "graph_captures_after": captures_after,
        "no_compilation_or_capture_during_timing": stable,
        "repeat_texts_identical": len({r["text_sha256"] for r in runs}) == 1,
        "telemetry_after": telemetry(),
    }
    save(out / "results.json", result)
    with Path(args.out).open("wb") as stream:
        pickle.dump({"idxs": list(range(count)), "hyps": first_texts, "raw_ids": first_ids,
                     "refs": [""] * count if unconditional else ns["refs"][:count],
                     "seed": seed}, stream)
    if not stable:
        raise RuntimeError("Compilation or CUDA graph capture occurred during timing; increase warmup")
