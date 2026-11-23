"""Warm end-to-end timing for the evaluation worker's actual inference callable."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import statistics
import subprocess
import time
from typing import Any, Mapping

from omegaconf import OmegaConf
import torch


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compile_state() -> dict[str, int]:
    from torch._dynamo.utils import counters

    return {
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "graph_breaks": sum(counters["graph_break"].values()),
    }


def gpu_telemetry() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used",
             "--format=csv"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


class InferenceBenchmark:
    """Own a benchmark's inputs, warmup, measurements and saved artifacts."""

    def __init__(self, namespace: Mapping[str, Any]):
        # Adapt the existing evaluation worker once; keep generation on its own path.
        self.args = namespace["a"]
        self.model_config = namespace["CFG"]
        self.config = OmegaConf.load(self.args.benchmark_config)
        self.settings = self.config.benchmark
        self.out = Path(self.args.out).parent
        self.device = torch.device("cuda", torch.cuda.current_device())

        self.unconditional = namespace["UNCOND"]
        self.items = namespace["items"]
        self.refs = namespace["refs"]
        self.raw_ids = namespace["RAW"]
        self.generate = namespace["run"]
        self.prepare_prefix = namespace["prefix"]
        self.graph_runner = namespace["generate_tokens"]
        self.eos_token_id = namespace["EOS"]
        self.cut_at_eos = namespace["CUT_AT_EOS"]
        self.checkpoint_dir = Path(namespace["ck"])
        self.data_dir = Path(namespace["dd"])
        levels = namespace["tk"].vqvae.vq.levels
        self.initial_length, self.output_length = int(levels[0]), int(levels[-1])
        probabilities = namespace["L0_P"]
        self.initial_probabilities = None if probabilities is None else probabilities.cpu()

        self.available = None if self.unconditional else len(self.items)
        self.count = int(self.settings.samples) if self.settings.samples is not None else self.available
        if not self.count or (self.available is not None and self.count > self.available):
            raise ValueError(f"Requested {self.count} samples from {self.available} inputs")
        self.batch_size = int(self.settings.batch_size)
        self.seed = int(self.settings.seed)
        self.batches = [
            (start, min(self.batch_size, self.count - start))
            for start in range(0, self.count, self.batch_size)
        ]

    def request(self, start: int, size: int) -> list[str]:
        """Include CPU input preparation and H2D transfers in every timed request."""
        initial_codes = None
        if self.unconditional:
            if self.initial_probabilities is not None:
                initial_codes = torch.multinomial(
                    self.initial_probabilities, size * self.initial_length, replacement=True,
                ).view(size, self.initial_length).to(self.device)
            prefix = torch.full((size, 1), self.eos_token_id, dtype=torch.long, device=self.device)
        else:
            prefix = torch.tensor(
                [self.prepare_prefix(item) for item in self.items[start:start + size]],
                dtype=torch.long, device=self.device,
            )
        return self.generate(prefix, initial_codes)

    def capture_count(self) -> int:
        return len(self.graph_runner.graphs) if self.args.cuda_graphs else 0

    def metadata(self) -> dict[str, Any]:
        props = torch.cuda.get_device_properties(self.device)
        paths = {
            key: str((self.checkpoint_dir / value).resolve())
            for key, value in self.model_config.checkpoints.items()
        }
        metadata = {
            "task": str(self.model_config.task),
            "samples": self.count,
            "batch_size": self.batch_size,
            "batches": len(self.batches),
            "seed": self.seed,
            "repeats": int(self.settings.repeats),
            "precision": self.args.precision,
            "soft_decisions": bool(self.model_config.generation.get("soft_decisions", False)),
            "soft_temperature": float(self.model_config.generation.get("soft_temperature", 1.0)),
            "vcycles": 0,
            "inference": OmegaConf.to_container(self.config.inference, resolve=True),
            "generation": OmegaConf.to_container(self.model_config.generation, resolve=True),
            "cut_at_eos": self.cut_at_eos,
            "full_split_size": self.available,
            "selection": "unconditional" if self.unconditional else "first N in stored order",
            "full_split": self.count == self.available if self.available is not None else None,
            "length_filtering": False,
            "partial_batch_size": self.count % self.batch_size,
            "device": props.name,
            "device_memory_bytes": props.total_memory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "checkpoint_sha256": {key: file_hash(path) for key, path in paths.items()},
            "checkpoint_paths": paths,
            "timing_includes": [
                "input preparation/tokenization", "initial-code sampling", "H2D",
                "all generator levels", "codec decoder", "token selection", "D2H", "detokenization",
            ],
            "timing_excludes": [
                "checkpoint/data loading", "compilation", "CUDA graph capture", "warmup",
                "RNG reset", "scoring", "file I/O",
            ],
            "timer": "perf_counter with CUDA synchronization before and after each complete job",
            "telemetry_before": gpu_telemetry(),
        }
        if not self.unconditional:
            split = str(self.model_config.eval.get("split", "test"))
            metadata["data_sha256"] = {
                key: file_hash(self.data_dir / self.model_config.data[key])
                for key in (split, split + "_refs")
            }
        root = Path(__file__).resolve().parents[1]
        files = [
            root / "benchmark.py", Path(__file__), root / "scripts/eval_shard.py",
            *sorted((root / "models").glob("*.py")), root / "utils/fast_generation.py",
            root / "utils/generation.py", root / "utils/inference.py",
        ]
        metadata["source_sha256"] = {str(path.relative_to(root)): file_hash(path) for path in files}
        return metadata

    def warmup(self) -> list[dict[str, Any]]:
        """Compile and capture every actual batch shape, including the final partial batch."""
        torch.manual_seed(self.seed)
        shapes = {}
        for start, size in self.batches:
            shapes.setdefault(size, start)

        warmups = []
        for size, start in shapes.items():
            torch.cuda.synchronize(self.device)
            began = time.perf_counter()
            for _ in range(int(self.settings.warmup)):
                self.request(start, size)
                self.raw_ids.clear()
            torch.cuda.synchronize(self.device)
            warmups.append({
                "batch_size": size,
                "requests": int(self.settings.warmup),
                "seconds": time.perf_counter() - began,
            })
            print(f"Warmup complete: batch size {size}", flush=True)
        return warmups

    def measure(self, repeat: int) -> tuple[dict[str, Any], list[str]]:
        """Time one complete generation job; reset RNG and collect stats outside the timer."""
        torch.manual_seed(self.seed)
        self.raw_ids.clear()
        gc.collect()
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

        texts = []
        began = time.perf_counter()
        for start, size in self.batches:
            texts.extend(self.request(start, size))
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - began

        if len(texts) != self.count or len(self.raw_ids) != self.count:
            raise RuntimeError("Inference did not return every requested sample")
        row = {
            "repeat": repeat,
            "seconds": seconds,
            "samples_per_second": self.count / seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            "text_sha256": hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
        }
        return row, texts

    def summarize(
        self,
        runs: list[dict[str, Any]],
        before: dict[str, int],
        after: dict[str, int],
        captures_before: int,
        captures_after: int,
    ) -> dict[str, Any]:
        stable = before == after and captures_before == captures_after
        seconds = [row["seconds"] for row in runs]
        median = statistics.median(seconds)
        return {
            "completed": stable,
            "samples": self.count,
            "batch_size": self.batch_size,
            "batches": len(self.batches),
            "median_seconds": median,
            "min_seconds": min(seconds),
            "max_seconds": max(seconds),
            "mean_seconds": statistics.mean(seconds),
            "stdev_seconds": statistics.stdev(seconds) if len(seconds) > 1 else 0.0,
            "samples_per_second": self.count / median,
            "amortized_batch_ms": median / len(self.batches) * 1000,
            "output_positions_per_second": self.count * self.output_length / median,
            "peak_allocated_gib": max(row["peak_allocated_bytes"] for row in runs) / 2**30,
            "peak_reserved_gib": max(row["peak_reserved_bytes"] for row in runs) / 2**30,
            "runs": runs,
            "compile_state_before": before,
            "compile_state_after": after,
            "graph_captures_before": captures_before,
            "graph_captures_after": captures_after,
            "no_compilation_or_capture_during_timing": stable,
            "repeat_texts_identical": len({row["text_sha256"] for row in runs}) == 1,
            "telemetry_after": gpu_telemetry(),
        }

    def save_samples(self, texts: list[str], raw_ids: list[list[int]]) -> None:
        samples = {
            "idxs": list(range(self.count)),
            "hyps": texts,
            "raw_ids": raw_ids,
            "refs": [""] * self.count if self.unconditional else self.refs[:self.count],
            "seed": self.seed,
        }
        with Path(self.args.out).open("wb") as stream:
            pickle.dump(samples, stream)

    @torch.inference_mode()
    def run(self) -> None:
        write_json(self.out / "metadata.json", self.metadata())
        write_json(self.out / "warmup.json", self.warmup())
        before, captures_before = compile_state(), self.capture_count()

        runs = []
        first_texts, first_ids = [], []
        for repeat in range(int(self.settings.repeats)):
            row, texts = self.measure(repeat)
            runs.append(row)
            if repeat == 0:
                first_texts, first_ids = texts, list(self.raw_ids)
            write_json(self.out / "runs.json", runs)
            print(
                f"Repeat {repeat + 1}/{self.settings.repeats}: {row['seconds']:.3f} s, "
                f"{row['samples_per_second']:.2f} samples/s",
                flush=True,
            )

        result = self.summarize(runs, before, compile_state(), captures_before, self.capture_count())
        write_json(self.out / "results.json", result)
        self.save_samples(first_texts, first_ids)
        if not result["completed"]:
            raise RuntimeError("Compilation or CUDA graph capture occurred during timing; increase warmup")


@torch.inference_mode()
def benchmark_inference(namespace: Mapping[str, Any]) -> None:
    """Benchmark the worker without changing its inference or preprocessing functions."""
    InferenceBenchmark(namespace).run()
