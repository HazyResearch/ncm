import argparse, os, subprocess, sys

import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="one generation shard per GPU, then merge and score")
    ap.add_argument("--config", required=True, help="eval config, e.g. config/eval/xsum.yaml")
    ap.add_argument("--checkpoint-dir", required=True, help="directory with the files named under checkpoints:")
    ap.add_argument("--data-dir", default="", help="directory with the files named under data: (not needed for unconditional tasks)")
    ap.add_argument("--out", default="", help="output directory (default eval/<task>)")
    ap.add_argument("--gpus", default="", help="comma-separated GPU ids (default: all visible)")
    ap.add_argument("--n", type=int, default=0, help="evaluate only the first n inputs")
    args, passthrough = ap.parse_known_args()

    cfg = OmegaConf.load(args.config)
    gpus = args.gpus.split(",") if args.gpus else [str(i) for i in range(torch.cuda.device_count())]
    if not gpus:
        # no CUDA: run a single shard on the local device (MPS on Apple silicon, else CPU)
        gpus = [None]
    out = args.out or os.path.join("eval", cfg.task)
    os.makedirs(out, exist_ok=True)

    common = ["--config", args.config, "--checkpoint-dir", args.checkpoint_dir] + (["--data-dir", args.data_dir] if args.data_dir else []) + [
              "--nshards", str(len(gpus))] + (["--n", str(args.n)] if args.n else []) + passthrough
    procs = []
    for k, gpu in enumerate(gpus):
        env = dict(os.environ, TOKENIZERS_PARALLELISM="false")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        log = open(os.path.join(out, f"shard_{k}.log"), "w")
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "scripts", "eval_shard.py"), *common,
             "--shard", str(k), "--out", os.path.join(out, f"shard_{k}.pkl")],
            env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT))
    where = f"GPU(s) {','.join(gpus)}" if gpus != [None] else "local device (no CUDA)"
    print(f"[evaluate] {len(gpus)} shard(s) on {where} -> {out}/", flush=True)
    failed = [k for k, p in enumerate(procs) if p.wait() != 0]
    if failed:
        sys.exit(f"shard(s) {failed} failed; see {out}/shard_*.log")

    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "eval_merge.py"), "--config", args.config,
                    "--glob", os.path.join(out, "shard_*.pkl")], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
