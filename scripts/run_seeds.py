"""Launch N seeds of a config and append rows to results/MASTER_LOG.csv.

Default: sequential — fits the constrained-VRAM environment (~4 GB free per
GPU). Parallel mode (`--parallel K`) spawns K subprocesses on `--gpus`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MASTER_LOG = ROOT / "results" / "MASTER_LOG.csv"


def append_master_log(row: dict) -> None:
    fieldnames = ["timestamp", "git_sha", "config", "run_name", "seed",
                  "total_steps", "final_eval_mean", "final_eval_std",
                  "wall_time_s", "gpu_hours"]
    new = not MASTER_LOG.exists()
    MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MASTER_LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "nogit"


def run_one(config: str, seed: int, run_name: str, total_steps: int,
            gpu: int) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(ROOT)
    t0 = time.time()
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py"),
           "--config", config, "--seed", str(seed),
           "--run-name", run_name, "--total-steps", str(total_steps)]
    print("launch", " ".join(cmd), "GPU", gpu, flush=True)
    proc = subprocess.run(cmd, env=env, cwd=ROOT, check=False,
                          capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        print("STDERR", proc.stderr[-2000:])
        return {"seed": seed, "ok": False, "wall_time_s": wall}
    out_dir = ROOT / "results" / run_name / f"seed_{seed}"
    eval_json = json.loads((out_dir / "final_eval.json").read_text())
    return {
        "seed": seed, "ok": True, "wall_time_s": wall,
        "final_eval_mean": eval_json["eval_return_mean"],
        "final_eval_std": eval_json["eval_return_std"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--total-steps", type=int, required=True)
    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument("--parallel", type=int, default=1,
                   help="Run this many seeds simultaneously.")
    args = p.parse_args()

    sha = git_sha()
    completed: list[dict] = []

    # Sequential fallback when parallel == 1 or only one GPU.
    if args.parallel == 1:
        for seed in args.seeds:
            gpu = args.gpus[0]
            r = run_one(args.config, seed, args.run_name, args.total_steps, gpu)
            completed.append(r)
            if r["ok"]:
                append_master_log({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "git_sha": sha,
                    "config": args.config,
                    "run_name": args.run_name,
                    "seed": seed,
                    "total_steps": args.total_steps,
                    "final_eval_mean": r["final_eval_mean"],
                    "final_eval_std": r["final_eval_std"],
                    "wall_time_s": r["wall_time_s"],
                    "gpu_hours": r["wall_time_s"] / 3600,
                })
    else:
        # Pool of pending seeds → GPU assignment.
        pending = list(args.seeds)
        active: list[tuple[subprocess.Popen, int, int, float, str]] = []
        gpu_pool = list(args.gpus)
        while pending or active:
            while pending and len(active) < args.parallel and gpu_pool:
                seed = pending.pop(0)
                gpu = gpu_pool.pop(0)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["PYTHONPATH"] = str(ROOT)
                cmd = [sys.executable, str(ROOT / "scripts" / "train.py"),
                       "--config", args.config, "--seed", str(seed),
                       "--run-name", args.run_name,
                       "--total-steps", str(args.total_steps)]
                t0 = time.time()
                log_path = ROOT / "results" / args.run_name / f"seed_{seed}_launch.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_f = log_path.open("w")
                proc = subprocess.Popen(cmd, env=env, cwd=ROOT,
                                        stdout=log_f, stderr=subprocess.STDOUT)
                active.append((proc, seed, gpu, t0, str(log_path)))
                print(f"launch seed={seed} gpu={gpu} pid={proc.pid}")
            time.sleep(5)
            still_active = []
            for proc, seed, gpu, t0, log_path in active:
                if proc.poll() is None:
                    still_active.append((proc, seed, gpu, t0, log_path))
                    continue
                wall = time.time() - t0
                gpu_pool.append(gpu)
                if proc.returncode != 0:
                    print(f"FAIL seed={seed} rc={proc.returncode} log={log_path}")
                    continue
                out_dir = ROOT / "results" / args.run_name / f"seed_{seed}"
                eval_json = json.loads((out_dir / "final_eval.json").read_text())
                append_master_log({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "git_sha": sha,
                    "config": args.config,
                    "run_name": args.run_name,
                    "seed": seed,
                    "total_steps": args.total_steps,
                    "final_eval_mean": eval_json["eval_return_mean"],
                    "final_eval_std": eval_json["eval_return_std"],
                    "wall_time_s": wall,
                    "gpu_hours": wall / 3600,
                })
                completed.append({"seed": seed, "ok": True,
                                  "wall_time_s": wall,
                                  "final_eval_mean": eval_json["eval_return_mean"]})
                print(f"done seed={seed} wall={wall:.0f}s eval={eval_json['eval_return_mean']:.2f}")
            active = still_active

    print("ALL DONE", json.dumps(completed, indent=2))


if __name__ == "__main__":
    main()
