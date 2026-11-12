"""Hardware counters for every kernel each implementation launches.

The byte criterion and the claim that selection runs far below the bandwidth
roof are argued from analytic traffic and measured time. This script asks the
hardware: per kernel, the DRAM bytes moved, the L2 sectors touched, achieved
DRAM throughput as a fraction of peak, SM throughput and achieved occupancy,
under Nsight Compute.

Two modes:

    --one IMPL N K     run IMPL once inside a cudaProfilerStart/Stop window
                       (after warm-up calls that compile and cache), for ncu
    (default)          for every (impl, N, k) in the grid, run ncu on --one,
                       parse its CSV, and write one JSON

Implementations and tilings are those of scripts/bench_streaming_topk.py, so
the tuned dense and streaming tilings come from results/autotune*.json of the
checkout this runs in. Counters are collected with --cache-control all and
--clock-control base, ncu's defaults, which flush caches and pin clocks per
kernel: the numbers describe each kernel cold, not the timed sweeps.

Writes results/ncu_counters.json (or --out).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sectors.sum",
    "lts__t_sector_hit_rate.pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
]

IMPLS = ["dense_fused", "dense_tuned", "topk_gather", "topk_fused_gather",
         "streaming", "fixed_set"]
RESIDENT_ONLY = {"topk_fused_gather", "fixed_set"}
STREAMING_MAX_K = 8  # the streaming kernel's register candidate buffer


def run_one(impl: str, N: int, k: int, d: int, B: int) -> None:
    import torch
    from scripts.bench_attention_core import (_scores, dense_fused,
                                              topk_fused_gather, topk_gather)
    from scripts.bench_streaming_topk import dense_tuned, streaming
    from src.kernels.fused_prescore_attention import fused_prescore_attention

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda:0"
    g = torch.Generator(device=dev).manual_seed(1234 + N * 7 + d)
    q, kt, v = (torch.randn(B, N, d, device=dev, generator=g) for _ in range(3))
    eye = torch.eye(N, device=dev, dtype=torch.bool).unsqueeze(0)
    idx = _scores(q, kt, eye).topk(k, dim=-1, sorted=False).indices
    fns = {
        "dense_fused": lambda: dense_fused(q, kt, v, eye, k),
        "dense_tuned": lambda: dense_tuned(q, kt, v, eye, k),
        "topk_gather": lambda: topk_gather(q, kt, v, eye, k),
        "topk_fused_gather": lambda: topk_fused_gather(q, kt, v, eye, k),
        "streaming": lambda: streaming(q, kt, v, eye, k),
        "fixed_set": lambda: fused_prescore_attention(q, kt, v, idx),
    }
    f = fns[impl]
    with torch.no_grad():
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        f()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


def _parse(text: str) -> list[dict]:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    rows = [r for r in rows if r.get("ID", "").isdigit()]  # drop the units row
    out = []
    for r in rows:
        rec = {"kernel": r["Kernel Name"][:160]}
        for m in METRICS:
            val = r.get(m, "").replace(",", "")
            try:
                rec[m] = float(val)
            except ValueError:
                rec[m] = None
        out.append(rec)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--one", nargs=3, metavar=("IMPL", "N", "K"))
    p.add_argument("--impls", nargs="+", default=IMPLS)
    p.add_argument("--n-list", nargs="+", type=int, default=[192, 512, 1024])
    p.add_argument("--k-list", nargs="+", type=int, default=[1, 4, 16])
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--ncu", default=os.environ.get("NCU", "ncu"))
    p.add_argument("--out", default=str(ROOT / "results" / "ncu_counters.json"))
    args = p.parse_args()

    if args.one:
        impl, N, k = args.one[0], int(args.one[1]), int(args.one[2])
        run_one(impl, N, k, args.d, args.batch)
        return 0

    t0 = time.time()
    groups, failures = [], []
    for N in args.n_list:
        for k in args.k_list:
            for impl in args.impls:
                if impl in RESIDENT_ONLY and N > 512:
                    continue
                if impl == "streaming" and k > STREAMING_MAX_K:
                    continue
                cmd = [args.ncu, "--profile-from-start", "off", "--csv",
                       "--page", "raw", "--print-units", "base",
                       "--metrics", ",".join(METRICS),
                       sys.executable, __file__, "--one", impl, str(N), str(k),
                       "--d", str(args.d), "--batch", str(args.batch)]
                r = subprocess.run(cmd, capture_output=True, text=True)
                try:
                    kernels = _parse(r.stdout)
                except StopIteration:
                    failures.append({"impl": impl, "N": N, "k": k,
                                     "stderr": (r.stderr + r.stdout)[-2000:]})
                    print(f"FAILED {impl} N={N} k={k}\n{failures[-1]['stderr']}",
                          flush=True)
                    continue
                tot = {
                    "time_s": sum(x["gpu__time_duration.sum"] or 0
                                  for x in kernels) * 1e-9,
                    "dram_bytes": sum((x["dram__bytes_read.sum"] or 0)
                                      + (x["dram__bytes_write.sum"] or 0)
                                      for x in kernels),
                    "l2_sectors": sum(x["lts__t_sectors.sum"] or 0
                                      for x in kernels),
                }
                tot["dram_GBps"] = (tot["dram_bytes"] / tot["time_s"] / 1e9
                                    if tot["time_s"] else None)
                groups.append({"impl": impl, "n_agents": N, "k": k,
                               "head_dim": args.d, "batch": args.batch,
                               "n_kernels": len(kernels), "total": tot,
                               "kernels": kernels})
                print(f"{impl:<18} N={N:<5} k={k:<3} kernels={len(kernels)} "
                      f"t={tot['time_s'] * 1e6:8.1f}us "
                      f"dram={tot['dram_bytes'] / 1e6:8.2f}MB "
                      f"({tot['dram_GBps'] or 0:7.1f} GB/s)", flush=True)

    import torch
    ver = subprocess.run([args.ncu, "--version"], capture_output=True,
                         text=True).stdout.strip().splitlines()
    Path(args.out).write_text(json.dumps({
        "meta": {"script": "scripts/ncu_counters.py", "args": vars(args),
                 "device_name": torch.cuda.get_device_name(0),
                 "torch": torch.__version__, "ncu": ver[-1] if ver else None,
                 "metrics": METRICS, "elapsed_s": time.time() - t0},
        "groups": groups, "failures": failures}, indent=1))
    print(f"wrote {args.out}: {len(groups)} groups, {len(failures)} failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
