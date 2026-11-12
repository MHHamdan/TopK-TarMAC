"""Per-stage cost accounting: arithmetic, traffic, intensity and achieved rate.

The manuscript explains a latency ordering by an arithmetic-intensity argument
applied to whole implementations. A reviewer asked for that argument at the
granularity it is actually made at -- the individual stage -- so this script
reports, for every stage of the dense and sparse paths:

    flops     multiply--accumulates x2, analytic
    bytes     global-memory traffic, analytic, each tensor read and written once
    ai        flops / bytes, the analytic arithmetic intensity
    time      measured in isolation under the manuscript's timing protocol
    gflops    flops / time, the achieved arithmetic rate
    gbps      bytes / time, the achieved bandwidth
    roof_frac the larger of (achieved rate / peak rate) and (achieved bandwidth
              / peak bandwidth), i.e. how close the stage runs to whichever of
              the two machine limits binds it

The last column is what makes the table diagnostic rather than descriptive: a
stage far below both limits is badly implemented, while a stage near the
bandwidth limit at low intensity is doing the only thing it can.

Peak rates are measured on the same device in the same process, a large GEMM
for arithmetic and a large device-to-device copy for bandwidth, so the fractions
are against what this machine attains rather than against a datasheet.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# machine limits, measured
# ---------------------------------------------------------------------------

def measure_peaks(device: str, reps: int = 30) -> dict[str, float]:
    n = 8192
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize(device)
    ts = []
    for _ in range(reps):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record(); a @ b; e.record(); torch.cuda.synchronize(device)
        ts.append(s.elapsed_time(e))
    gemm_flops = 2 * n ** 3
    peak_flops = gemm_flops / (min(ts) * 1e-3)

    m = 1 << 27                                    # 512 MB source buffer
    # Written, not merely allocated: an untouched host allocation reads the
    # shared zero page, and the same habit on the device leaves the contents
    # undefined for no benefit.
    src = torch.rand(m, device=device, dtype=torch.float32)
    dst = torch.rand(m, device=device, dtype=torch.float32)
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize(device)
    ts = []
    for _ in range(reps):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record(); dst.copy_(src); e.record(); torch.cuda.synchronize(device)
        ts.append(s.elapsed_time(e))
    peak_bw = 2 * m * 4 / (min(ts) * 1e-3)
    del a, b, src, dst
    torch.cuda.empty_cache()
    return {"peak_flops": peak_flops, "peak_bytes_per_s": peak_bw,
            "ridge_flop_per_byte": peak_flops / peak_bw,
            "gemm_n": n, "copy_bytes": 2 * m * 4}


# ---------------------------------------------------------------------------
# stages, with their analytic accounting alongside the closure that runs them
# ---------------------------------------------------------------------------

def build(B: int, N: int, k: int, d: int, device: str
          ) -> dict[str, tuple[Callable, int, int, str]]:
    """name -> (closure, flops, bytes, which path the stage belongs to)."""
    g = torch.Generator(device=device).manual_seed(0)
    f = 4                                          # fp32
    q = torch.randn(B, N, d, device=device, generator=g)
    kk = torch.randn(B, N, d, device=device, generator=g)
    v = torch.randn(B, N, d, device=device, generator=g)
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)

    scores = torch.matmul(q, kk.transpose(-1, -2)) / (d ** 0.5)
    scores = scores.masked_fill(eye, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    top_s, top_i = scores.topk(k, dim=-1, sorted=False)
    w_k = F.softmax(top_s, dim=-1)
    idx = top_i.reshape(B, N * k, 1).expand(B, N * k, d)
    v_g = v.gather(1, idx).view(B, N, k, d)

    smat = B * N * N * f
    kmat = B * N * k * f
    vec = B * N * d * f
    gath = B * N * k * d * f
    idxb = B * N * k * 8

    stages: dict[str, tuple[Callable, int, int, str]] = {}

    stages["scores"] = (
        lambda: torch.matmul(q, kk.transpose(-1, -2)).masked_fill_(
            eye, float("-inf")),
        2 * B * N * N * d, 2 * vec + smat, "shared")

    stages["softmax_N"] = (
        lambda: F.softmax(scores, dim=-1), 5 * B * N * N, 2 * smat, "dense")

    stages["aggregate_dense"] = (
        lambda: torch.matmul(weights, v),
        2 * B * N * N * d, smat + vec + vec, "dense")

    stages["topk_select"] = (
        lambda: scores.topk(k, dim=-1, sorted=False),
        # a selection kernel performs comparisons rather than multiply--adds;
        # the arithmetic column is not meaningful for it and is recorded as 0
        0, smat + kmat + idxb, "sparse")

    stages["softmax_k"] = (
        lambda: F.softmax(top_s, dim=-1), 5 * B * N * k, 2 * kmat, "sparse")

    stages["gather_v"] = (
        lambda: v.gather(1, idx).view(B, N, k, d),
        0, vec + idxb + gath, "sparse")

    stages["aggregate_sparse"] = (
        lambda: torch.matmul(w_k.unsqueeze(-2), v_g).squeeze(-2),
        2 * B * N * k * d, kmat + gath + vec, "sparse")

    return stages


def time_stages(stages: dict[str, tuple[Callable, int, int, str]], device: str,
                reps: int, warmup: int, runs: int) -> dict[str, float]:
    names = list(stages)

    def one(n: str) -> float:
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        with torch.no_grad():
            stages[n][0]()
        e.record()
        torch.cuda.synchronize(device)
        return s.elapsed_time(e)

    for _ in range(warmup):
        for n in names:
            one(n)
    torch.cuda.synchronize(device)
    samples = {n: [] for n in names}
    for _r in range(runs):
        for _i in range(reps):
            for n in names:
                samples[n].append(one(n))
    return {n: min(samples[n]) for n in names}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--n-list", nargs="+", type=int, default=[48, 192, 512])
    p.add_argument("--k-frac", type=float, default=0.25)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--head-dim", type=int, default=16)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--out", default=str(ROOT / "results" / "stage_accounting.json"))
    args = p.parse_args()

    device = args.device
    torch.cuda.set_device(device)
    t0 = time.time()
    peaks = measure_peaks(device)
    print(f"peak {peaks['peak_flops']/1e12:.1f} TFLOP/s  "
          f"{peaks['peak_bytes_per_s']/1e9:.0f} GB/s  "
          f"ridge {peaks['ridge_flop_per_byte']:.1f} FLOP/byte\n")

    rows: list[dict[str, Any]] = []
    for N in args.n_list:
        k = max(1, int(args.k_frac * (N - 1)))
        stages = build(args.batch, N, k, args.head_dim, device)
        times = time_stages(stages, device, args.reps, args.warmup, args.runs)
        print(f"--- N={N} k={k} batch={args.batch} d={args.head_dim}")
        for name, (_fn, fl, by, path) in stages.items():
            t = times[name]
            s = t * 1e-3
            gf = fl / s / 1e9
            gb = by / s / 1e9
            frac = max(fl / s / peaks["peak_flops"],
                       by / s / peaks["peak_bytes_per_s"])
            rows.append({
                "n_agents": N, "k": k, "batch": args.batch,
                "head_dim": args.head_dim, "stage": name, "path": path,
                "flops": fl, "bytes": by,
                "ai": (fl / by) if by else 0.0,
                "time_ms": t, "gflops": gf, "gbytes_per_s": gb,
                "frac_of_binding_roof": frac,
            })
            print(f"  {name:<18} {path:<7} {t:8.4f} ms  "
                  f"AI {(fl/by if by else 0):6.2f}  "
                  f"{gf:8.1f} GFLOP/s  {gb:8.1f} GB/s  "
                  f"roof {100*frac:5.1f}%")
        del stages
        torch.cuda.empty_cache()

    payload = {
        "device": device, "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "dtype": "fp32", "batch": args.batch, "head_dim": args.head_dim,
        "k_frac": args.k_frac, "reps": args.reps, "runs": args.runs,
        "peaks": peaks, "elapsed_s": round(time.time() - t0, 1), "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
