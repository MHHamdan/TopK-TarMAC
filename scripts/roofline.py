"""Roofline account of why fewer FLOPs does not mean less time.

The component decomposition shows the gather path losing even though it
executes fewer multiply-accumulates. A roofline explains that without appeal
to implementation quality: the sparse path has *lower arithmetic intensity*
than the dense one, so it sits further into the memory-bound region, where
time is set by bytes moved rather than by FLOPs performed.

Arithmetic intensity is computed analytically per stage as
    AI = FLOPs / bytes moved to and from DRAM
counting each tensor read and written once (i.e. assuming no cache reuse
between stages, which is the right assumption here: every intermediate is
larger than L2 at the N where the gap opens).

The machine balance -- the AI above which a kernel is compute-bound -- is
**measured**, not taken from a datasheet: a large GEMM gives achievable
FLOP/s, a large device-to-device copy gives achievable bandwidth, and their
ratio is the ridge point. This keeps the analysis honest about the specific
device and dtype, and makes the CPU comparison meaningful, since the two
architectures have very different balances.

Writes results/roofline.json and .md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
F32 = 4  # bytes


# ---------------------------------------------------------------------------
# analytic intensities
# ---------------------------------------------------------------------------

def dense_stages(b: int, n: int, d: int) -> list[dict[str, Any]]:
    """FLOPs and DRAM traffic per stage of materialised dense attention."""
    return [
        {"stage": "scores  q@k^T",
         "flops": 2 * b * n * n * d,
         "bytes": F32 * (2 * b * n * d + b * n * n)},
        {"stage": "softmax over N",
         "flops": 5 * b * n * n,
         "bytes": F32 * (2 * b * n * n)},
        {"stage": "aggregate (B,N,N)x(B,N,d)",
         "flops": 2 * b * n * n * d,
         "bytes": F32 * (b * n * n + b * n * d + b * n * d)},
    ]


def gather_stages(b: int, n: int, k: int, d: int) -> list[dict[str, Any]]:
    """Same for the genuinely sparse gather path."""
    return [
        {"stage": "scores  q@k^T",
         "flops": 2 * b * n * n * d,
         "bytes": F32 * (2 * b * n * d + b * n * n)},
        # top-k performs comparisons, not floating-point work: it moves the
        # whole score matrix and emits 2*B*N*k values, for zero FLOPs. This is
        # the stage that drags the path's intensity down.
        {"stage": "top-k select",
         "flops": 0,
         "bytes": F32 * (b * n * n + 2 * b * n * k)},
        {"stage": "softmax over k",
         "flops": 5 * b * n * k,
         "bytes": F32 * (2 * b * n * k)},
        # The gather reads v with a scattered access pattern and writes a
        # (B,N,k,d) tensor: again no FLOPs, substantial traffic.
        {"stage": "gather v -> (B,N,k,d)",
         "flops": 0,
         "bytes": F32 * (b * n * k + b * n * k * d + b * n * k * d)},
        {"stage": "aggregate (B*N,1,k)x(B*N,k,d)",
         "flops": 2 * b * n * k * d,
         "bytes": F32 * (b * n * k + b * n * k * d + b * n * d)},
    ]


def path_ai(stages: list[dict[str, Any]]) -> dict[str, float]:
    fl = sum(s["flops"] for s in stages)
    by = sum(s["bytes"] for s in stages)
    return {"flops": fl, "bytes": by, "ai": fl / by if by else 0.0}


# ---------------------------------------------------------------------------
# measured machine balance
# ---------------------------------------------------------------------------

def measure_peak_flops(device: str, reps: int = 30) -> float:
    """Achievable fp32 GEMM throughput, FLOP/s."""
    n = 8192 if device.startswith("cuda") else 2048
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    on_cuda = device.startswith("cuda")
    for _ in range(5):
        a @ b
    if on_cuda:
        torch.cuda.synchronize(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        a @ b
        if on_cuda:
            torch.cuda.synchronize(device)
        ts.append(time.perf_counter() - t0)
    return 2.0 * n ** 3 / min(ts)


def measure_peak_bandwidth(device: str, reps: int = 30) -> float:
    """Achievable copy bandwidth, bytes/s (read + write counted)."""
    elems = 256 * 1024 * 1024 // F32 if device.startswith("cuda") else \
        32 * 1024 * 1024 // F32
    src = torch.empty(elems, device=device, dtype=torch.float32)
    dst = torch.empty_like(src)
    on_cuda = device.startswith("cuda")
    for _ in range(5):
        dst.copy_(src)
    if on_cuda:
        torch.cuda.synchronize(device)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        dst.copy_(src)
        if on_cuda:
            torch.cuda.synchronize(device)
        ts.append(time.perf_counter() - t0)
    return 2.0 * elems * F32 / min(ts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", nargs="+", default=["cuda:1", "cpu"])
    p.add_argument("--n-list", nargs="+", type=int, default=[12, 48, 192, 512])
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--msg-dim", type=int, default=16)
    p.add_argument("--k-frac", type=float, default=0.25)
    p.add_argument("--out", default=str(ROOT / "results" / "roofline.json"))
    p.add_argument("--md", default=str(ROOT / "results" / "roofline.md"))
    args = p.parse_args()

    machines = {}
    for dev in args.devices:
        if dev.startswith("cuda"):
            if not torch.cuda.is_available():
                continue
            torch.cuda.set_device(dev)
        fl = measure_peak_flops(dev)
        bw = measure_peak_bandwidth(dev)
        machines[dev] = {
            "name": (torch.cuda.get_device_name(dev)
                     if dev.startswith("cuda") else "x86_64 CPU"),
            "achievable_gflops": fl / 1e9,
            "achievable_gbytes_per_s": bw / 1e9,
            "machine_balance_flops_per_byte": fl / bw,
        }
        print(f"{dev:<8} {machines[dev]['name'][:38]:<40} "
              f"{fl / 1e12:7.2f} TFLOP/s  {bw / 1e9:8.1f} GB/s  "
              f"ridge point {fl / bw:7.2f} FLOP/byte", flush=True)

    rows = []
    for n in args.n_list:
        k = max(1, int(args.k_frac * (n - 1)))
        ds = dense_stages(args.batch, n, args.msg_dim)
        gs = gather_stages(args.batch, n, k, args.msg_dim)
        rows.append({
            "n_agents": n, "k": k, "batch": args.batch, "d": args.msg_dim,
            "dense": {**path_ai(ds), "stages": ds},
            "gather": {**path_ai(gs), "stages": gs},
        })

    payload = {"machines": machines, "rows": rows,
               "note": "AI below the ridge point means memory-bound: time is "
                       "set by bytes moved, so removing FLOPs cannot help."}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# Roofline: why fewer FLOPs does not buy less time", "",
             "## Measured machine balance", "",
             "| Device | Achievable fp32 GEMM | Achievable bandwidth | "
             "Ridge point (FLOP/byte) |", "|---|---|---|---|"]
    for dev, m in machines.items():
        lines.append(f"| `{dev}` {m['name']} | {m['achievable_gflops'] / 1e3:.2f} "
                     f"TFLOP/s | {m['achievable_gbytes_per_s']:.0f} GB/s | "
                     f"**{m['machine_balance_flops_per_byte']:.1f}** |")

    lines += ["", f"## Path arithmetic intensity "
              f"(batch {args.batch}, d={args.msg_dim}, k={args.k_frac:.0%} of peers)",
              "",
              "| N | k | dense GFLOP | dense GB | dense AI | gather GFLOP | "
              "gather GB | gather AI | AI ratio (gather/dense) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d_, g_ = r["dense"], r["gather"]
        lines.append(
            f"| {r['n_agents']} | {r['k']} | {d_['flops'] / 1e9:.3f} | "
            f"{d_['bytes'] / 1e9:.3f} | **{d_['ai']:.2f}** | "
            f"{g_['flops'] / 1e9:.3f} | {g_['bytes'] / 1e9:.3f} | "
            f"**{g_['ai']:.2f}** | {g_['ai'] / d_['ai']:.2f}x |")

    biggest = rows[-1]
    lines += ["", f"## Stage breakdown at N={biggest['n_agents']}, "
              f"k={biggest['k']}", "",
              "| Path | Stage | GFLOP | GB moved | AI |",
              "|---|---|---|---|---|"]
    for path in ("dense", "gather"):
        for s in biggest[path]["stages"]:
            ai = s["flops"] / s["bytes"] if s["bytes"] else 0.0
            lines.append(f"| {path} | {s['stage']} | {s['flops'] / 1e9:.3f} | "
                         f"{s['bytes'] / 1e9:.3f} | {ai:.2f} |")

    lines += ["", "## Reading", "",
              "The two zero-FLOP stages the gather path adds -- the top-k "
              "selection and the gather itself -- move substantial memory for "
              "no arithmetic, so they have arithmetic intensity 0 and pull the "
              "whole path below the dense path's intensity. Both paths sit far "
              "beneath every ridge point measured above, i.e. both are "
              "memory-bound, and in that regime latency tracks bytes moved. "
              "Removing multiply-accumulates from a memory-bound kernel does "
              "not make it faster; adding traffic to one makes it slower.", "",
              "This is why the result reproduces on CPU despite a machine "
              "balance an order of magnitude different from the GPU's: the "
              "conclusion follows from the ratio of the two paths' intensities, "
              "which is a property of the algorithm, not of the device.", ""]
    Path(args.md).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}\nwrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
