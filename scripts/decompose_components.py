"""Attribute the top-k gather path's latency to its individual components.

The headline result is that a genuinely sparse gather is slower than dense
attention despite executing fewer FLOPs. That is a *what*, not a *why*. This
script settles the *why* by timing each stage in isolation:

    qkv          the input projection (shared by both paths)
    scores       q @ k^T + diagonal mask   (shared -- irreducible, D-022)
    softmax_N    softmax over all N peers  (dense only)
    topk_select  torch.topk on the scores  (sparse only)
    softmax_k    softmax over k scores     (sparse only)
    gather_v     building (B,N,k,d) from v (sparse only)
    agg_dense    (B,N,N) x (B,N,d)         (dense only)
    agg_sparse   (B*N,1,k) x (B*N,k,d)     (sparse only)
    proj         the output projection     (shared)

The decisive quantity is

    saving   = agg_dense - agg_sparse            (what sparsity buys)
    overhead = topk_select + gather_v + softmax_k - softmax_N
                                                 (what sparsity costs)

If `overhead > saving`, the loss is caused by the *selection and gather*
machinery rather than by any failure of the sparse contraction itself. That
distinction matters: selection overhead is intrinsic to data-dependent top-k
(you must rank all N scores), whereas a slow contraction would merely be an
implementation problem someone could fix.

Same protocol as the main microbenchmark: CUDA events, interleaved rep-by-rep
across components so contention lands on all of them, `min_ms` primary.

Writes results/component_decomposition.json and .md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent

SHARED = ("qkv", "scores", "proj")
DENSE_ONLY = ("softmax_N", "agg_dense")
SPARSE_ONLY = ("topk_select", "softmax_k", "gather_v", "agg_sparse")


def build_stages(b: int, n: int, k: int, d: int, in_dim: int, device: str,
                 dtype: torch.dtype) -> dict[str, Callable[[], Any]]:
    """One closure per stage, each operating on realistic pre-built inputs."""
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(b, n, in_dim, device=device, dtype=dtype, generator=g)
    w_qkv = torch.randn(3 * d, in_dim, device=device, dtype=dtype, generator=g)
    w_proj = torch.randn(d, d, device=device, dtype=dtype, generator=g)

    q = torch.randn(b, n, d, device=device, dtype=dtype, generator=g)
    kk = torch.randn(b, n, d, device=device, dtype=dtype, generator=g)
    v = torch.randn(b, n, d, device=device, dtype=dtype, generator=g)
    msg = torch.randn(b, n, d, device=device, dtype=dtype, generator=g)
    eye = torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)

    scores = torch.matmul(q, kk.transpose(-1, -2)) / (d ** 0.5)
    scores = scores.masked_fill(eye, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    top_scores, top_idx = scores.topk(k, dim=-1)
    idx_exp = top_idx.reshape(b, n * k, 1).expand(b, n * k, d)
    v_g = v.gather(1, idx_exp).view(b, n, k, d)
    w_k = F.softmax(top_scores, dim=-1)

    def qkv() -> Any:
        return F.linear(x, w_qkv)

    def scores_fn() -> Any:
        s = torch.matmul(q, kk.transpose(-1, -2)) / (d ** 0.5)
        return s.masked_fill(eye, float("-inf"))

    def softmax_N() -> Any:
        return F.softmax(scores, dim=-1)

    def topk_select() -> Any:
        return scores.topk(k, dim=-1)

    def softmax_k() -> Any:
        return F.softmax(top_scores, dim=-1)

    def gather_v() -> Any:
        return v.gather(1, idx_exp).view(b, n, k, d)

    def agg_dense() -> Any:
        return torch.matmul(weights, v)

    def agg_sparse() -> Any:
        return torch.matmul(w_k.unsqueeze(-2), v_g).squeeze(-2)

    def proj() -> Any:
        return F.linear(F.gelu(msg), w_proj)

    return {"qkv": qkv, "scores": scores_fn, "softmax_N": softmax_N,
            "topk_select": topk_select, "softmax_k": softmax_k,
            "gather_v": gather_v, "agg_dense": agg_dense,
            "agg_sparse": agg_sparse, "proj": proj}


def time_interleaved(stages: dict[str, Callable], device: str, reps: int,
                     warmup: int, runs: int) -> dict[str, dict[str, float]]:
    names = list(stages)
    on_cuda = device.startswith("cuda")
    # `torch.cuda.synchronize()` with no argument synchronises the *current*
    # device, which is 0 unless set. Timing tensors on cuda:1 against a
    # synchronise of cuda:0 waits for the wrong queue and returns the event
    # resolution floor (~0.4us) for every kernel regardless of size.
    if on_cuda:
        torch.cuda.set_device(device)

    def once(name: str) -> float:
        if not on_cuda:
            t0 = time.perf_counter()
            stages[name]()
            return (time.perf_counter() - t0) * 1e3
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        stages[name]()
        e.record()
        torch.cuda.synchronize(device)
        return s.elapsed_time(e)

    for _ in range(warmup):
        for nm in names:
            once(nm)
    if on_cuda:
        torch.cuda.synchronize(device)

    per_run: dict[str, list[list[float]]] = {nm: [] for nm in names}
    for _ in range(runs):
        block: dict[str, list[float]] = {nm: [] for nm in names}
        for _ in range(reps):
            for nm in names:          # interleaved
                block[nm].append(once(nm))
        for nm in names:
            per_run[nm].append(block[nm])

    out = {}
    for nm in names:
        flat = [x for bl in per_run[nm] for x in bl]
        meds = [statistics.median(bl) for bl in per_run[nm]]
        out[nm] = {
            "min_ms": min(flat),
            "median_ms": statistics.median(flat),
            "p25_ms": statistics.quantiles(flat, n=4)[0],
            "p75_ms": statistics.quantiles(flat, n=4)[2],
            "inter_run_spread_pct": (100.0 * (max(meds) - min(meds)) / min(meds)
                                     if min(meds) > 0 else 0.0),
        }
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--n-list", nargs="+", type=int,
                   default=[12, 48, 96, 192, 384, 512])
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--msg-dim", type=int, default=16)
    p.add_argument("--in-dim", type=int, default=64)
    p.add_argument("--k-frac", type=float, default=0.25)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--out", default=str(ROOT / "results" / "component_decomposition.json"))
    p.add_argument("--md", default=str(ROOT / "results" / "component_decomposition.md"))
    args = p.parse_args()

    rows = []
    for n in args.n_list:
        k = max(1, int(args.k_frac * (n - 1)))
        stages = build_stages(args.batch, n, k, args.msg_dim, args.in_dim,
                              args.device, torch.float32)
        t = time_interleaved(stages, args.device, args.reps, args.warmup,
                             args.runs)
        saving = t["agg_dense"]["min_ms"] - t["agg_sparse"]["min_ms"]
        overhead = (t["topk_select"]["min_ms"] + t["gather_v"]["min_ms"]
                    + t["softmax_k"]["min_ms"] - t["softmax_N"]["min_ms"])
        rows.append({
            "n_agents": n, "k": k, "batch": args.batch,
            "msg_dim": args.msg_dim, "stages": t,
            "aggregation_saving_ms": saving,
            "selection_overhead_ms": overhead,
            "net_ms": overhead - saving,
            "overhead_over_saving": (overhead / saving) if saving > 0 else None,
            "dominant_overhead_stage": max(
                ("topk_select", "gather_v"), key=lambda s: t[s]["min_ms"]),
        })
        print(f"N={n:>4} k={k:<4} saving {saving:+7.4f} ms   "
              f"overhead {overhead:+7.4f} ms   net {overhead - saving:+7.4f} ms"
              f"   (topk {t['topk_select']['min_ms']:.4f}, "
              f"gather {t['gather_v']['min_ms']:.4f}, "
              f"agg_dense {t['agg_dense']['min_ms']:.4f}, "
              f"agg_sparse {t['agg_sparse']['min_ms']:.4f})", flush=True)

    payload = {
        "device": args.device,
        "device_name": (torch.cuda.get_device_name(args.device)
                        if args.device.startswith("cuda") else "cpu"),
        "torch": torch.__version__,
        "batch": args.batch, "msg_dim": args.msg_dim, "in_dim": args.in_dim,
        "k_frac": args.k_frac, "reps": args.reps, "warmup": args.warmup,
        "runs": args.runs, "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# Where the top-k gather path's time actually goes", "",
             f"batch {args.batch}, msg_dim {args.msg_dim}, k = "
             f"{args.k_frac:.0%} of peers, {payload['device_name']}. "
             "Latency is `min_ms` over interleaved samples.", "",
             "`saving` is what the sparse contraction buys "
             "(`agg_dense - agg_sparse`); `overhead` is what selection costs "
             "(`topk_select + gather_v + softmax_k - softmax_N`). A positive "
             "`net` means selection overhead exceeds the aggregation saving.",
             "",
             "| N | k | topk_select | gather_v | softmax_N | agg_dense | "
             "agg_sparse | saving | overhead | net | overhead/saving |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["stages"]
        ratio = r["overhead_over_saving"]
        lines.append(
            f"| {r['n_agents']} | {r['k']} | {s['topk_select']['min_ms']:.4f} | "
            f"{s['gather_v']['min_ms']:.4f} | {s['softmax_N']['min_ms']:.4f} | "
            f"{s['agg_dense']['min_ms']:.4f} | {s['agg_sparse']['min_ms']:.4f} | "
            f"{r['aggregation_saving_ms']:+.4f} | "
            f"{r['selection_overhead_ms']:+.4f} | {r['net_ms']:+.4f} | "
            f"{ratio:.2f}x |" if ratio else
            f"| {r['n_agents']} | {r['k']} | {s['topk_select']['min_ms']:.4f} | "
            f"{s['gather_v']['min_ms']:.4f} | {s['softmax_N']['min_ms']:.4f} | "
            f"{s['agg_dense']['min_ms']:.4f} | {s['agg_sparse']['min_ms']:.4f} | "
            f"{r['aggregation_saving_ms']:+.4f} | "
            f"{r['selection_overhead_ms']:+.4f} | {r['net_ms']:+.4f} | — |")
    Path(args.md).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}\nwrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
