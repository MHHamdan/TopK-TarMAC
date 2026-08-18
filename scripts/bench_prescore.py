"""Does selecting peers before scoring them convert saved operations into time?

Proposition 2 says a selector that fixes the retained set without forming the
complete score vector is not bound by the two-fold ceiling, and that its
operation-count ratio against dense attention approaches N/k. Whether the
removed arithmetic appears as removed time is a separate question, and it is the
one the rest of the manuscript shows cannot be answered by a count. This script
answers it for two concrete pre-score selectors.

Timed, separately and together:

    selector        positions -> retained peer indices, with no score formed
    attention       scores for the Nk retained pairs only, softmax, aggregate
    total           the two together, which is what a deployment would pay

against dense attention through both the unfused and the fused path, and
against the score-then-select gathered implementation at the same k.

The comparison is a cost comparison. The pre-score paths attend to a different
set of peers than top-k does, so they compute a different function; what is
being asked is whether the cost structure Proposition 2 describes is realisable,
not whether spatial neighbours are as informative as attention-selected ones.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.kernels.prescore_selection import (  # noqa: E402
    SELECTORS, analytic_flops, prescore_attention)


def _stats(s: list[float]) -> dict[str, float]:
    s = sorted(s)
    q = statistics.quantiles(s, n=4) if len(s) >= 4 else [s[0]] * 3
    return {"min_ms": s[0], "median_ms": statistics.median(s),
            "p25_ms": q[0], "p75_ms": q[2]}


def time_interleaved(fns: dict[str, Any], device: str, reps: int, warmup: int,
                     runs: int) -> dict[str, dict[str, float]]:
    names = list(fns)

    def one(name: str) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            fns[name]()
        end.record()
        torch.cuda.synchronize(device)
        return start.elapsed_time(end)

    for _ in range(warmup):
        for n in names:
            one(n)
    torch.cuda.synchronize(device)
    per_run = {n: [] for n in names}
    for _r in range(runs):
        block = {n: [] for n in names}
        for _i in range(reps):
            for n in names:
                block[n].append(one(n))
        for n in names:
            per_run[n].append(block[n])
    out = {}
    for n in names:
        flat = [x for b in per_run[n] for x in b]
        med = [statistics.median(b) for b in per_run[n]]
        st = _stats(flat)
        st["inter_run_spread_pct"] = (
            100.0 * (max(med) - min(med)) / min(med) if min(med) > 0 else 0.0)
        out[n] = st
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--n-list", nargs="+", type=int,
                   default=[48, 96, 192, 384, 512])
    p.add_argument("--k-list", nargs="+", type=int, default=[2, 4, 8, 16])
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--head-dim", type=int, default=16)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--out", default=str(ROOT / "results" / "prescore.json"))
    args = p.parse_args()

    device, B, d = args.device, args.batch, args.head_dim
    torch.cuda.set_device(device)
    t0 = time.time()
    rows: list[dict[str, Any]] = []

    for N in args.n_list:
        gen = torch.Generator(device=device).manual_seed(4242 + N)
        q = torch.randn(B, N, d, device=device, generator=gen)
        kt = torch.randn(B, N, d, device=device, generator=gen)
        v = torch.randn(B, N, d, device=device, generator=gen)
        pos = torch.rand(B, N, 2, device=device, generator=gen)
        eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)

        def dense_unfused():
            s = torch.matmul(q, kt.transpose(-1, -2)) / (d ** 0.5)
            return torch.matmul(F.softmax(s.masked_fill(eye, float("-inf")),
                                          dim=-1), v)

        mask = torch.zeros(N, N, device=device).masked_fill_(
            eye.squeeze(0), float("-inf"))

        def dense_fused():
            return F.scaled_dot_product_attention(
                q.unsqueeze(1), kt.unsqueeze(1), v.unsqueeze(1),
                attn_mask=mask).squeeze(1)

        for k in args.k_list:
            if k >= N:
                continue

            def gathered(k=k):
                s = torch.matmul(q, kt.transpose(-1, -2)) / (d ** 0.5)
                s = s.masked_fill(eye, float("-inf"))
                ts, ti = s.topk(k, dim=-1, sorted=False)
                w = F.softmax(ts, dim=-1)
                idx = ti.reshape(B, N * k, 1).expand(B, N * k, d)
                vg = v.gather(1, idx).view(B, N, k, d)
                return torch.matmul(w.unsqueeze(-2), vg).squeeze(-2)

            fns: dict[str, Any] = {
                "dense_unfused": dense_unfused,
                "dense_fused": dense_fused,
                "topk_gather": gathered,
            }
            for sel_name, sel in SELECTORS.items():
                fns[f"sel:{sel_name}"] = (lambda s=sel, k=k: s(pos, k))
                fns[f"total:{sel_name}"] = (
                    lambda s=sel, k=k: prescore_attention(q, kt, v, s(pos, k)))
            # The attention stage alone, with selection hoisted out, isolates
            # what the sparse attention costs once the set is known.
            for sel_name, sel in SELECTORS.items():
                idx_fixed = sel(pos, k)
                fns[f"attn:{sel_name}"] = (
                    lambda i=idx_fixed: prescore_attention(q, kt, v, i))

            timed = time_interleaved(fns, device, args.reps, args.warmup,
                                     args.runs)
            for sel_name in SELECTORS:
                fl = analytic_flops(sel_name, B, N, d, k)
                rows.append({
                    "n_agents": N, "k": k, "batch": B, "head_dim": d,
                    "selector": sel_name,
                    "t_selector_ms": timed[f"sel:{sel_name}"]["min_ms"],
                    "t_attention_ms": timed[f"attn:{sel_name}"]["min_ms"],
                    "t_total_ms": timed[f"total:{sel_name}"]["min_ms"],
                    "t_dense_unfused_ms": timed["dense_unfused"]["min_ms"],
                    "t_dense_fused_ms": timed["dense_fused"]["min_ms"],
                    "t_topk_gather_ms": timed["topk_gather"]["min_ms"],
                    "flops": fl,
                    "flop_ratio_vs_dense": fl["dense_total"] / fl["total"],
                    "spread_pct": timed[f"total:{sel_name}"][
                        "inter_run_spread_pct"],
                })
                r = rows[-1]
                print(f"N={N:>4} k={k:<3} {sel_name:<14} "
                      f"sel {r['t_selector_ms']:7.4f} attn "
                      f"{r['t_attention_ms']:7.4f} total "
                      f"{r['t_total_ms']:7.4f} ms | dense unfused "
                      f"{r['t_dense_unfused_ms']:7.4f} fused "
                      f"{r['t_dense_fused_ms']:7.4f} | FLOP ratio "
                      f"{r['flop_ratio_vs_dense']:6.2f}x | time ratio vs fused "
                      f"{r['t_total_ms'] / r['t_dense_fused_ms']:5.2f}x",
                      flush=True)
        del q, kt, v, pos, eye
        torch.cuda.empty_cache()

    payload = {
        "device": device,
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "batch": B, "head_dim": d, "dtype": "fp32",
        "reps": args.reps, "warmup": args.warmup, "runs": args.runs,
        "timing": "cuda_events_interleaved_min",
        "elapsed_s": round(time.time() - t0, 1),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
