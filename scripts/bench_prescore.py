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
from src.kernels.fused_dense_attention import fused_dense_attention  # noqa: E402
from src.kernels.fused_prescore_attention import (  # noqa: E402
    fused_prescore_attention)


def tuned_dense(N: int, d: int, batch: int) -> dict[str, int] | None:
    """The best dense tiling the manuscript's own search found at this shape.

    Section 4 quotes every crossover against the better of the vendor library
    and a searched dense kernel. Quoting the pre-score results against an
    untuned dense kernel would apply a weaker standard to the mechanism this
    paper proposes than to the one it criticises, so the same searched
    configuration is used here.
    """
    best = None
    for name in ("autotune.json", "autotune_small.json"):
        f = ROOT / "results" / name
        if not f.exists():
            continue
        for r in json.loads(f.read_text())["rows"]:
            if (r.get("kind") == "dense" and r["n_agents"] == N
                    and r["head_dim"] == d and r["batch"] == batch):
                if best is None or r["best"]["ms"] < best["ms"]:
                    best = r["best"]
    return best["cfg"] if best else None


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

        def dense_triton():
            return fused_dense_attention(q, kt, v)

        cfg = tuned_dense(N, d, B)

        def dense_tuned(cfg=cfg):
            return fused_dense_attention(
                q, kt, v, block_m=cfg["BLOCK_M"], block_kn=cfg["BLOCK_KN"],
                num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])

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
                "dense_triton": dense_triton,
                "topk_gather": gathered,
            }
            if cfg is not None:
                fns["dense_tuned"] = dense_tuned
            for sel_name, sel in SELECTORS.items():
                fns[f"sel:{sel_name}"] = (lambda s=sel, k=k: s(pos, k))
                fns[f"total:{sel_name}"] = (
                    lambda s=sel, k=k: prescore_attention(q, kt, v, s(pos, k)))
            # The attention stage alone, with selection hoisted out, isolates
            # what the sparse attention costs once the set is known. Timed both
            # as a composition of library operators and as a fused kernel,
            # because the manuscript's own argument about the score-then-select
            # path is that a composed constant is an upper bound.
            for sel_name, sel in SELECTORS.items():
                idx_fixed = sel(pos, k)
                fns[f"attn:{sel_name}"] = (
                    lambda i=idx_fixed: prescore_attention(q, kt, v, i))
                fns[f"fused:{sel_name}"] = (
                    lambda i=idx_fixed: fused_prescore_attention(q, kt, v, i))
                fns[f"total_fused:{sel_name}"] = (
                    lambda s=sel, k=k: fused_prescore_attention(
                        q, kt, v, s(pos, k)))

            timed = time_interleaved(fns, device, args.reps, args.warmup,
                                     args.runs)
            for sel_name in SELECTORS:
                fl = analytic_flops(sel_name, B, N, d, k)
                rows.append({
                    "n_agents": N, "k": k, "batch": B, "head_dim": d,
                    "selector": sel_name,
                    "t_selector_ms": timed[f"sel:{sel_name}"]["min_ms"],
                    "t_attention_ms": timed[f"attn:{sel_name}"]["min_ms"],
                    "t_attention_fused_ms": timed[f"fused:{sel_name}"]["min_ms"],
                    "t_total_ms": timed[f"total:{sel_name}"]["min_ms"],
                    "t_total_fused_ms":
                        timed[f"total_fused:{sel_name}"]["min_ms"],
                    "t_dense_unfused_ms": timed["dense_unfused"]["min_ms"],
                    "t_dense_fused_ms": timed["dense_fused"]["min_ms"],
                    "t_dense_triton_ms": timed["dense_triton"]["min_ms"],
                    "t_dense_tuned_ms": (timed["dense_tuned"]["min_ms"]
                                         if cfg is not None else None),
                    "dense_tuned_cfg": cfg,
                    # The baseline every ratio is quoted against: the better of
                    # the library fused path and a dense kernel written in the
                    # same framework as the selectors' attention stage. Quoting
                    # against the library alone would be the pitfall this paper
                    # names in its own measurement-error section.
                    "t_dense_best_ms": min(
                        [timed["dense_fused"]["min_ms"],
                         timed["dense_triton"]["min_ms"]]
                        + ([timed["dense_tuned"]["min_ms"]]
                           if cfg is not None else [])),
                    "t_topk_gather_ms": timed["topk_gather"]["min_ms"],
                    "flops": fl,
                    "flop_ratio_vs_dense": fl["dense_total"] / fl["total"],
                    # Dispersion for every quantity a claim is read off, not
                    # only for the composed total: the conclusions rest on the
                    # fused columns, and a margin has to be compared against
                    # the spread of the number that carries it.
                    "spread_pct": timed[f"total:{sel_name}"][
                        "inter_run_spread_pct"],
                    "spread_pct_by_series": {
                        series: timed[f"{series}:{sel_name}"][
                            "inter_run_spread_pct"]
                        for series in ("sel", "attn", "fused", "total",
                                       "total_fused")},
                    "dispersion": {
                        series: {kk: timed[f"{series}:{sel_name}"][kk]
                                 for kk in ("min_ms", "median_ms", "p25_ms",
                                            "p75_ms")}
                        for series in ("sel", "attn", "fused", "total",
                                       "total_fused")},
                    "dense_spread_pct": {
                        nm: timed[nm]["inter_run_spread_pct"]
                        for nm in ("dense_fused", "dense_triton")
                        + (("dense_tuned",) if cfg is not None else ())},
                    # The dense baselines' own dispersion, so that a ratio can
                    # be recomputed under an estimator other than the minimum
                    # without re-running the sweep.
                    "dense_dispersion": {
                        nm: {kk: timed[nm][kk] for kk in
                             ("min_ms", "median_ms", "p25_ms", "p75_ms")}
                        for nm in ("dense_fused", "dense_triton")
                        + (("dense_tuned",) if cfg is not None else ())},
                })
                r = rows[-1]
                print(f"N={N:>4} k={k:<3} {sel_name:<14} "
                      f"sel {r['t_selector_ms']:7.4f} attn "
                      f"{r['t_attention_ms']:7.4f} (fused "
                      f"{r['t_attention_fused_ms']:7.4f}) total "
                      f"{r['t_total_ms']:7.4f} (fused "
                      f"{r['t_total_fused_ms']:7.4f}) ms | dense best "
                      f"{r['t_dense_best_ms']:7.4f} (lib "
                      f"{r['t_dense_fused_ms']:.4f} ours "
                      f"{r['t_dense_triton_ms']:.4f}) | FLOP ratio "
                      f"{r['flop_ratio_vs_dense']:6.2f}x | time ratio "
                      f"{r['t_total_ms'] / r['t_dense_best_ms']:5.2f}x",
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
