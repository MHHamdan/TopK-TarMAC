"""Map the region of (d, batch, N, k) where gathering actually wins.

Reporting two favourable cells as anomalies is not a characterisation. This
sweeps head dimension d in {32,64,128,256} against batch in {32,256,4096} and,
for every (d, batch, N), takes the *best* k for the gather path -- the most
favourable case sparsity can construct for itself -- and records its ratio to
both dense baselines.

Reuses the main microbenchmark's `prepare_cell` / `interleaved_timing`, so the
protocol (CUDA events, rep-by-rep interleaving, min_ms) is identical rather
than merely similar.

Cells whose activation memory would exceed --mem-budget-mib are skipped and
recorded as skipped, never silently dropped: at d=256, batch 4096, N=512 the
gathered value tensor alone is ~273 GiB.

Writes results/crossover_surface.json/.md and paper/figures/fig_crossover.pdf
(figure only if --figure).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from scripts.microbenchmark_comm import (
    analytic_activation_bytes, interleaved_timing, prepare_cell, release,
)

ROOT = Path(__file__).resolve().parent.parent


def cell_fits(arm: str, b: int, n: int, k: int, d: int,
              budget_mib: float) -> bool:
    mib = analytic_activation_bytes(arm, b, n, k, d, 4) / 2 ** 20
    # x3 headroom: the closed form covers the dominant tensor, not the whole
    # forward, and the backward keeps activations alive.
    return mib * 3 < budget_mib


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--d-list", nargs="+", type=int, default=[32, 64, 128, 256])
    p.add_argument("--batch-list", nargs="+", type=int, default=[32, 256, 4096])
    p.add_argument("--n-list", nargs="+", type=int, default=[12, 48, 192, 512])
    p.add_argument("--k-frac", nargs="+", type=float,
                   default=[0.01, 0.0625, 0.25, 0.5])
    p.add_argument("--in-dim", type=int, default=64)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--mem-budget-mib", type=float, default=20000.0)
    p.add_argument("--figure", action="store_true")
    p.add_argument("--out", default=str(ROOT / "results" / "crossover_surface.json"))
    p.add_argument("--md", default=str(ROOT / "results" / "crossover_surface.md"))
    args = p.parse_args()

    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)

    cells: list[dict[str, Any]] = []
    for d in args.d_list:
        for b in args.batch_list:
            for n in args.n_list:
                ks = sorted({max(1, int(f * (n - 1))) for f in args.k_frac})
                skipped = [k for k in ks
                           if not cell_fits("topk_gather", b, n, k, d,
                                            args.mem_budget_mib)]
                ks = [k for k in ks if k not in skipped]
                dense_fits = cell_fits("dense", b, n, ks[0] if ks else 1, d,
                                       args.mem_budget_mib)
                if not ks or not dense_fits:
                    cells.append({"d": d, "batch": b, "n_agents": n,
                                  "skipped": True,
                                  "reason": "exceeds memory budget"})
                    print(f"d={d:<4} B={b:<5} N={n:<4} SKIPPED "
                          f"(memory budget)", flush=True)
                    continue

                group = [prepare_cell("dense", n, ks[0], args.k_frac[0], b, d,
                                      args.in_dim, args.device, "eager",
                                      torch.float32),
                         prepare_cell("dense_sdpa", n, ks[0], args.k_frac[0], b,
                                      d, args.in_dim, args.device, "eager",
                                      torch.float32)]
                for k in ks:
                    group.append(prepare_cell("topk_gather", n, k, k / (n - 1),
                                              b, d, args.in_dim, args.device,
                                              "eager", torch.float32))
                interleaved_timing(group, args.device, args.reps, args.warmup,
                                   args.runs, do_backward=False)

                def lat(row: dict) -> float | None:
                    f = row.get("forward")
                    return f["min_ms"] if f and not row["oom"] else None

                dn = next((lat(r) for r in group if r["arm"] == "dense"), None)
                sd = next((lat(r) for r in group if r["arm"] == "dense_sdpa"),
                          None)
                gathers = [(r["k"], lat(r)) for r in group
                           if r["arm"] == "topk_gather" and lat(r)]
                release(group, args.device)

                if not gathers or dn is None:
                    cells.append({"d": d, "batch": b, "n_agents": n,
                                  "skipped": True, "reason": "OOM at runtime"})
                    continue
                best_k, best = min(gathers, key=lambda t: t[1])
                row = {
                    "d": d, "batch": b, "n_agents": n, "skipped": False,
                    "best_k": best_k, "gathered_ms": best,
                    "dense_ms": dn, "sdpa_ms": sd,
                    "ratio_vs_dense": best / dn,
                    "ratio_vs_sdpa": (best / sd) if sd else None,
                    "wins_vs_dense": best < dn,
                    "wins_vs_sdpa": bool(sd and best < sd),
                    "all_k": [{"k": k, "ms": v} for k, v in gathers],
                    "skipped_k_for_memory": skipped,
                }
                cells.append(row)
                print(f"d={d:<4} B={b:<5} N={n:<4} best k={best_k:<4} "
                      f"gath {best:8.4f} dense {dn:8.4f} "
                      f"ratio {best / dn:6.2f}x vs dense, "
                      f"{(best / sd if sd else float('nan')):6.2f}x vs SDPA "
                      f"{'WIN' if best < dn else ''}", flush=True)

    payload = {
        "device": args.device,
        "device_name": (torch.cuda.get_device_name(args.device)
                        if args.device.startswith("cuda") else "cpu"),
        "torch": torch.__version__,
        "in_dim": args.in_dim, "reps": args.reps, "runs": args.runs,
        "mem_budget_mib": args.mem_budget_mib,
        "note": "ratio is best-k gathered latency over dense; <1 means "
                "gathering wins",
        "cells": cells,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    live = [c for c in cells if not c.get("skipped")]
    wins_d = [c for c in live if c["wins_vs_dense"]]
    wins_s = [c for c in live if c["wins_vs_sdpa"]]
    lines = ["# Crossover surface — where does gathering win?", "",
             f"{len(live)} measured cells, {len(cells) - len(live)} skipped "
             f"for memory. For each (d, batch, N) the *best* k for the gather "
             f"path is used, i.e. the most favourable case sparsity can "
             f"construct.", "",
             f"- Cells where gathering beats **unfused dense**: "
             f"**{len(wins_d)}/{len(live)}**",
             f"- Cells where gathering beats **fused SDPA**: "
             f"**{len(wins_s)}/{len(live)}**", ""]
    if wins_d:
        lines += ["Winning cells vs unfused dense:", "",
                  "| d | batch | N | best k | gathered (ms) | dense (ms) | ratio |",
                  "|---|---|---|---|---|---|---|"]
        for c in wins_d:
            lines.append(f"| {c['d']} | {c['batch']} | {c['n_agents']} | "
                         f"{c['best_k']} | {c['gathered_ms']:.4f} | "
                         f"{c['dense_ms']:.4f} | {c['ratio_vs_dense']:.2f}x |")
    lines += ["", "## Full surface (ratio of best-k gathered to unfused dense)",
              "", "| d | batch | " + " | ".join(f"N={n}" for n in args.n_list)
              + " |", "|" + "---|" * (2 + len(args.n_list))]
    for d in args.d_list:
        for b in args.batch_list:
            cs = []
            for n in args.n_list:
                c = next((x for x in cells if x["d"] == d and x["batch"] == b
                          and x["n_agents"] == n), None)
                cs.append("skip" if not c or c.get("skipped")
                          else f"{c['ratio_vs_dense']:.2f}")
            lines.append(f"| {d} | {b} | " + " | ".join(cs) + " |")
    Path(args.md).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}\nwrote {args.md}")

    if args.figure:
        make_figure(cells, args)
    return 0


def make_figure(cells: list[dict], args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: F401

    rows = [(d, b) for d in args.d_list for b in args.batch_list]
    grid = np.full((len(rows), len(args.n_list)), np.nan)
    for i, (d, b) in enumerate(rows):
        for j, n in enumerate(args.n_list):
            c = next((x for x in cells if x["d"] == d and x["batch"] == b
                      and x["n_agents"] == n and not x.get("skipped")), None)
            if c:
                grid[i, j] = c["ratio_vs_dense"]

    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(rows) + 1.9))
    norm = LogNorm(vmin=max(np.nanmin(grid), 1e-2), vmax=np.nanmax(grid))
    im = ax.imshow(grid, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(args.n_list)),
                  [str(n) for n in args.n_list])
    ax.set_yticks(range(len(rows)), [f"d={d}, B={b}" for d, b in rows],
                  fontsize=8)
    ax.set_xlabel("agents $N$")
    ax.set_title("best-$k$ gathered top-$k$ latency $\\div$ dense\n"
                 "(<1 = sparsity wins; blank = exceeded memory budget)",
                 fontsize=9)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if grid[i, j] > 2 or grid[i, j] < 0.5
                        else "black")
    fig.colorbar(im, ax=ax, label="ratio (log scale)")
    fig.tight_layout()
    out = ROOT / "paper" / "figures" / "fig_crossover.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(ROOT / "results" / "fig_crossover.png", dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
