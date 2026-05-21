"""Build paper figures from results/.

Outputs:
  paper/figures/fig2_learning_curves_n6.pdf  — method vs baseline at N=6.
  paper/figures/fig3_scaling_return.pdf     — final return vs N.
  paper/figures/fig4_flops_pareto.pdf       — FLOPs vs return.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_curves(run: str) -> pd.DataFrame | None:
    seed_dirs = sorted((RESULTS / run).glob("seed_*"))
    frames = []
    for d in seed_dirs:
        f = d / "metrics.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["seed"] = int(d.name.removeprefix("seed_"))
        df["run"] = run
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def collect_eval(run: str) -> list[float]:
    out = []
    for d in sorted((RESULTS / run).glob("seed_*")):
        f = d / "final_eval.json"
        if f.exists():
            out.append(float(json.loads(f.read_text())["eval_return_mean"]))
    return out


def plot_curves(runs: list[tuple[str, str]], out: Path, title: str = "",
                metric: str = "ep_return_mean") -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for run, lbl in runs:
        df = load_curves(run)
        if df is None or df.empty:
            continue
        max_step = df["global_step"].max()
        bins = np.linspace(0, max_step, 26)
        df["bucket"] = pd.cut(df["global_step"], bins, labels=False, include_lowest=True)
        agg = (df.groupby(["bucket", "seed"])[metric].mean().reset_index()
                 .groupby("bucket")[metric].agg(["mean", "std", "count"]))
        x = bins[:-1] + np.diff(bins) / 2
        x = x[: len(agg)]
        m, s = agg["mean"].values, agg["std"].fillna(0).values
        n_seeds = int(agg["count"].max())
        ax.plot(x, m, label=f"{lbl} ($n{{=}}{n_seeds}$)")
        ax.fill_between(x, m - s, m + s, alpha=0.2)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Team episode return")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_scaling(N_list: list[int], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = [("baseline", "MAPPO"), ("dense", "Dense Attn-Comm"),
               ("topk", "Adaptive TopK (ours)")]
    for tag, lbl in methods:
        xs, ms, ss = [], [], []
        for N in N_list:
            vals = collect_eval(f"mappo_mpe_n{N}_{tag}")
            if not vals:
                continue
            xs.append(N)
            ms.append(np.mean(vals))
            ss.append(np.std(vals))
        if xs:
            xs, ms, ss = np.array(xs), np.array(ms), np.array(ss)
            ax.errorbar(xs, ms, yerr=ss, marker="o", capsize=3, label=lbl)
    ax.set_xlabel("Number of agents $N$")
    ax.set_ylabel("Final greedy eval mean return")
    ax.set_xticks(N_list)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_flops_pareto(flops_path: Path, out: Path) -> None:
    rows = json.loads(flops_path.read_text())
    fig, ax = plt.subplots(figsize=(6, 4))
    by_method: dict[str, list[tuple[int, float, float]]] = {}
    for r in rows:
        if r.get("error"):
            continue
        run = r["run"]
        if "topk" in run:
            method = "Adaptive TopK"
        elif "dense" in run:
            method = "Dense"
        else:
            method = "Baseline"
        N = r.get("n_agents", 0)
        ret_vals = collect_eval(run)
        if not ret_vals:
            continue
        flops_ratio = r.get("flops", {}).get("msg_gather_over_dense", 1.0) if r.get("use_comm") else 0.0
        by_method.setdefault(method, []).append((N, flops_ratio, float(np.mean(ret_vals))))

    for method, pts in by_method.items():
        pts.sort()
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        Ns = [p[0] for p in pts]
        ax.scatter(xs, ys, label=method)
        for N, x, y in zip(Ns, xs, ys):
            ax.annotate(f"N={N}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Comm message-gather FLOPs / dense (= $k_{\\mathrm{eff}}/(N-1)$)")
    ax.set_ylabel("Mean return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6, help="N for the headline curves figure.")
    p.add_argument("--n-list", nargs="+", type=int, default=[3, 6, 12])
    p.add_argument("--flops-summary", default=str(RESULTS / "flops_summary.json"))
    args = p.parse_args()

    plot_curves(
        [(f"mappo_mpe_n{args.n}_baseline", "MAPPO (baseline)"),
         (f"mappo_mpe_n{args.n}_dense", "Dense Attn-Comm"),
         (f"mappo_mpe_n{args.n}_topk", "Adaptive TopK (ours)")],
        out=FIGURES / f"fig2_learning_curves_n{args.n}.pdf",
        title=f"MPE \\texttt{{simple\\_spread}} ($N{{=}}{args.n}$): learning curves",
    )
    plot_scaling(args.n_list, FIGURES / "fig3_scaling_return.pdf")
    flops_path = Path(args.flops_summary)
    if flops_path.exists():
        plot_flops_pareto(flops_path, FIGURES / "fig4_flops_pareto.pdf")


if __name__ == "__main__":
    main()
