"""Build LaTeX tables from results/MASTER_LOG.csv and seed metrics.

Outputs:
  paper/tables/table_headline.tex — method vs baseline per N, mean ± std.
  paper/tables/table_ablation.tex — ablation cells per condition.

Numbers come from final_eval.json (greedy, 64 episodes) aggregated across
seeds in a run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TABLES = ROOT / "paper" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def collect_eval(run: str) -> list[float]:
    seed_dirs = sorted((RESULTS / run).glob("seed_*"))
    out = []
    for d in seed_dirs:
        f = d / "final_eval.json"
        if f.exists():
            j = json.loads(f.read_text())
            out.append(float(j["eval_return_mean"]))
    return out


def fmt_mean_std(vals: list[float]) -> str:
    if not vals:
        return "--"
    a = np.array(vals)
    return f"${a.mean():.2f} \\pm {a.std():.2f}$"


def paired_test(a: list[float], b: list[float]) -> float:
    """Welch's t-test p-value, two-sided. NaN if too few samples."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    t = stats.ttest_ind(a, b, equal_var=False)
    return float(t.pvalue)


def cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    aa = np.asarray(a, float); bb = np.asarray(b, float)
    pooled = np.sqrt((aa.var(ddof=1) + bb.var(ddof=1)) / 2.0)
    if pooled == 0:
        return float("nan")
    return float((bb.mean() - aa.mean()) / pooled)


def bootstrap_ci(a: list[float], b: list[float], n: int = 10000,
                 rng_seed: int = 0) -> tuple[float, float]:
    """Two-sample percentile bootstrap CI for the mean difference (b - a)."""
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    aa = np.asarray(a, float); bb = np.asarray(b, float)
    diffs = np.empty(n)
    for i in range(n):
        sa = rng.choice(aa, size=aa.size, replace=True)
        sb = rng.choice(bb, size=bb.size, replace=True)
        diffs[i] = sb.mean() - sa.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return (float(lo), float(hi))


def load_random_baseline() -> dict:
    p = RESULTS / "random_baseline.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def headline_table(N_list: list[int]) -> str:
    rows = []
    random_b = load_random_baseline()
    for N in N_list:
        base = collect_eval(f"mappo_mpe_n{N}_baseline")
        dense = collect_eval(f"mappo_mpe_n{N}_dense")
        topk = collect_eval(f"mappo_mpe_n{N}_topk")
        p_dense = paired_test(base, dense)
        p_topk = paired_test(base, topk)
        rinfo = random_b.get(str(N), {})
        r_mean = rinfo.get("mean")
        r_std = rinfo.get("std")
        rand_cell = (rf"${r_mean:.2f} \pm {r_std:.2f}$"
                     if r_mean is not None else "--")

        def cell(vals: list[float], better_p: float = 1.0) -> str:
            if not vals:
                return "--"
            txt = fmt_mean_std(vals)
            if better_p < 0.05:
                txt = "\\textbf{" + txt + "}"
            return txt + f" ($n{{=}}{len(vals)}$)"

        rows.append((N, rand_cell,
                     cell(base),
                     cell(dense, better_p=p_dense if (np.mean(dense or [-1e9]) >
                                                       np.mean(base or [-1e9])) else 1.0),
                     cell(topk, better_p=p_topk if (np.mean(topk or [-1e9]) >
                                                      np.mean(base or [-1e9])) else 1.0),
                     p_dense, p_topk))

    head = (r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lccccc c}
\toprule
$N$ & Random & MAPPO & + Dense Attn-Comm & + Adaptive TopK (ours) &
$p_{\text{dense}}$ & $p_{\text{topk}}$\\
\midrule
""")
    body = "\n".join(
        rf"{N} & {rand} & {b} & {d} & {t} & {pd_:.3f} & {pt:.3f} \\"
        for (N, rand, b, d, t, pd_, pt) in rows
    )
    tail = r"""
\bottomrule
\end{tabular}
\caption{Headline result: greedy eval mean $\pm$ std across seeds, MPE
\texttt{simple\_spread} for $N \in \{N_LIST\}$ agents. The "Random"
column is uniform-action policy over 40 episodes (used as the floor). Bold
= significantly better than baseline (Welch's $t$-test, $p<0.05$). $p$
columns report the two-sided Welch p-value vs MAPPO.}
\label{tab:headline}
\end{table}"""
    tail = tail.replace("N_LIST", ", ".join(str(N) for N in N_list))
    return head + body + tail


def stats_json(N_list: list[int]) -> dict:
    """Detailed numbers for the manuscript: mean, std, p, d, bootstrap CI."""
    out = {}
    for N in N_list:
        base = collect_eval(f"mappo_mpe_n{N}_baseline")
        dense = collect_eval(f"mappo_mpe_n{N}_dense")
        topk = collect_eval(f"mappo_mpe_n{N}_topk")
        out[str(N)] = {
            "baseline_mean": float(np.mean(base)) if base else None,
            "baseline_std": float(np.std(base)) if base else None,
            "baseline_n": len(base),
            "dense_mean": float(np.mean(dense)) if dense else None,
            "dense_std": float(np.std(dense)) if dense else None,
            "dense_n": len(dense),
            "dense_p_vs_baseline": paired_test(base, dense),
            "dense_cohens_d": cohens_d(base, dense),
            "dense_ci95": bootstrap_ci(base, dense),
            "topk_mean": float(np.mean(topk)) if topk else None,
            "topk_std": float(np.std(topk)) if topk else None,
            "topk_n": len(topk),
            "topk_p_vs_baseline": paired_test(base, topk),
            "topk_cohens_d": cohens_d(base, topk),
            "topk_ci95": bootstrap_ci(base, topk),
        }
    return out


def ablation_table(N: int = 6) -> str:
    """Compare adaptive-k vs fixed-k variants on the same MAPPO+comm
    backbone at one $N$."""
    cells = {}
    runs = {
        "MAPPO (baseline)": f"mappo_mpe_n{N}_baseline",
        "Dense Attn-Comm": f"mappo_mpe_n{N}_dense",
        "Adaptive TopK (ours)": f"mappo_mpe_n{N}_topk",
        f"Fixed TopK ($k{{=}}2$)": f"mappo_mpe_n{N}_topk_fixed_k2",
        f"Fixed TopK ($k{{=}}4$)": f"mappo_mpe_n{N}_topk_fixed_k4",
    }
    body_lines = []
    for name, run in runs.items():
        vals = collect_eval(run)
        body_lines.append(rf"{name} & {fmt_mean_std(vals) if vals else '--'} & {len(vals)} \\")
    body = "\n".join(body_lines)
    return (r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lcc}
\toprule
Variant & Final eval mean $\pm$ std & $n$ seeds \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{Ablation table at $N{=}""" + str(N) + r"""$ on MPE
\texttt{simple\_spread}. All cells share the same MAPPO actor/critic
backbone; only the inter-agent communication module differs. Adaptive TopK
learns $k_i$ per agent per step; Fixed TopK uses a static $k$.}
\label{tab:ablation}
\end{table}""")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-list", nargs="+", type=int, default=[3, 6, 12])
    p.add_argument("--ablation-n", type=int, default=6)
    args = p.parse_args()
    tex = headline_table(args.n_list)
    out = TABLES / "table_headline.tex"
    out.write_text(tex)
    print(f"wrote {out}")
    abl = ablation_table(args.ablation_n)
    out2 = TABLES / "table_ablation.tex"
    out2.write_text(abl)
    print(f"wrote {out2}")
    # Detailed stats JSON for the prose paragraphs.
    stats_path = TABLES / "headline_stats.json"
    stats_path.write_text(json.dumps(stats_json(args.n_list), indent=2))
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
