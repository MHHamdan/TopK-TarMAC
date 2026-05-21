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
import pandas as pd
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


def headline_table(N_list: list[int]) -> str:
    rows = []
    for N in N_list:
        base = collect_eval(f"mappo_mpe_n{N}_baseline")
        dense = collect_eval(f"mappo_mpe_n{N}_dense")
        topk = collect_eval(f"mappo_mpe_n{N}_topk")
        p_dense = paired_test(base, dense)
        p_topk = paired_test(base, topk)

        def cell(vals: list[float], better_p: float = 1.0) -> str:
            if not vals:
                return "--"
            txt = fmt_mean_std(vals)
            if better_p < 0.05:
                txt = "\\textbf{" + txt + "}"
            return txt + f" ($n{{=}}{len(vals)}$)"

        rows.append((N,
                     cell(base),
                     cell(dense, better_p=p_dense if (np.mean(dense or [-1e9]) >
                                                       np.mean(base or [-1e9])) else 1.0),
                     cell(topk, better_p=p_topk if (np.mean(topk or [-1e9]) >
                                                      np.mean(base or [-1e9])) else 1.0),
                     p_dense, p_topk))

    head = (r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lccc cc}
\toprule
$N$ & MAPPO (baseline) & + Dense Attn-Comm & + Adaptive TopK (ours) &
$p_{\text{dense}}$ & $p_{\text{topk}}$\\
\midrule
""")
    body = "\n".join(
        rf"{N} & {b} & {d} & {t} & {pd_:.3f} & {pt:.3f} \\"
        for (N, b, d, t, pd_, pt) in rows
    )
    tail = r"""
\bottomrule
\end{tabular}
\caption{Headline result: greedy eval mean $\pm$ std across seeds, MPE
\texttt{simple\_spread} for $N \in \{N_LIST\}$ agents. Bold = significantly
better than baseline (Welch's $t$-test, $p<0.05$). $p$ columns report the
two-sided Welch p-value vs MAPPO.}
\label{tab:headline}
\end{table}"""
    tail = tail.replace("N_LIST", ", ".join(str(N) for N in N_list))
    return head + body + tail


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-list", nargs="+", type=int, default=[3, 6, 12])
    args = p.parse_args()
    tex = headline_table(args.n_list)
    out = TABLES / "table_headline.tex"
    out.write_text(tex)
    print(f"wrote {out}")
    print(tex)


if __name__ == "__main__":
    main()
