"""Every table in the revised manuscript, generated from the released artifacts.

A number that appears in the manuscript and also in a JSON file is a number that
can disagree with itself. Generating the tables removes that possibility for the
tabular numbers; ``scripts/check_paper_numbers.py`` covers the ones that appear
in prose.

Writes .tex fragments into paper_submission_revision/tables/, which main.tex
inputs directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
OUT = ROOT / "paper_submission_revision" / "tables"

DENSE = ("dense_unfused", "dense_fused", "dense_triton")


def load(name: str) -> dict:
    p = RES / name
    if not p.exists():
        raise SystemExit(f"missing {p}; run scripts/run_revision_campaign.sh")
    return json.loads(p.read_text())


def live(payload: dict) -> list[dict]:
    return [r for r in payload["rows"] if not r.get("failed")]


def at(rows, impl, N, k=None):
    c = [r for r in rows if r["impl"] == impl and r["n_agents"] == N
         and (k is None or r["k"] == k)]
    return c[0] if c else None


def ms(r, field="forward"):
    return r[field]["min_ms"] if r and r.get(field) else None


def fmt(x, p=3):
    return "---" if x is None else f"{x:.{p}f}"


# ---------------------------------------------------------------------------

def table_headline() -> str:
    """Forward latency of every implementation at a fixed sparsity."""
    pay = load("attention_core_d16.json")
    rows = live(pay)
    ns = sorted({r["n_agents"] for r in rows})
    body = []
    for N in ns:
        k = max(1, N // 4)
        ks = sorted({r["k"] for r in rows if r["impl"] == "topk_gather"
                     and r["n_agents"] == N})
        k = min(ks, key=lambda x: abs(x - k)) if ks else 1
        du, df, dt = (ms(at(rows, i, N)) for i in DENSE)
        gg = ms(at(rows, "topk_gather", N, k))
        fg = ms(at(rows, "topk_fused_gather", N, k))
        fm = ms(at(rows, "topk_fused_masked", N, k))
        bd = min(x for x in (df, dt) if x is not None)
        bs = min(x for x in (gg, fg, fm) if x is not None)
        body.append(
            f"{N} & {k} & {fmt(du)} & {fmt(df)} & {fmt(dt)} & "
            f"{fmt(gg)} & {fmt(fg)} & {fmt(fm)} & {bs / bd:.2f} \\\\")
    return r"""\begin{tabular}{rrrrrrrrr}
\toprule
& & \multicolumn{3}{c}{\textbf{dense}} & \multicolumn{3}{c}{\textbf{top-$k$}} & \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
$N$ & $k$ & unfused & fused & ours & gathered & fused & fused & best sparse \\
& & & (library) & & (composed) & gather & mask & $\div$ best dense \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}"""


def table_crossover() -> str:
    """The k at which the best sparse implementation stops being faster."""
    pay = load("attention_core_d16.json")
    rows = live(pay)
    ns = sorted({r["n_agents"] for r in rows})
    body = []
    for N in ns:
        bd = min(x for x in (ms(at(rows, "dense_fused", N)),
                             ms(at(rows, "dense_triton", N))) if x is not None)
        ks = sorted({r["k"] for r in rows if r["impl"] == "topk_gather"
                     and r["n_agents"] == N})
        ratios = []
        for k in ks:
            cand = [ms(at(rows, i, N, k)) for i in
                    ("topk_gather", "topk_fused_gather", "topk_fused_masked")]
            cand = [c for c in cand if c is not None]
            if cand:
                ratios.append((k, min(cand) / bd))
        below = [k for k, r in ratios if r < 1.0]
        best_k, best_r = min(ratios, key=lambda t: t[1])
        kstar = (f"$\\le {max(below)}$" if below else "none")
        frac = (f"{100 * max(below) / N:.1f}\\%" if below else "---")
        worst = max(ratios, key=lambda t: t[0])
        body.append(
            f"{N} & {kstar} & {frac} & {best_r:.2f} & {best_k} & "
            f"{worst[1]:.1f} & {worst[0]} \\\\")
    return r"""\begin{tabular}{rccrrrr}
\toprule
& \multicolumn{2}{c}{\textbf{sparsity that pays}} &
  \multicolumn{2}{c}{\textbf{best ratio}} &
  \multicolumn{2}{c}{\textbf{at the widest $k$}} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
$N$ & $k$ & of peers & ratio & at $k$ & ratio & at $k$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}"""


def table_backward() -> str:
    """Forward, backward and the two together, for the differentiable paths."""
    pay = load("attention_core_d16.json")
    rows = live(pay)
    ns = [n for n in sorted({r["n_agents"] for r in rows}) if n >= 48]
    body = []
    for N in ns:
        k = min(sorted({r["k"] for r in rows if r["impl"] == "topk_gather"
                        and r["n_agents"] == N}),
                key=lambda x: abs(x - max(1, N // 4)))
        line = [str(N), str(k)]
        ref = None
        for impl in ("dense_unfused", "dense_fused", "topk_masked",
                     "topk_gather"):
            r = at(rows, impl, N, k if impl.startswith("topk") else None)
            f, t = ms(r), ms(r, "backward_total")
            if f is None or t is None:
                line += ["---", "---", "---"]
                continue
            if impl == "dense_fused":
                ref = t
            line += [f"{f:.3f}", f"{t - f:.3f}", f"{t:.3f}"]
        line.append(f"{ms(at(rows, 'topk_gather', N, k), 'backward_total') / ref:.1f}"
                    if ref else "---")
        body.append(" & ".join(line) + r" \\")
    return r"""\begin{tabular}{rr rrr rrr rrr rrr r}
\toprule
& & \multicolumn{3}{c}{dense, unfused} & \multicolumn{3}{c}{dense, fused}
& \multicolumn{3}{c}{top-$k$ masked} & \multicolumn{3}{c}{top-$k$ gathered} & \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}
$N$ & $k$ & f & b & f+b & f & b & f+b & f & b & f+b & f & b & f+b
& \begin{tabular}[b]{@{}c@{}}gathered f+b\\$\div$ fused f+b\end{tabular} \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}"""


def table_stages() -> str:
    """Per-stage arithmetic, traffic, intensity and attainment."""
    pay = load("stage_accounting.json")
    rows = pay["rows"]
    ns = sorted({r["n_agents"] for r in rows})
    N = ns[-1]
    order = ["scores", "softmax_N", "aggregate_dense", "topk_select",
             "softmax_k", "gather_v", "aggregate_sparse"]
    label = {"scores": "score product", "softmax_N": "softmax over $N$",
             "aggregate_dense": "dense aggregation",
             "topk_select": "top-$k$ selection", "softmax_k": "softmax over $k$",
             "gather_v": "gather of $V$",
             "aggregate_sparse": "sparse aggregation"}
    body = []
    for s in order:
        r = [x for x in rows if x["stage"] == s and x["n_agents"] == N][0]
        ai = "---" if r["flops"] == 0 else f"{r['ai']:.2f}"
        gf = "---" if r["flops"] == 0 else f"{r['gflops'] / 1e3:.2f}"
        body.append(
            f"{label[s]} & {r['flops'] / 1e9:.2f} & {r['bytes'] / 1e9:.2f} & "
            f"{ai} & {r['time_ms']:.3f} & {gf} & "
            f"{r['gbytes_per_s']:.0f} & {100 * r['frac_of_binding_roof']:.0f} \\\\")
    peaks = pay["peaks"]
    cap = (f"Peak measured on the same device: "
           f"{peaks['peak_flops'] / 1e12:.1f}~TFLOP/s and "
           f"{peaks['peak_bytes_per_s'] / 1e9:.0f}~GB/s, a ridge point of "
           f"{peaks['ridge_flop_per_byte']:.0f}~FLOP/byte.")
    return (r"""\begin{tabular}{lrrrrrrr}
\toprule
\textbf{stage} & GFLOP & GB & FLOP/byte & ms & TFLOP/s & GB/s & \% of limit \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}""", cap, N)


def table_prescore() -> str:
    """What selecting before scoring saves, and what it delivers."""
    pay = load("prescore.json")
    rows = pay["rows"]
    ns = sorted({r["n_agents"] for r in rows})
    ks = sorted({r["k"] for r in rows})
    k = ks[1] if len(ks) > 1 else ks[0]
    body = []
    for N in ns:
        for sel, nm in (("radius_knn", "nearest neighbours"),
                        ("morton_window", "Z-order window")):
            r = [x for x in rows if x["selector"] == sel and x["n_agents"] == N
                 and x["k"] == k]
            if not r:
                continue
            r = r[0]
            body.append(
                f"{N} & {nm} & {r['flop_ratio_vs_dense']:.1f} & "
                f"{r['t_selector_ms']:.3f} & {r['t_attention_ms']:.3f} & "
                f"{r['t_total_ms']:.3f} & {r['t_dense_fused_ms']:.3f} & "
                f"{r['t_total_ms'] / r['t_dense_fused_ms']:.2f} \\\\")
    return r"""\begin{tabular}{rlrrrrrr}
\toprule
& & \textbf{operations} & \multicolumn{3}{c}{\textbf{measured (ms)}} & &  \\
\cmidrule(lr){4-6}
$N$ & selector & saved ($\times$) & select & attend & total & dense & ratio \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}""", k


def table_training_step() -> str:
    """Where a training iteration spends its time."""
    pay = load("training_step_timing.json")
    rows = pay["rows"]
    ns = sorted({r["n_agents"] for r in rows})
    order = ["environment", "policy_forward", "advantage", "update_forward",
             "update_backward", "optimiser"]
    label = {"environment": "environment", "policy_forward": "acting",
             "advantage": "advantage", "update_forward": "update fwd",
             "update_backward": "update bwd", "optimiser": "optimiser"}
    body = []
    for N in ns:
        base = [r for r in rows if r["n_agents"] == N
                and r["arm"] == "dense_sdpa"]
        base = base[0] if base else None
        for r in [x for x in rows if x["n_agents"] == N]:
            shares = " & ".join(
                f"{100 * r['phase_share'].get(s, 0):.0f}" for s in order)
            rel = (f"{r['iteration_ms'] / base['iteration_ms']:.3f}"
                   if base else "---")
            body.append(f"{N} & \\texttt{{{r['arm'].replace('_', chr(92) + '_')}}}"
                        f" & {shares} & {r['iteration_ms'] / 1e3:.2f} & {rel} & "
                        f"{r['iteration_ms_spread_pct']:.1f} \\\\")
    head = " & ".join(label[s] for s in order)
    return r"""\begin{tabular}{ll rrrrrr rrr}
\toprule
& & \multicolumn{6}{c}{\textbf{share of the iteration (\%)}} & & & \\
\cmidrule(lr){3-8}
$N$ & communication & """ + head + r""" & s/iter & $\div$ dense & spread \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    def put(name: str, body: str) -> None:
        (OUT / f"{name}.tex").write_text(body + "\n")
        written.append(name)

    put("headline", table_headline())
    put("crossover", table_crossover())
    put("backward", table_backward())
    stages, cap, N = table_stages()
    put("stages", stages)
    put("stages_caption", cap)
    put("stages_n", str(N))
    pre, k = table_prescore()
    put("prescore", pre)
    put("prescore_k", str(k))
    try:
        put("training_step", table_training_step())
    except SystemExit as exc:
        print(f"  (skipped training_step: {exc})")
    print("wrote:", ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
