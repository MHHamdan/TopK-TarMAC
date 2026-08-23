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


#: Every dense implementation a ratio may be quoted against: the library path,
#: our own kernel at its default tiling, and our own kernel at the tiling the
#: search found. Taking the minimum is what makes the ratios in this manuscript
#: statements about sparsity rather than about how hard we tried on each side.
DENSE_BASELINES = ("dense_fused", "dense_triton", "dense_tuned")


def best_dense(rows, N, k=None):
    c = [ms(at(rows, i, N, k)) for i in DENSE_BASELINES]
    c = [x for x in c if x is not None]
    return min(c) if c else None


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
    # Below 192 agents the cost of every implementation is governed by how many
    # kernels it launches rather than by arithmetic or bandwidth, and the
    # repeated sweep shows those configurations do not reproduce: at 96 agents
    # the two runs disagree by a median of 61% in the ratio, against 0.3% at
    # 192. A crossover read off them would be a fact about device contention.
    ns = [n for n in sorted({r["n_agents"] for r in rows}) if n >= 192]
    body = []
    for N in ns:
        bd = best_dense(rows, N)
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


def table_autotune() -> str:
    """Both kernels searched over the same tiling space, with what each attains."""
    pay = load("autotune.json")
    rows = pay["rows"]
    peak = pay["peak_flops"]
    ds = sorted({r["head_dim"] for r in rows})
    body = []
    for d in ds:
        ns = sorted({r["n_agents"] for r in rows if r["head_dim"] == d})
        for N in ns:
            dn = [r for r in rows if r["kind"] == "dense"
                  and r["head_dim"] == d and r["n_agents"] == N]
            if not dn:
                continue
            dn = dn[0]
            B = dn["batch"]
            d_ms = dn["best"]["ms"]
            d_eff = 100 * (4 * B * N * N * d) / (d_ms * 1e-3) / peak
            sp = sorted([r for r in rows if r["kind"] == "sparse"
                         and r["head_dim"] == d and r["n_agents"] == N],
                        key=lambda r: r["k"])
            if not sp:
                continue
            best = min(sp, key=lambda r: r["best"]["ms"] / d_ms)
            k = best["k"]
            s_ms = best["best"]["ms"]
            s_eff = 100 * (2 * B * N * N * d + 2 * B * N * k * d) / (s_ms * 1e-3) / peak
            wins = [r["k"] for r in sp if r["best"]["ms"] < d_ms]
            pays = ("$\\le " + str(max(wins)) + "$") if wins else "none"
            src = dn["best"]["name"].split("_")[0]
            body.append(
                f"{d} & {N} & {src} & {dn['n_ran']} & "
                f"{d_ms:.4f} & {d_eff:.0f} & {k} & {best['n_ran']} & "
                f"{s_ms:.4f} & {s_eff:.0f} & "
                f"{s_ms / d_ms:.2f} & {pays} " + r"\\")
    return r"""\begin{tabular}{rr lrrr rrrr r c}
\toprule
& & \multicolumn{4}{c}{\textbf{best dense}} &
    \multicolumn{4}{c}{\textbf{best sparse}} & & \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
$d$ & $N$ & from & cfgs & ms & \% peak & $k$ & cfgs & ms & \% peak & ratio
& $k$ that pays \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}"""


def table_backward() -> str:
    """Forward, backward and the two together, for the differentiable paths."""
    pay = load("attention_core_d16.json")
    rows = live(pay)
    # Same restriction as the crossover table: below 192 agents the forward and
    # the forward-plus-backward passes are timed in separate interleaved blocks
    # and the contention between them exceeds the backward's own cost, which
    # produces differences a backward pass cannot have.
    ns = [n for n in sorted({r["n_agents"] for r in rows}) if n >= 192]
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


def table_selection() -> str:
    """The three fused selection strategies, at the two agent counts that carry
    the crossover claim. Each column is the same aggregation reached a different
    way: k argmax rounds over the score row, a bit-pattern bisection that yields
    a threshold, and one bitonic sort of score-index pairs."""
    rows = live(load("attention_core_sort.json"))
    ns = [192, 512]
    ks = [1, 2, 4, 8, 16, 32]
    body = []
    for N in ns:
        bd = best_dense(rows, N)
        for k in ks:
            cells = []
            for impl in ("topk_fused_gather", "topk_fused_masked",
                         "topk_fused_sort"):
                t = ms(at(rows, impl, N, k))
                cells.append("--- & ---" if t is None
                             else f"{t:.3f} & {t / bd:.1f}")
            body.append(f"{N if k == ks[0] else ''} & {k} & "
                        + " & ".join(cells) + r" \\")
        if N != ns[-1]:
            body.append(r"\midrule")
    return r"""\begin{tabular}{rr rr rr rr}
\toprule
& & \multicolumn{2}{c}{\textbf{$k$ argmax rounds}}
& \multicolumn{2}{c}{\textbf{bisection}}
& \multicolumn{2}{c}{\textbf{one sort}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
$N$ & $k$ & ms & $\div$dense & ms & $\div$dense & ms & $\div$dense \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}"""


def table_prescore() -> str:
    """What selecting before scoring saves, and what it delivers."""
    pay = load("prescore.json")
    rows = pay["rows"]
    ns = sorted({r["n_agents"] for r in rows})
    # The sparsest setting measured, which is where the escape is strongest and
    # where the number the abstract quotes comes from. The manuscript gives the
    # next setting up in prose, so both the best and a more conservative case
    # are on the page.
    ks = sorted({r["k"] for r in rows})
    k = ks[0]
    body = []
    for N in ns:
        for sel, nm in (("radius_knn", "nearest neighbours"),
                        ("morton_window", "Z-order window")):
            r = [x for x in rows if x["selector"] == sel and x["n_agents"] == N
                 and x["k"] == k]
            if not r:
                continue
            r = r[0]
            d = r.get("t_dense_best_ms", r["t_dense_fused_ms"])
            af = r.get("t_attention_fused_ms")
            tf = r.get("t_total_fused_ms")
            body.append(
                f"{N} & {nm} & {r['flop_ratio_vs_dense']:.0f} & "
                f"{r['t_selector_ms']:.3f} & {r['t_attention_ms']:.3f} & "
                f"{af:.3f} & {d:.3f} & "
                f"{r['t_attention_ms'] / d:.2f} & {af / d:.2f} & "
                f"{tf / d:.2f} \\\\")
    return r"""\begin{tabular}{rlr rrr r rrr}
\toprule
& & \textbf{ops} & \multicolumn{3}{c}{\textbf{measured (ms)}} & \textbf{best}
& \multicolumn{3}{c}{\textbf{$\div$ best dense}} \\
\cmidrule(lr){4-6}\cmidrule(lr){8-10}
& & & & \multicolumn{2}{c}{attend} & & \multicolumn{2}{c}{attend} & total \\
\cmidrule(lr){5-6}\cmidrule(lr){8-9}
$N$ & selector & saved & select & composed & fused & dense & composed & fused
& fused \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}""", k


def table_audit() -> str:
    """The reporting audit, from the released sheet rather than from prose."""
    pay = load("literature_latency_audit.json")
    # Two groups, matching the split the manuscript reports, rather than the
    # four sub-fields the sheet records.
    marl = ("MARL communication", "MARL attention")
    groups: dict[str, list] = {
        "Multi-agent communication and attention": [],
        "Efficient attention and sparse mixture-of-experts": []}
    for r in pay["papers"]:
        key = ("Multi-agent communication and attention" if r["field"] in marl
               else "Efficient attention and sparse mixture-of-experts")
        groups[key].append(r)
    label = {"qualitative": "qualitative", "counts": "counts",
             "asymptotic": "asymptotic", "flops": "FLOPs", "none": "---",
             "None": "---"}
    body = []
    for field in groups:
        body.append(r"\multicolumn{4}{l}{\emph{" + field + r"}} \\")
        for r in sorted(groups[field], key=lambda x: x["bibkey"]):
            ct = label.get(r.get("claim_type", "none"),
                           r.get("claim_type", "---"))
            yes = r.get("reports_measured_latency")
            mark = (r"\textbf{yes}" if yes else
                    ("n/a" if ct == "---" else "no"))
            claim = r.get("efficiency_claim", "")
            claim = "none explicit" if ct == "---" else claim.split(":")[0]
            claim = claim.split(".")[0].strip().strip("'")
            if len(claim) > 46:                      # truncate on a word break
                claim = claim[:46].rsplit(" ", 1)[0] + r"\,\ldots"
            # The sheet quotes papers verbatim, so it contains characters that
            # are markup in LaTeX and mathematics outside maths mode.
            for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                         ("#", r"\#"), ("^2", "$^2$"), ("^", "")):
                claim = claim.replace(a, b)
            body.append(f"\\citet{{{r['bibkey']}}} & {claim} & {ct} & {mark} "
                        + r"\\")
    return r"""\begin{tabular}{p{0.29\linewidth}p{0.36\linewidth}p{0.14\linewidth}l}
\toprule
\textbf{Paper} & \textbf{Efficiency claim} & \textbf{Type} &
\begin{tabular}[b]{@{}l@{}}\textbf{Measured}\\\textbf{time?}\end{tabular} \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}"""


def table_grid() -> str:
    """Best sparse over best dense across head width and batch, per agent count."""
    blocks = []
    for fname, axis, label in (("attention_core_dsweep.json", "head_dim",
                                "head width $d$"),
                               ("attention_core_batch.json", "batch",
                                "batch $B$")):
        try:
            pay = load(fname)
        except SystemExit:
            continue
        rows = live(pay)
        ns = sorted({r["n_agents"] for r in rows})
        vals = sorted({r[axis] for r in rows})
        body = []
        for v in vals:
            cells = []
            for N in ns:
                sub = [r for r in rows if r[axis] == v and r["n_agents"] == N]
                dense = [ms(r) for r in sub if r["impl"] in DENSE_BASELINES]
                sparse = [ms(r) for r in sub if r["impl"].startswith("topk")]
                dense = [x for x in dense if x is not None]
                sparse = [x for x in sparse if x is not None]
                cells.append(f"{min(sparse) / min(dense):.2f}"
                             if dense and sparse else "n/m")
            body.append(f"{v} & " + " & ".join(cells) + r" \\")
        cols = "r" * (len(ns) + 1)
        head = " & ".join(f"$N{{=}}{n}$" for n in ns)
        blocks.append(f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n"
                      f"{label} & {head} \\\\\n\\midrule\n"
                      + "\n".join(body)
                      + "\n\\bottomrule\n\\end{tabular}")
    if not blocks:
        return r"\emph{[grid pending]}"
    return "\n\n\\vspace{1em}\n\n".join(blocks)


def table_drift() -> str:
    """How far the headline sweep moves when it is repeated on the same device."""
    a = live(load("attention_core_d16.json"))
    b = live(load("attention_core_d16_repeat.json"))

    def index(rows):
        return {(r["impl"], r["n_agents"], r["k"]): ms(r) for r in rows}

    ia, ib = index(a), index(b)
    common = [k for k in ia if k in ib and ia[k] and ib[k]]
    ns = sorted({k[1] for k in common})
    body = []
    for N in ns:
        lat, rat = [], []
        keys = [k for k in common if k[1] == N]
        for key in keys:
            lat.append(abs(ib[key] - ia[key]) / ia[key])
        # Ratios are against the best dense implementation in each run, which is
        # the quantity every claim in the paper is stated in.
        def best_dense(idx):
            return min(idx[k] for k in idx if k[1] == N
                       and k[0] in DENSE_BASELINES)
        da, db = best_dense(ia), best_dense(ib)
        for key in keys:
            if key[0].startswith("topk"):
                ra, rb = ia[key] / da, ib[key] / db
                rat.append(abs(rb - ra) / ra)
        body.append(f"{N} & {len(keys)} & {100 * sorted(lat)[len(lat)//2]:.1f} "
                    f"& {100 * max(lat):.1f} & "
                    f"{100 * sorted(rat)[len(rat)//2]:.1f} & "
                    f"{100 * max(rat):.1f} \\\\")
    return r"""\begin{tabular}{rrrrrr}
\toprule
& & \multicolumn{2}{c}{\textbf{latency} $|\Delta|$ (\%)}
& \multicolumn{2}{c}{\textbf{ratio} $|\Delta|$ (\%)} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}
$N$ & configurations & median & max & median & max \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}"""


def table_training_step() -> str:
    """Where a training iteration spends its time."""
    pay = load("training_step_timing.json")
    rows = pay["rows"]
    ns = sorted({r["n_agents"] for r in rows})
    # The advantage computation is below 0.5% of every iteration and is folded
    # into the optimiser column rather than given one of its own.
    order = ["environment", "policy_forward", "update_forward",
             "update_backward", "optimiser"]
    label = {"environment": "environment", "policy_forward": "acting",
             "update_forward": "update fwd",
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
            name = {"no_comm": "none", "dense": "dense, unfused",
                    "dense_sdpa": "dense, library fused",
                    "topk_masked": "top-$k$, masked",
                    "topk_gather": "top-$k$, gathered"}.get(r["arm"], r["arm"])
            body.append(f"{N} & {name}"
                        f" & {shares} & {r['iteration_ms'] / 1e3:.2f} & {rel} & "
                        f"{r['iteration_ms_spread_pct']:.1f} \\\\")
    head = " & ".join(label[s] for s in order)
    return r"""\begin{tabular}{ll rrrrr rrr}
\toprule
& & \multicolumn{5}{c}{\textbf{share of the iteration (\%)}} & & & \\
\cmidrule(lr){3-7}
$N$ & communication & """ + head + r""" & s/iter & $\div$ fused & spread \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    def put(name: str, body: str) -> None:
        # Trailing "%" suppresses the space \input would otherwise insert, which
        # matters for the fragments that appear inline inside a caption.
        (OUT / f"{name}.tex").write_text(body + "%\n")
        written.append(name)

    # Values that appear inside captions are emitted as macros rather than as
    # \input, because hyperref expands a caption when it builds the bookmark and
    # an \input inside one breaks that expansion.
    macros: dict[str, str] = {}

    put("headline", table_headline())
    put("crossover", table_crossover())
    put("selection", table_selection())
    put("backward", table_backward())
    stages, cap, N = table_stages()
    put("stages", stages)
    macros["StagesCaption"] = cap
    macros["StagesN"] = str(N)
    pre, k = table_prescore()
    put("prescore", pre)
    macros["PrescoreK"] = str(k)
    for name, fn in (("training_step", table_training_step),
                     ("grid", table_grid),
                     ("drift", table_drift),
                     ("autotune", table_autotune),
                     ("audit", table_audit)):
        try:
            put(name, fn())
        except SystemExit as exc:
            print(f"  (skipped {name}: {exc})")
    (OUT / "macros.tex").write_text(
        "\n".join(f"\\newcommand{{\\{n}}}{{{v}}}"
                  for n, v in sorted(macros.items())) + "\n")
    written.append("macros")
    print("wrote:", ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
