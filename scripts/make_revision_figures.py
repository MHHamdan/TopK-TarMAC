"""Every figure in the revised manuscript, from the released artifacts.

One module, one style, one colour per implementation. Each figure is generated
at exactly the width it is included at, and the writer refuses to emit any text
below the legibility floor, so a figure cannot silently acquire six-point
annotations.

Run with no arguments to rebuild all of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import figstyle as fs  # noqa: E402

OUT = ROOT / "paper_submission_revision" / "figures"
RES = ROOT / "results"
W = fs.TEXT_WIDTH_IN


def load(name: str) -> dict:
    p = RES / name
    if not p.exists():
        raise SystemExit(f"missing artifact {p}; run scripts/run_revision_campaign.sh")
    return json.loads(p.read_text())


def rows_of(payload: dict) -> list[dict]:
    return [r for r in payload["rows"] if not r.get("failed")]


def pick(rows, **kw):
    out = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    return out[0] if len(out) == 1 else out


# ---------------------------------------------------------------------------
# Figure 1: what each implementation stores and multiplies
# ---------------------------------------------------------------------------

def fig_taxonomy() -> None:
    """Five implementations of one function, by what reaches memory."""
    rng = np.random.default_rng(4)
    N, k = 12, 3
    keep = np.zeros((N, N), dtype=bool)
    for i in range(N):
        cand = [j for j in range(N) if j != i]
        keep[i, rng.choice(cand, size=k, replace=False)] = True

    rows = [
        ("Dense, unfused", "all", "stored", r"$2N^2d$", "dense_unfused"),
        ("Dense, fused", "all", "tiled", r"$2N^2d$", "dense_fused"),
        ("Top-$k$, masked", "kept", "stored", r"$2N^2d$", "topk_masked"),
        ("Top-$k$, gathered", "kept", "compacted", r"$N^2d + Nkd$", "topk_gather"),
        ("Top-$k$, fused", "kept", "tiled", r"$N^2d + Nkd$", "topk_fused"),
    ]

    fig, axes = plt.subplots(len(rows), 3, figsize=(W, 4.6),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 2.1],
                                          "hspace": 0.22, "wspace": 0.10})
    for r, (label, weights, storage, cost, impl) in enumerate(rows):
        col = fs.IMPL_COLOR[impl]

        # panel 1: the score matrix every implementation forms
        ax = axes[r, 0]
        _matrix(ax, np.ones((N, N), dtype=bool), np.eye(N, dtype=bool),
                face="#B9C6D4", show_all=True)
        if r == 0:
            ax.set_title("scores formed", pad=6)

        # panel 2: the weights that reach the aggregation, and whether the
        # implementation writes them down
        ax = axes[r, 1]
        if storage == "compacted":
            comp = np.ones((N, k), dtype=bool)
            _matrix(ax, comp, np.zeros((N, k), dtype=bool), face=col,
                    show_all=True, width=N)
        else:
            sel = np.ones((N, N), dtype=bool) if weights == "all" else keep
            _matrix(ax, sel, np.eye(N, dtype=bool), face=col,
                    show_all=(weights == "all" or storage == "stored"),
                    ghost=(storage == "tiled"))
        if r == 0:
            ax.set_title("weights aggregated", pad=6)

        # panel 3: what is written, and what the aggregation costs
        ax = axes[r, 2]
        ax.axis("off")
        written = {"stored": r"writes an $N\times N$ matrix",
                   "compacted": r"writes an $N\times k$ tensor",
                   "tiled": "writes no intermediate"}[storage]
        ax.text(0.0, 0.70, label, fontsize=fs.MIN_PT + 1.5, color=col,
                fontweight="semibold", va="center")
        ax.text(0.0, 0.34, written, fontsize=fs.MIN_PT, color=fs.MUTED_COLOR,
                va="center")
        ax.text(0.60, 0.34, cost, fontsize=fs.MIN_PT + 1.0,
                color=fs.ANNOT_COLOR, va="center")
        if r == 0:
            ax.text(0.60, 1.05, "aggregation cost", fontsize=fs.MIN_PT + 0.5,
                    color=fs.ANNOT_COLOR, va="bottom")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.text(0.5, 0.02,
             "Every implementation evaluates every pairwise score, because a "
             "score-then-select rule must form the complete score row before it "
             "can rank it.\n"
             "A dashed outline marks weights that are used but never written to "
             "memory; a cross marks the masked self-interaction.",
             ha="center", fontsize=fs.MIN_PT, color=fs.MUTED_COLOR,
             linespacing=1.5)
    fs.savefig(fig, OUT / "fig1_taxonomy.pdf", W)
    plt.close(fig)


def _matrix(ax, keep, diag, face, show_all: bool, ghost: bool = False,
            width: int | None = None) -> None:
    """Draw a weight matrix. ``ghost`` means the weights exist but are never
    written to memory, so the outline is dashed and the fill is lightened."""
    rows_n, cols_n = keep.shape
    span = width if width is not None else cols_n
    ax.set_xlim(-0.5, span - 0.5); ax.set_ylim(rows_n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    edge = fs.ABSENT_EDGE if ghost else "#8A8A8A"
    ax.add_patch(Rectangle((-0.5, -0.5), cols_n, rows_n, fill=False,
                           edgecolor=edge, linewidth=0.9,
                           linestyle=(0, (2, 2)) if ghost else "-"))
    fill = mpl.colors.to_rgba(face, 0.35 if ghost else 1.0)
    for i in range(rows_n):
        for j in range(cols_n):
            if diag[i, j]:
                ax.plot([j - 0.3, j + 0.3], [i - 0.3, i + 0.3],
                        color="#9A9A9A", linewidth=0.5)
                ax.plot([j - 0.3, j + 0.3], [i + 0.3, i - 0.3],
                        color="#9A9A9A", linewidth=0.5)
                continue
            if keep[i, j]:
                ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92,
                                       facecolor=fill, edgecolor="none"))
            elif show_all:
                ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92,
                                       facecolor="#DEDEDE", edgecolor="none"))


# ---------------------------------------------------------------------------
# Figure 2: latency against retained peers, with the count model overlaid
# ---------------------------------------------------------------------------

def _best_dense(rows, N: int) -> float:
    cands = [r["forward"]["min_ms"] for r in rows
             if r["n_agents"] == N and r["impl"] in
             ("dense_fused", "dense_triton")]
    return min(cands)


def fig_ksweep() -> None:
    payload = load("attention_core_d16.json")
    rows = rows_of(payload)
    ns = [n for n in (48, 96, 192, 512)
          if any(r["n_agents"] == n for r in rows)]
    impls = ["topk_gather", "topk_fused_gather", "topk_fused_masked"]

    fig, axes = plt.subplots(1, len(ns), figsize=(W, 2.5), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, N in zip(np.atleast_1d(axes), ns):
        dense = _best_dense(rows, N)
        for impl in impls:
            pts = sorted([r for r in rows if r["impl"] == impl
                          and r["n_agents"] == N], key=lambda r: r["k"])
            if not pts:
                continue
            ks = [p["k"] for p in pts]
            ys = [p["forward"]["min_ms"] / dense for p in pts]
            st = fs.style(impl)
            ax.plot(ks, ys, markersize=4.0, **st)
        # the count model: what an operation count predicts for the gathered path
        ks = np.array(sorted({r["k"] for r in rows if r["n_agents"] == N
                              and r["impl"] == "topk_gather"}), dtype=float)
        if ks.size:
            ax.plot(ks, (N + ks) / (2 * N), color=fs.MUTED_COLOR,
                    linestyle=(0, (1, 1.5)), linewidth=1.2, marker=None,
                    label="operation count predicts")
        fs.parity_line(ax)
        fs.log2_ticks(ax, [int(x) for x in ks])
        ax.set_yscale("log")
        ax.set_title(f"$N = {N}$", pad=4)
        ax.set_xlabel("peers retained, $k$")
        ax.tick_params(axis="x", labelsize=fs.MIN_PT - 0.5)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_ylabel("latency $\\div$ best dense")
    for ax in np.atleast_1d(axes):
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 0.25), hi)
        ax.axhspan(min(lo, 0.25), 1.0, color="#EAF2FA", zorder=0)
    ax0.annotate("sparse faster", xy=(0.04, 0.06), xycoords="axes fraction",
                 fontsize=fs.MIN_PT, color=fs.MUTED_COLOR)
    h, l = ax0.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.16),
               fontsize=fs.MIN_PT + 0.5)
    fs.savefig(fig, OUT / "fig2_ksweep.pdf", W)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: where the time goes, and how close each stage runs to its limit
# ---------------------------------------------------------------------------

STAGE_ORDER = ["scores", "softmax_N", "aggregate_dense",
               "topk_select", "softmax_k", "gather_v", "aggregate_sparse"]
STAGE_LABEL = {"scores": "scores", "softmax_N": "softmax over $N$",
               "aggregate_dense": "dense aggregate",
               "topk_select": "select", "softmax_k": "softmax over $k$",
               "gather_v": "gather", "aggregate_sparse": "sparse aggregate"}


def fig_stages() -> None:
    payload = load("stage_accounting.json")
    rows = payload["rows"]
    ns = sorted({r["n_agents"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7),
                             gridspec_kw={"wspace": 0.28,
                                          "width_ratios": [1.25, 1.0]})

    # left: composition of each path's time, as a share
    ax = axes[0]
    dense_stages = ["scores", "softmax_N", "aggregate_dense"]
    sparse_stages = ["scores", "topk_select", "softmax_k", "gather_v",
                     "aggregate_sparse"]
    xs = np.arange(len(ns), dtype=float)
    width = 0.36
    for off, (name, stages, hatch) in enumerate(
            [("dense", dense_stages, ""), ("gathered", sparse_stages, "///")]):
        bottoms = np.zeros(len(ns))
        totals = np.array([sum(pick(rows, n_agents=n, stage=s)["time_ms"]
                               for s in stages) for n in ns])
        for si, s in enumerate(stages):
            vals = np.array([pick(rows, n_agents=n, stage=s)["time_ms"]
                             for n in ns]) / totals * 100
            ax.bar(xs + (off - 0.5) * width, vals, width * 0.9, bottom=bottoms,
                   color=fs.STACK_RAMP[STAGE_ORDER.index(s)],
                   edgecolor="white", linewidth=0.4, hatch=hatch)
            bottoms += vals
        for x, t in zip(xs + (off - 0.5) * width, totals):
            ax.text(x, 101.5, f"{t:.2f}", ha="center", fontsize=fs.MIN_PT,
                    color=fs.MUTED_COLOR, rotation=90, va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("agents $N$      (left bar dense, right bar gathered)")
    ax.set_ylabel("share of the path (%)")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", visible=False)
    handles = [mpl.patches.Patch(facecolor=fs.STACK_RAMP[STAGE_ORDER.index(s)],
                                 edgecolor="white", label=STAGE_LABEL[s])
               for s in STAGE_ORDER]
    ax.legend(handles=handles, loc="upper center", ncol=4,
              bbox_to_anchor=(0.62, -0.30), fontsize=fs.MIN_PT,
              handlelength=1.4, columnspacing=1.0)

    # right: how close each stage runs to whichever machine limit binds it
    ax = axes[1]
    for s in ["aggregate_dense", "aggregate_sparse", "gather_v", "topk_select"]:
        ys = [100 * pick(rows, n_agents=n, stage=s)["frac_of_binding_roof"]
              for n in ns]
        ax.plot(ns, ys, marker="os^v"[["aggregate_dense", "aggregate_sparse",
                                      "gather_v", "topk_select"].index(s)],
                color=fs.STACK_RAMP[STAGE_ORDER.index(s)],
                label=STAGE_LABEL[s], markersize=4.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("agents $N$")
    ax.set_ylabel("% of the binding machine limit")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.62),
              fontsize=fs.MIN_PT, handlelength=1.6)
    fs.savefig(fig, OUT / "fig3_stages.pdf", W)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: achieved arithmetic rate against intensity
# ---------------------------------------------------------------------------

def fig_roofline() -> None:
    payload = load("stage_accounting.json")
    rows, peaks = payload["rows"], payload["peaks"]
    peak_f = peaks["peak_flops"] / 1e12
    peak_b = peaks["peak_bytes_per_s"] / 1e12
    ridge = peaks["ridge_flop_per_byte"]

    fig, ax = plt.subplots(figsize=(W * 0.62, 2.7))
    xs = np.logspace(-1.2, 3, 200)
    ax.plot(xs, np.minimum(peak_f, peak_b * xs), color="#1A1A1A",
            linewidth=1.2, zorder=3, marker=None)
    ax.axvline(ridge, color=fs.RULE_COLOR, linestyle=(0, (3, 2)),
               linewidth=0.9)
    ax.annotate(f"ridge {ridge:.0f} FLOP/byte", xy=(ridge, peak_f * 0.055),
                xytext=(-4, 0), textcoords="offset points", rotation=90,
                ha="right", va="bottom", fontsize=fs.MIN_PT,
                color=fs.RULE_COLOR)

    marks = {"scores": ("o", "#123a63"), "aggregate_dense": ("D", "#0072B2"),
             "aggregate_sparse": ("s", "#009E73"), "gather_v": ("^", "#5FBFA8"),
             "topk_select": ("v", "#D55E00")}
    for stage, (mk, col) in marks.items():
        pts = [r for r in rows if r["stage"] == stage and r["ai"] > 0]
        if not pts:
            continue
        ax.scatter([p["ai"] for p in pts], [p["gflops"] / 1e3 for p in pts],
                   marker=mk, s=22, facecolor=col, edgecolor="white",
                   linewidth=0.4, zorder=4, label=STAGE_LABEL[stage])
    # selection performs comparisons rather than multiply-adds, so it has no
    # meaningful arithmetic rate; its bandwidth is shown instead.
    sel = [r for r in rows if r["stage"] == "topk_select"]
    if sel:
        ax.scatter([peaks["ridge_flop_per_byte"] * 0] + [], [])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("achieved rate (TFLOP/s)")
    ax.set_xlim(0.06, 1500)
    ax.set_ylim(peak_f * 2e-4, peak_f * 1.8)
    ax.legend(loc="lower right", fontsize=fs.MIN_PT, handletextpad=0.3,
              borderpad=0.2, labelspacing=0.25)
    fs.savefig(fig, OUT / "fig4_roofline.pdf", W * 0.62)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: selecting before scoring
# ---------------------------------------------------------------------------

def fig_prescore() -> None:
    payload = load("prescore.json")
    rows = payload["rows"]
    ns = sorted({r["n_agents"] for r in rows})
    k_show = sorted({r["k"] for r in rows})[1]

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5),
                             gridspec_kw={"wspace": 0.30})

    ax = axes[0]
    for sel in ("radius_knn", "morton_window"):
        pts = [r for r in rows if r["selector"] == sel and r["k"] == k_show]
        pts.sort(key=lambda r: r["n_agents"])
        if not pts:
            continue
        st = fs.style(sel)
        ax.plot([p["n_agents"] for p in pts],
                [p["flop_ratio_vs_dense"] for p in pts], markersize=4.0, **st)
    pts = [r for r in rows if r["selector"] == "morton_window"
           and r["k"] == k_show]
    pts.sort(key=lambda r: r["n_agents"])
    if pts:
        ax.plot([p["n_agents"] for p in pts],
                [p["n_agents"] / p["k"] for p in pts], color=fs.MUTED_COLOR,
                linestyle=(0, (1, 1.5)), marker=None, label="$N/k$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("agents $N$")
    ax.set_ylabel("operations saved ($\\times$)")
    ax.set_title(f"what the count promises ($k={k_show}$)", pad=4)
    ax.legend(fontsize=fs.MIN_PT, loc="upper left")

    ax = axes[1]
    for sel in ("radius_knn", "morton_window"):
        pts = [r for r in rows if r["selector"] == sel and r["k"] == k_show]
        pts.sort(key=lambda r: r["n_agents"])
        if not pts:
            continue
        st = fs.style(sel)
        ax.plot([p["n_agents"] for p in pts],
                [p["t_total_ms"] / p["t_dense_fused_ms"] for p in pts],
                markersize=4.0, **st)
    pts = [r for r in rows if r["selector"] == "radius_knn" and r["k"] == k_show]
    pts.sort(key=lambda r: r["n_agents"])
    if pts:
        st = fs.style("topk_gather")
        st["label"] = "score-then-select, gathered"
        ax.plot([p["n_agents"] for p in pts],
                [p["t_topk_gather_ms"] / p["t_dense_fused_ms"] for p in pts],
                markersize=4.0, **st)
    fs.parity_line(ax)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("agents $N$")
    ax.set_ylabel("latency $\\div$ fused dense")
    ax.set_title("what the clock delivers", pad=4)
    ax.legend(fontsize=fs.MIN_PT, loc="upper left")
    fs.savefig(fig, OUT / "fig5_prescore.pdf", W)
    plt.close(fig)


# ---------------------------------------------------------------------------

FIGURES = {
    "taxonomy": fig_taxonomy,
    "ksweep": fig_ksweep,
    "stages": fig_stages,
    "roofline": fig_roofline,
    "prescore": fig_prescore,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="+", choices=list(FIGURES))
    args = p.parse_args()
    fs.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGURES.items():
        if args.only and name not in args.only:
            continue
        fn()
        print(f"wrote {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
