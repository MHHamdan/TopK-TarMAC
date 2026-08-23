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

    fig, axes = plt.subplots(len(rows), 3, figsize=(W, 3.6),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.35],
                                          "hspace": 0.30, "wspace": 0.06,
                                          "left": 0.02, "right": 0.99,
                                          "top": 0.90, "bottom": 0.10})
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
        ax.text(0.0, 0.72, label, fontsize=fs.MIN_PT + 1.5, color=col,
                fontweight="semibold", va="center")
        ax.text(0.0, 0.40, written, fontsize=fs.MIN_PT, color=fs.MUTED_COLOR,
                va="center")
        ax.text(0.0, 0.10, cost, fontsize=fs.MIN_PT + 1.0,
                color=fs.ANNOT_COLOR, va="center")
        if r == 0:
            ax.text(0.0, 1.12, "stored, and multiplied",
                    fontsize=fs.MIN_PT + 0.5, color=fs.ANNOT_COLOR, va="bottom")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.text(0.5, 0.01,
             "A dashed outline marks weights that are used but never written to "
             "memory; a cross marks the masked self-interaction.",
             ha="center", fontsize=fs.MIN_PT, color=fs.MUTED_COLOR)
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
             ("dense_fused", "dense_triton", "dense_tuned")]
    return min(cands)


def fig_ksweep() -> None:
    payload = load("attention_core_d16.json")
    rows = rows_of(payload)
    # The agent counts the paper reads claims off, plus the smallest one that
    # does not reproduce, kept for continuity and labelled as such.
    ns = [n for n in (96, 192, 384, 512)
          if any(r["n_agents"] == n for r in rows)]
    impls = ["topk_gather", "topk_fused_gather", "topk_fused_masked"]

    fig, axes = plt.subplots(1, len(ns), figsize=(W, 2.6), sharey=True,
                             gridspec_kw={"wspace": 0.10})
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
        # Thin the ticks so the widest two values do not collide.
        shown = [int(x) for x in ks if x in (1, 2, 4, 8, 16, 32, 64, 128)]
        fs.log2_ticks(ax, shown)
        ax.set_yscale("log")
        ax.set_title(f"$N = {N}$", pad=4)
        ax.set_xlabel("peers retained, $k$")
        ax.tick_params(axis="x", labelsize=fs.MIN_PT)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_ylabel("latency / best dense")
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
STAGE_LABEL = {"scores": "scores", "softmax_N": "softmax",
               "aggregate_dense": "aggregate",
               "topk_select": "select", "softmax_k": "softmax",
               "gather_v": "gather", "aggregate_sparse": "aggregate"}
#: Colour by the role a stage plays, not by its name, so that the dense and
#: sparse aggregations share a colour and the hatch distinguishes the path.
STAGE_RAMP = {"scores": 0, "topk_select": 1, "aggregate_dense": 2,
              "aggregate_sparse": 2, "gather_v": 3, "softmax_N": 4,
              "softmax_k": 4}


def _stage_color(s: str) -> str:
    return fs.STACK_RAMP[STAGE_RAMP[s]]


def fig_stages() -> None:
    payload = load("stage_accounting.json")
    rows = payload["rows"]
    ns = sorted({r["n_agents"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.8),
                             gridspec_kw={"wspace": 0.30,
                                          "width_ratios": [1.15, 1.0]})

    ax = axes[0]
    dense_stages = ["scores", "softmax_N", "aggregate_dense"]
    sparse_stages = ["scores", "topk_select", "softmax_k", "gather_v",
                     "aggregate_sparse"]
    xs = np.arange(len(ns), dtype=float)
    width = 0.38
    for off, (stages, hatch) in enumerate([(dense_stages, ""),
                                           (sparse_stages, "///")]):
        bottoms = np.zeros(len(ns))
        totals = np.array([sum(pick(rows, n_agents=n, stage=s)["time_ms"]
                               for s in stages) for n in ns])
        for s in stages:
            vals = np.array([pick(rows, n_agents=n, stage=s)["time_ms"]
                             for n in ns]) / totals * 100
            ax.bar(xs + (off - 0.5) * width, vals, width * 0.88, bottom=bottoms,
                   color=_stage_color(s), edgecolor="white", linewidth=0.5,
                   hatch=hatch)
            bottoms += vals
        for x, t in zip(xs + (off - 0.5) * width, totals):
            ax.text(x, 102, f"{t:.2f}", ha="center", fontsize=fs.MIN_PT,
                    color=fs.MUTED_COLOR, rotation=90, va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("agents $N$   (left bar dense, right bar gathered)")
    ax.set_ylabel("share of the path (%)")
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", visible=False)
    handles = [mpl.patches.Patch(facecolor=_stage_color(s), edgecolor="white",
                                 label=STAGE_LABEL[s])
               for s in ("scores", "topk_select", "aggregate_dense",
                         "gather_v", "softmax_N")]
    handles.append(mpl.patches.Patch(facecolor="white", edgecolor="#888888",
                                     hatch="///", label="gathered path"))
    ax.legend(handles=handles, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.28), fontsize=fs.MIN_PT,
              handlelength=1.5, columnspacing=1.0, handletextpad=0.5)

    ax = axes[1]
    for s, mk in (("aggregate_dense", "o"), ("aggregate_sparse", "s"),
                  ("gather_v", "^"), ("topk_select", "v")):
        ys = [100 * pick(rows, n_agents=n, stage=s)["frac_of_binding_roof"]
              for n in ns]
        lbl = {"aggregate_dense": "dense aggregate",
               "aggregate_sparse": "sparse aggregate",
               "gather_v": "gather", "topk_select": "select"}[s]
        ax.plot(ns, ys, marker=mk, color=_stage_color(s), label=lbl,
                markersize=4.2,
                markeredgecolor="#FFFFFF", markeredgewidth=0.5)
    ax.axhline(100, color=fs.RULE_COLOR, linestyle=(0, (3, 2)), linewidth=1.0)
    ax.annotate("machine limit", xy=(0.98, 100), xycoords=("axes fraction",
                                                           "data"),
                ha="right", va="bottom", fontsize=fs.MIN_PT,
                color=fs.RULE_COLOR)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns[::2] + [ns[-1]])
    ax.set_xticklabels([str(n) for n in ns[::2] + [ns[-1]]])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("agents $N$")
    ax.set_ylabel("% of the binding limit")
    ax.set_ylim(0, 135)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.28),
              fontsize=fs.MIN_PT, handlelength=1.8, columnspacing=1.0)
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
    ax.axvspan(0.06, ridge, color="#F2F6FA", zorder=0)
    ax.annotate(f"ridge {ridge:.0f}", xy=(ridge, peak_f * 1.35),
                xytext=(-3, 0), textcoords="offset points",
                ha="right", va="top", fontsize=fs.MIN_PT,
                color=fs.RULE_COLOR)
    ax.annotate("memory-bound", xy=(0.09, peak_f * 1.35), ha="left", va="top",
                fontsize=fs.MIN_PT, color=fs.MUTED_COLOR)

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
    ks = sorted({r["k"] for r in rows})
    k_show = ks[0]
    n_show = ns[-1]

    fig, axes = plt.subplots(1, 3, figsize=(W, 2.5),
                             gridspec_kw={"wspace": 0.42})

    def dense_of(p):
        return p.get("t_dense_best_ms", p["t_dense_fused_ms"])

    def series(sel, field, k=k_show):
        pts = sorted([r for r in rows if r["selector"] == sel and r["k"] == k],
                     key=lambda r: r["n_agents"])
        return ([p["n_agents"] for p in pts],
                [p[field] / dense_of(p) for p in pts])

    # ---- what the count promises -----------------------------------------
    ax = axes[0]
    for sel in ("radius_knn", "morton_window"):
        pts = sorted([r for r in rows if r["selector"] == sel
                      and r["k"] == k_show], key=lambda r: r["n_agents"])
        if not pts:
            continue
        ax.plot([p["n_agents"] for p in pts],
                [p["flop_ratio_vs_dense"] for p in pts], markersize=4.0,
                **fs.style(sel))
    pts = sorted([r for r in rows if r["selector"] == "morton_window"
                  and r["k"] == k_show], key=lambda r: r["n_agents"])
    if pts:
        ax.plot([p["n_agents"] for p in pts],
                [p["n_agents"] / p["k"] for p in pts], color=fs.MUTED_COLOR,
                linestyle=(0, (1, 1.5)), marker=None, label="$N/k$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    # 384 and 512 are close enough on a log axis that their labels collide, so
    # the intermediate one keeps its tick and loses its label.
    ticks = [n for n in ns if n != 384]
    ax.set_xticks(ticks); ax.set_xticklabels([str(n) for n in ticks])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("agents $N$")
    ax.set_ylabel("operations saved ($\\times$)")
    ax.set_title(f"the count promises ($k={k_show}$)", pad=4)
    ax.legend(fontsize=fs.MIN_PT, loc="lower right", handlelength=2.2)

    # ---- what the clock delivers, against N ------------------------------
    ax = axes[1]
    x, y = series("morton_window", "t_total_ms")
    if x:
        ax.plot(x, y, color=fs.MUTED_COLOR, linestyle=(0, (2, 1.5)),
                marker="o", markersize=3.2, markerfacecolor="white",
                linewidth=0.9, label="Z-order, composed")
    for sel in ("radius_knn", "morton_window"):
        x, y = series(sel, "t_total_fused_ms")
        if x:
            ax.plot(x, y, markersize=4.0, **fs.style(sel))
    x, y = series("morton_window", "t_attention_fused_ms")
    if x:
        ax.plot(x, y, color="#444444", linestyle=(0, (1, 1)), marker="o",
                markersize=3.6, markerfacecolor="white",
                label="attention only, selector free")
    fs.parity_line(ax)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(ticks); ax.set_xticklabels([str(n) for n in ticks])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("agents $N$")
    ax.set_ylabel("latency / best dense")
    ax.set_title(f"the clock delivers ($k={k_show}$)", pad=4)
    ax.set_ylim(top=ax.get_ylim()[1] * 3.6)
    ax.legend(fontsize=fs.MIN_PT, loc="upper center", ncol=1,
              handlelength=2.2, borderpad=0.1, labelspacing=0.22)
    fs.parity_line(ax, label="dense")

    # ---- and how far the crossing extends in k ---------------------------
    # The end-to-end crossing exists only at the sparsest settings, so the
    # panel that shows how far it extends belongs beside the one that shows it.
    # The nearest-neighbour path is thirty times dense here and would flatten
    # everything else against the axis, so this panel carries the selector that
    # crosses and the stage the crossing is made of.
    ax = axes[2]
    pts = sorted([r for r in rows if r["selector"] == "morton_window"
                  and r["n_agents"] == n_show], key=lambda r: r["k"])
    if pts:
        ax.plot([p["k"] for p in pts],
                [p["t_total_fused_ms"] / dense_of(p) for p in pts],
                markersize=4.0, **fs.style("morton_window"))
    pts = sorted([r for r in rows if r["selector"] == "morton_window"
                  and r["n_agents"] == n_show], key=lambda r: r["k"])
    if pts:
        ax.plot([p["k"] for p in pts],
                [p["t_attention_fused_ms"] / dense_of(p) for p in pts],
                color="#444444", linestyle=(0, (1, 1)), marker="o",
                markersize=3.6, markerfacecolor="white",
                label="attention only, selector free")
    fs.parity_line(ax, label="dense")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel("retained peers $k$")
    ax.set_ylabel("latency / best dense")
    ax.set_title(f"extent in $k$ (${n_show}$ agents)", pad=4)
    ax.set_ylim(0.035, 3.0)
    ax.legend(fontsize=fs.MIN_PT, loc="lower right", handlelength=2.2,
              borderpad=0.1, labelspacing=0.22)

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
