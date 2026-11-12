"""Generate every figure in the manuscript from the measurement artifacts.

Design rules applied throughout, so the figures are right by construction
rather than by taste:

* **Form follows the data's job.** Magnitude across a grid -> sequential
  heatmap. Composition of a total -> stacked bars. Two quantities with a
  physical ceiling -> roofline. Growth rates -> log-log with fitted slopes.
* **Sequential means one hue, light to dark.** The crossover ratios are all
  above 1, so there is no meaningful midpoint and a diverging map would invent
  one.
* **Colour is never the only channel.** Every series also carries a distinct
  marker and dash pattern, so the figures survive greyscale printing and
  colour-vision deficiency.
* **Direct labels over legends where a legend would force a lookup**, and
  never a number on every point.
* Sized to the TMLR text block (6.5 in) so nothing is scaled at inclusion
  time, which is what makes captions and axis labels come out at inconsistent
  sizes.

Writes paper/figures/*.pdf. Run after the measurement campaign.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
FIG = ROOT / "paper" / "figures"

TEXTWIDTH = 6.5  # inches, TMLR \textwidth

# Validated categorical palette (adjacent-pair CVD dE 11.0, normal-vision 25.8,
# all slots >= 3:1 on a light surface).
C_DENSE = "#0072B2"   # unfused dense
C_MASK = "#D55E00"    # masked top-k
C_GATH = "#009E73"    # gathered top-k
C_FUSED = "#7B3294"   # fused SDPA
C_GATE = "#444444"    # adaptive gate
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": "#999999",
    "axes.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(name: str) -> dict:
    return json.loads((R / name).read_text())


# ---------------------------------------------------------------------------
# Figure 1 -- the conceptual schematic
# ---------------------------------------------------------------------------

def draw_matrix(ax, x, y, w, h, nrow=7, ncol=7, kept=2, style="full",
                gap_frac=0.16, keep_cols=None):
    """Draw an attention-weight pictogram of nrow x ncol cells.

    The inter-cell gap is a *fraction of the cell*, not an absolute offset:
    an absolute gap larger than the cell width collapses every cell into a
    sliver, which is exactly what it did on the first attempt.

    style: 'ghost'  -- dashed outline only (never materialised)
           'masked' -- every cell drawn; kept coloured, discarded grey but
                       PRESENT, because that is the point of the variant
           'sparse' -- only kept cells drawn; discarded genuinely absent
    """
    cw, ch = w / ncol, h / nrow
    gx, gy = cw * gap_frac, ch * gap_frac
    if style == "ghost":
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=MUTED, lw=0.9,
                               ls=(0, (2.5, 2)), zorder=3))
        return
    rng = np.random.default_rng(4)
    for r in range(nrow):
        if keep_cols is not None:
            cols = keep_cols
        else:
            choices = [c for c in range(ncol) if c != r]
            cols = sorted(rng.choice(choices, size=min(kept, len(choices)),
                                     replace=False))
        for c in range(ncol):
            if ncol == nrow and c == r:
                face = "#ffffff"                  # self-attention masked out
            elif c in cols:
                face = C_GATH if style == "sparse" else C_MASK
            else:
                if style == "sparse":
                    continue                      # genuinely absent
                face = "#cdcdcd"                  # materialised zero
            ax.add_patch(Rectangle(
                (x + c * cw + gx / 2, y + (nrow - 1 - r) * ch + gy / 2),
                cw - gx, ch - gy,
                facecolor=face, edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#9a9a9a", lw=0.7,
                           zorder=5))


def arrow(ax, x0, y0, x1, y1, label=None, color=MUTED, dy=0.028, fs=7.2):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8,
        lw=0.9, color=color, shrinkA=0, shrinkB=0, zorder=6))
    if label:
        ax.text((x0 + x1) / 2, y0 + dy, label, ha="center", va="bottom",
                fontsize=fs, color=MUTED, zorder=6)


def fig_schematic() -> None:
    """Three implementations of one function.

    Laid out in axes coordinates with an explicit aspect correction, because
    the axes are 1x1 in data units on a non-square figure: without it the
    "square" matrix cells come out stretched and the labels drift into them.
    """
    FW, FH = TEXTWIDTH, 3.75
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    asp = FH / FW          # y-units per x-unit

    rows = [
        (0.720, "Fused dense", "softmax and value\nproduct fused", "ghost",
         C_FUSED, "$2N^2d$ MACs", "no $N\\!\\times\\!N$ matrix"),
        (0.470, "Top-$k$, masked", "zeros stored, then the\nsame dense matmul",
         "masked", C_MASK, "$2N^2d$ MACs — same as dense",
         "$N\\!\\times\\!N$ weights, zeros included"),
        (0.235, "Top-$k$, gathered", "kept rows compacted\nbefore the product", "sparse",
         C_GATH, "$N^2d + Nkd$ MACs",
         "$N\\!\\times\\!N$ + $N\\!\\times\\!k\\!\\times\\!d$"),
    ]

    hm = 0.160                 # matrix height in y-units
    wm = hm * asp              # square cells => square 7x7 matrix
    xm = 0.235                 # matrix left edge, clear of the label column
    ws = wm * 2 / 7            # compacted block is 2 columns wide
    x_mid = 0.400              # compaction block left edge
    x_out = 0.605              # cost column

    ax.text(0.0, 0.975, "The same function, three ways of computing it",
            fontsize=9.8, fontweight="bold", color=INK, ha="left", va="top")
    ax.text(0.0, 0.910,
            "All three compute $QK^{\\top}$ in full: selecting peers by score "
            "requires every score.",
            fontsize=7.5, color=MUTED, ha="left", va="top")

    ax.text(xm + wm / 2, 0.845, "attention weights", fontsize=7.4, color=INK,
            ha="center", va="center", fontweight="bold")
    ax.text((xm + wm + x_out) / 2, 0.845, "aggregation", fontsize=7.4,
            color=INK, ha="center", va="center", fontweight="bold")
    ax.text(x_out, 0.845, "executed / stored", fontsize=7.4, color=INK,
            ha="left", va="center", fontweight="bold")
    ax.plot([0.0, 1.0], [0.815, 0.815], color="#e2e2e2", lw=0.8, zorder=1)

    for ycen, name, sub, style, col, macs, mem in rows:
        yb = ycen - hm / 2
        ax.text(0.0, ycen + 0.042, name, fontsize=8.5, color=col,
                fontweight="bold", ha="left", va="center")
        ax.text(0.0, ycen - 0.030, sub, fontsize=6.9, color=MUTED,
                ha="left", va="center", linespacing=1.4)

        draw_matrix(ax, xm, yb, wm, hm, nrow=7, ncol=7, kept=2, style=style)
        if style == "ghost":
            # Below the box, not inside it: at this width the phrase does not
            # fit within the frame at a legible size.
            ax.text(xm + wm / 2, yb - 0.028, "never materialised",
                    fontsize=6.6, color=MUTED, ha="center", va="top",
                    style="italic", zorder=6)

        if style == "sparse":
            arrow(ax, xm + wm + 0.012, ycen, x_mid - 0.012, ycen)
            draw_matrix(ax, x_mid, yb, ws, hm, nrow=7, ncol=2,
                        style="sparse", keep_cols=[0, 1])
            ax.text(x_mid + ws / 2, yb - 0.028, "$N\\!\\times\\!k$",
                    fontsize=6.6, color=MUTED, ha="center", va="top")
            arrow(ax, x_mid + ws + 0.012, ycen, x_out - 0.014, ycen,
                  "$(N\\!\\times\\!k)(k\\!\\times\\!d)$", dy=0.026, fs=6.9)
        elif style == "masked":
            arrow(ax, xm + wm + 0.010, ycen, x_out - 0.014, ycen,
                  "$(N\\!\\times\\!N)(N\\!\\times\\!d)$ dense", dy=0.022,
                  fs=6.9)
        else:
            arrow(ax, xm + wm + 0.010, ycen, x_out - 0.014, ycen,
                  "fused, tiled", dy=0.022, fs=6.9)

        ax.text(x_out, ycen + 0.040, macs, fontsize=7.4, color=INK,
                ha="left", va="center")
        ax.text(x_out, ycen - 0.033, mem, fontsize=6.9, color=MUTED,
                ha="left", va="center")

    ax.plot([0.0, 1.0], [0.078, 0.078], color="#e2e2e2", lw=0.8)
    for i, (fc, lab) in enumerate([
            (C_MASK, "kept weight"),
            ("#c9c9c9", "discarded — still stored and multiplied"),
            (C_GATH, "kept weight, compacted")]):
        x = [0.0, 0.235, 0.655][i]
        ax.add_patch(Rectangle((x, 0.020), 0.016, 0.028, facecolor=fc,
                               edgecolor="none"))
        ax.text(x + 0.023, 0.034, lab, fontsize=6.9, color=MUTED,
                va="center", ha="left")

    fig.savefig(FIG / "fig1_schematic.pdf")
    plt.close(fig)
    print("wrote fig1_schematic.pdf")


# ---------------------------------------------------------------------------
# Figure 2 -- crossover surface
# ---------------------------------------------------------------------------

def fig_crossover() -> None:
    pay = load("crossover_surface.json")
    ds = sorted({c["d"] for c in pay["cells"]})
    bs = sorted({c["batch"] for c in pay["cells"]})
    ns = sorted({c["n_agents"] for c in pay["cells"]})

    grid = np.full((len(ds) * len(bs), len(ns)), np.nan)
    rows = [(d, b) for d in ds for b in bs]
    for i, (d, b) in enumerate(rows):
        for j, n in enumerate(ns):
            c = next((x for x in pay["cells"] if x["d"] == d and x["batch"] == b
                      and x["n_agents"] == n and not x.get("skipped")), None)
            if c:
                grid[i, j] = c["ratio_vs_dense"]

    # Sequential, one hue light->dark: every value is >1, so there is no
    # midpoint for a diverging map to sit on.
    cmap = LinearSegmentedColormap.from_list(
        "seq", ["#eaf2fa", "#b8d4ec", "#7fb0dc", "#4a89c8", "#1f5f9e",
                "#123a63"])

    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.82, 3.5))
    im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=1.0,
                   vmax=np.nanmax(grid))

    ax.set_xticks(range(len(ns)), [str(n) for n in ns])
    ax.set_xlabel("agents $N$")
    ax.set_yticks(range(len(rows)),
                  [f"$d$={d}, $B$={b}" for d, b in rows])
    ax.set_ylabel("head width and batch")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                # Hatch takes the edge colour, so a white edge would draw an
                # invisible hatch; the white cell gaps come from the minor
                # grid, which is painted over the patch.
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                       facecolor="#f6f6f6",
                                       edgecolor="#bdbdbd",
                                       lw=0.6, hatch="///", zorder=2))
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.6,
                        color=MUTED, zorder=3)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.4,
                        color="white" if v > 1.9 else INK, zorder=3)
    # 2px surface gap between cells
    ax.set_xticks(np.arange(-.5, len(ns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    # The denominator is named in full: the ratio is against *unfused* dense,
    # not against the fused SDPA baseline, and the two are not interchangeable.
    cb.set_label("latency ratio: best-$k$ gathered $\\div$ unfused dense",
                 fontsize=7.6)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=7.5)

    # Neutral and self-describing: the title names the quantity plotted, and
    # the claim about how the surface compares to any baseline is left to the
    # caption, which can qualify it.
    ax.set_title("Best-$k$ gathered latency $\\div$ unfused dense, by "
                 "$(d, B, N)$",
                 fontsize=9.2, fontweight="bold", pad=8, loc="left", x=-0.16)

    budget = pay.get("mem_budget_mib")
    note = "not measured" if budget is None else (
        f"not measured — exceeds the {budget / 1024:.0f} GiB "
        "activation budget")
    ax.legend(handles=[Patch(facecolor="#f6f6f6", edgecolor="#bdbdbd", lw=0.6,
                             hatch="///", label=note)],
              loc="upper left", bbox_to_anchor=(-0.16, -0.115), frameon=False,
              fontsize=7.0, handlelength=1.6, handletextpad=0.6,
              borderpad=0.0)
    fig.savefig(FIG / "fig2_crossover.pdf")
    plt.close(fig)
    print("wrote fig2_crossover.pdf")


# ---------------------------------------------------------------------------
# Figure 3 -- where the time goes
# ---------------------------------------------------------------------------

def fig_decomposition() -> None:
    """Where the sparse path's time goes, and what the aggregation costs.

    The left panel is a *share* composition on a linear axis, not stacked
    absolute latency on a log axis: segments stacked on a log scale have
    heights that do not correspond to their values, so the picture would lie
    about the very quantity it exists to show. Absolute totals are printed
    above each bar so no information is lost.
    """
    pay = load("component_decomposition.json")
    rows = sorted(pay["rows"], key=lambda r: r["n_agents"])
    ns = [r["n_agents"] for r in rows]
    x = np.arange(len(ns))

    def stage(name: str) -> np.ndarray:
        return np.array([r["stages"][name]["min_ms"] for r in rows])

    # The gathered forward pass is these SEVEN stages and no others. Leaving
    # out qkv/scores/proj shrinks the denominator by ~20-50% and inflates every
    # share; softmax_N and agg_dense belong to the *dense* path and must stay
    # out of this total. min_ms throughout, matching the paper's protocol.
    SPARSE_STAGES = ["topk_select", "gather_v", "softmax_k", "agg_sparse",
                     "scores", "qkv", "proj"]
    vals = {s: stage(s) for s in SPARSE_STAGES}
    aggs = vals["agg_sparse"]
    aggd = stage("agg_dense")
    total = np.sum([vals[s] for s in SPARSE_STAGES], axis=0)

    print("  fig3 left panel, gathered forward pass over all 7 stages "
          "(min_ms):")
    print("     N    total_ms   topk%   topk+gather+softmax_k%")
    for i, n in enumerate(ns):
        share_topk = 100 * vals["topk_select"][i] / total[i]
        share_sel = 100 * (vals["topk_select"][i] + vals["gather_v"][i]
                           + vals["softmax_k"][i]) / total[i]
        print(f"   {n:>4}   {total[i]:8.3f}  {share_topk:6.1f}  "
              f"{share_sel:16.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.95),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # --- left: share of the sparse path's time, by stage -----------------
    ax = axes[0]
    # Bottom four are the stages the top-k path adds or replaces; the top three
    # (grey) are the stages it shares with dense attention. Grouping them by
    # colour family keeps the headline -- selection's growing share -- readable
    # against a stack that is now seven deep.
    parts = [("topk_select", "#123a63", "top-$k$ select"),
             ("gather_v", "#3b7bb5", "gather $v$"),
             ("softmax_k", "#a8c8e4", "softmax over $k$"),
             ("agg_sparse", C_GATH, "sparse aggregate"),
             # A grey ramp that runs light->dark going up the stack: the
             # segment next to the green must not share its luminance, or the
             # two merge under greyscale printing.
             ("scores", "#dcdcdc", "scores $QK^{\\top}$"),
             ("qkv", "#a4a4a4", "$QKV$ projection"),
             ("proj", "#5f5f5f", "output projection")]
    bottom = np.zeros(len(ns))
    for key, col, lab in parts:
        share = 100 * vals[key] / total
        ax.bar(x, share, 0.66, bottom=bottom, color=col, label=lab,
               edgecolor="white", linewidth=0.7, zorder=3)
        bottom += share

    # Unit goes in the header, not on every label: repeating " ms" six times
    # makes the row collide with itself at this bar spacing.
    for xi, t in zip(x, total):
        ax.text(xi, 102.0, f"{t:.2f}", ha="center", va="bottom",
                fontsize=6.7, color=MUTED)
    # Sits between the value row (~1.02) and the title (pad 26pt); at 1.155
    # it collided with the title's descenders.
    ax.text(0.5, 1.072, "total gathered forward pass (ms)",
            transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.7, color=MUTED)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.set_xticks(x, [str(n) for n in ns])
    ax.set_xlim(-0.75, len(ns) - 0.25)
    ax.set_xlabel("agents $N$")
    ax.set_ylabel("share of gathered forward pass")
    ax.set_title("Selection grows to dominate the gathered path",
                 fontsize=8.8, fontweight="bold", loc="left", pad=26)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6.6, ncol=4, loc="upper center",
              bbox_to_anchor=(0.52, -0.20), handlelength=1.1,
              columnspacing=1.0, handletextpad=0.45, labelspacing=0.5)

    # --- right: the aggregation sparsity was supposed to shrink ----------
    ax = axes[1]
    ratio = aggs / aggd
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.plot(x, ratio, marker="s", ms=5, lw=1.8, color=C_GATH,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3)
    ax.set_xticks(x, [str(n) for n in ns])
    ax.set_xlim(-0.4, len(ns) - 0.6)
    ax.set_xlabel("agents $N$")
    ax.set_ylabel("sparse $\\div$ dense aggregation")
    ax.set_title("The aggregation gets slower, not faster",
                 fontsize=8.8, fontweight="bold", loc="left", pad=26)
    ax.text(len(ns) - 0.75, 1.0, "parity", fontsize=6.9, color=MUTED,
            va="bottom", ha="right")
    ax.annotate(f"{ratio[-1]:.1f}$\\times$ slower\nwith 4$\\times$ fewer MACs",
                xy=(x[-1], ratio[-1]), xytext=(x[-1] - 2.5, ratio[-1] + 0.30),
                fontsize=7.0, color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    ax.set_ylim(0.6, max(ratio) + 0.95)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.tight_layout(w_pad=2.4)
    fig.savefig(FIG / "fig3_decomposition.pdf")
    plt.close(fig)
    print("wrote fig3_decomposition.pdf")


# ---------------------------------------------------------------------------
# Figure 4 -- roofline
# ---------------------------------------------------------------------------

def fig_roofline() -> None:
    """Roofline for both paths on both architectures.

    Note the asymmetry the panels are there to expose: on the GPU both paths
    sit far below the ridge, but on the CPU -- whose machine balance is an
    order of magnitude lower -- the dense path *reaches* the ridge while the
    gathered path does not. Claiming "both memory-bound" on both machines
    would be wrong, and would also throw away the more interesting fact.
    """
    pay = load("roofline.json")
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.9))

    ns = [r["n_agents"] for r in pay["rows"]]
    dx = np.array([r["dense"]["ai"] for r in pay["rows"]])
    gx_all = np.array([r["gather"]["ai"] for r in pay["rows"]])

    for ax, (dev, m) in zip(axes, pay["machines"].items()):
        peak = m["achievable_gflops"] / 1e3          # TFLOP/s
        bw = m["achievable_gbytes_per_s"] / 1e3      # TB/s
        ridge = m["machine_balance_flops_per_byte"]
        gpu = dev.startswith("cuda")

        # Panel-specific x window. The CPU ridge sits at 3.7, in among the
        # dense intensities themselves, so a three-decade window would put the
        # two rightmost dense points a pixel from the ridge line and the
        # crossing -- the whole point of the panel -- would be invisible.
        xlo, xhi = (0.40, 400.0) if gpu else (0.40, 40.0)
        ai = np.logspace(np.log10(xlo), np.log10(xhi), 600)
        roof = np.minimum(peak, bw * ai)
        # Tight enough that the data fills the axes: the floor is just under
        # the roof's left end rather than four empty decades below it.
        ylo, yhi = bw * xlo * 0.50, peak * 2.7

        ax.fill_between(ai, ylo * 0.5, roof, color="#f2f2f2", zorder=0)
        ax.plot(ai, roof, color=INK, lw=1.5, zorder=4)
        ax.axvline(ridge, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)

        # Regime labels along the floor; the ridge value goes above the roof's
        # flat top, so neither can reach the y tick labels.
        ax.text(ridge / 1.12, ylo * 1.45, "memory-bound", fontsize=6.8,
                color=MUTED, ha="right", va="bottom")
        ax.text(ridge * 1.12, ylo * 1.45, "compute-bound", fontsize=6.8,
                color=MUTED, ha="left", va="bottom")
        rtxt = f"{ridge:.1f}" if ridge >= 10 else f"{ridge:.2f}"
        # When the ridge sits in the left third of the window (the CPU panel,
        # whose balance is ~1.4) a centred label collides with the legend, so
        # push it to the right of the line instead.
        left_third = ridge < (xlo * (xhi / xlo) ** 0.33)
        ax.text(ridge * (1.18 if left_third else 1.0), peak * 1.45,
                f"ridge {rtxt}", fontsize=6.6, color=MUTED,
                ha="left" if left_third else "center", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
                zorder=6)

        # Colour is not the only channel: solid vs dashed keeps the two paths
        # apart in greyscale, as the module docstring promises.
        for key, col, mk, ls, lab in (
                ("dense", C_DENSE, "o", "-", "dense"),
                ("gather", C_GATH, "s", (0, (4, 2)), "gathered")):
            xs = np.array([r[key]["ai"] for r in pay["rows"]])
            ys = np.minimum(peak, bw * xs)
            ax.plot(xs, ys, marker=mk, ms=4.6, lw=1.5, color=col, ls=ls,
                    label=lab, markeredgecolor="white", markeredgewidth=0.8,
                    zorder=5)

        # Label the endpoints only, each on a leader line so it cannot be read
        # as belonging to a neighbouring marker: the dense points at N=192 and
        # N=512 sit within 5% of each other in intensity.
        lead = dict(arrowstyle="-", lw=0.55, color=MUTED,
                    shrinkA=1.0, shrinkB=2.5)
        # Above the roof, while the gathered label goes below it: the two sit
        # on opposite sides of the line and cannot be swapped by eye.
        # On the CPU panel the ridge label occupies the space above-left of
        # the first dense point, so that label goes below the roof instead.
        f_off, f_ha, f_va = ((-15, 14), "right", "center") if gpu \
            else ((0, -15), "center", "top")
        ax.annotate(f"$N$={ns[0]}", xy=(dx[0], min(peak, bw * dx[0])),
                    xytext=f_off, textcoords="offset points",
                    fontsize=6.6, color=C_DENSE, ha=f_ha, va=f_va,
                    arrowprops=lead, zorder=6)
        ax.annotate(f"$N$={ns[-1]}", xy=(dx[-1], min(peak, bw * dx[-1])),
                    xytext=(10, 20), textcoords="offset points",
                    fontsize=6.6, color=C_DENSE, ha="left", va="center",
                    arrowprops=lead, zorder=6)
        gx = float(np.exp(np.log(gx_all).mean()))
        ax.annotate("all $N$", xy=(gx, min(peak, bw * gx)),
                    xytext=(0, -17), textcoords="offset points",
                    fontsize=6.6, color=C_GATH, ha="center", va="top",
                    arrowprops=lead, zorder=6)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel("arithmetic intensity (FLOP/byte)")
        # Marker height is the *roof* at each path's analytic intensity, not a
        # measurement; the label has to say so, because a reader who assumes
        # otherwise reads this figure as a throughput result.
        ax.set_ylabel("attainable TFLOP/s at analytic intensity")
        if gpu:
            title = "GPU: both paths far below the ridge"
        else:
            title = "CPU: dense crosses the ridge, gathered cannot"
        ax.set_title(title, fontsize=8.6, fontweight="bold", loc="left",
                     pad=16)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if not gpu:
            past = [(n, a) for n, a in zip(ns, dx) if a > ridge]
            ax.text(xhi / 1.05, peak * 0.55,
                    "dense AI " + ", ".join(f"{a:.2f}" for _, a in past)
                    + f"\nat $N$={', '.join(str(n) for n, _ in past)}"
                    + f"\nexceed the ridge {ridge:.2f}",
                    fontsize=6.2, color=C_DENSE, ha="right", va="top",
                    linespacing=1.35, zorder=6)
        # Both panels plot the same two series, so one legend serves both; on
        # the CPU panel it would sit under the ridge label.
        if gpu:
            ax.legend(frameon=False, fontsize=7.2, borderpad=0.2,
                      handlelength=1.6, loc="upper left")

    fig.tight_layout(w_pad=2.4, rect=(0, 0.055, 1, 1))
    fig.text(0.005, 0.005,
             "Markers are the roof evaluated at each path's analytic "
             "arithmetic intensity, $\\min(\\mathrm{peak},\\ "
             "\\mathrm{BW}\\times\\mathrm{AI})$ — an upper bound, not "
             "measured throughput.",
             fontsize=6.4, color=MUTED, ha="left", va="bottom")
    fig.savefig(FIG / "fig4_roofline.pdf")
    plt.close(fig)
    print("wrote fig4_roofline.pdf")


# ---------------------------------------------------------------------------
# Figure 5 -- memory scaling
# ---------------------------------------------------------------------------

def fig_memory() -> None:
    mem = load("memory_analysis.json")
    tag = next(t for t in mem["measured"] if "batch256" in t)
    arms = mem["measured"][tag]["arms"]

    order = [("adaptive", C_GATE, "^", "adaptive gate"),
             ("topk_masked", C_MASK, "v", "top-$k$ masked"),
             ("topk_gather", C_GATH, "s", "top-$k$ gathered"),
             ("dense", C_DENSE, "o", "dense"),
             ("dense_sdpa", C_FUSED, "D", "fused dense")]

    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.52, 2.7))
    for key, col, mk, lab in order:
        a = arms.get(key)
        if not a or not a["n"]:
            continue
        exp = a.get("fitted_exponent_offset_corrected")
        ax.plot(a["n"], a["peak_vram_mib"], marker=mk, ms=4.4, lw=1.5,
                color=col, markeredgecolor="white", markeredgewidth=0.7,
                label=f"{lab}  ($N^{{{exp:.2f}}}$)" if exp else lab)

    ad = arms.get("adaptive", {})
    if ad.get("oom_at_n"):
        n_oom = ad["oom_at_n"][0]
        ax.axvline(n_oom, color=C_GATE, lw=0.8, ls=(0, (2, 2)))
        ax.annotate(f"gate OOM\nat $N$={n_oom}", xy=(n_oom, 0.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(-5, 8), textcoords="offset points",
                    fontsize=6.8, color=C_GATE, ha="right", va="bottom",
                    linespacing=1.3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("agents $N$")
    ax.set_ylabel("peak activation memory (MiB)")
    ax.set_title("Cubic memory in the arm meant to save cost",
                 fontsize=8.8, fontweight="bold", loc="left")
    ax.set_xticks([6, 12, 24, 48, 96, 192, 384],
                  ["6", "12", "24", "48", "96", "192", "384"])
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(5, 780)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left",
              handlelength=1.3, labelspacing=0.28, borderpad=0.2)

    fig.savefig(FIG / "fig5_memory.pdf")
    plt.close(fig)
    print("wrote fig5_memory.pdf")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_schematic()
    fig_crossover()
    fig_decomposition()
    fig_roofline()
    fig_memory()
    print(f"\nall figures written to {FIG}")


if __name__ == "__main__":
    main()
