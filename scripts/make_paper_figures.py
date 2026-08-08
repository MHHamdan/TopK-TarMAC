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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                       facecolor="#f4f4f4", edgecolor="white",
                                       lw=1.2, hatch="///", zorder=2))
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
    cb.set_label("gathered top-$k$ latency $\\div$ dense", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=7.5)

    ax.set_title("Sparsity never wins: every cell is $>1$",
                 fontsize=9.5, fontweight="bold", pad=8, loc="left", x=-0.16)
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

    topk = np.array([r["stages"]["topk_select"]["min_ms"] for r in rows])
    gath = np.array([r["stages"]["gather_v"]["min_ms"] for r in rows])
    smk = np.array([r["stages"]["softmax_k"]["min_ms"] for r in rows])
    aggs = np.array([r["stages"]["agg_sparse"]["min_ms"] for r in rows])
    aggd = np.array([r["stages"]["agg_dense"]["min_ms"] for r in rows])
    total = topk + gath + smk + aggs

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.95),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # --- left: share of the sparse path's time, by stage -----------------
    ax = axes[0]
    parts = [(topk, "#123a63", "top-$k$ select"),
             (gath, "#3b7bb5", "gather $v$"),
             (smk, "#a8c8e4", "softmax over $k$"),
             (aggs, C_GATH, "sparse aggregate")]
    bottom = np.zeros(len(ns))
    for vals, col, lab in parts:
        share = 100 * vals / total
        ax.bar(x, share, 0.66, bottom=bottom, color=col, label=lab,
               edgecolor="white", linewidth=1.1, zorder=3)
        bottom += share

    # Unit goes in the header, not on every label: repeating " ms" six times
    # makes the row collide with itself at this bar spacing.
    for xi, t in zip(x, total):
        ax.text(xi, 102.0, f"{t:.2f}", ha="center", va="bottom",
                fontsize=6.7, color=MUTED)
    # Sits between the value row (~1.02) and the title (pad 26pt); at 1.155
    # it collided with the title's descenders.
    ax.text(0.5, 1.072, "total sparse-path latency (ms)", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.7, color=MUTED)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.set_xticks(x, [str(n) for n in ns])
    ax.set_xlim(-0.75, len(ns) - 0.25)
    ax.set_xlabel("agents $N$")
    ax.set_ylabel("share of sparse-path time")
    ax.set_title("Selection dominates the sparse path",
                 fontsize=8.8, fontweight="bold", loc="left", pad=26)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=6.8, ncol=4, loc="upper center",
              bbox_to_anchor=(0.52, -0.20), handlelength=1.1,
              columnspacing=1.0, handletextpad=0.45)

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
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.75))

    for ax, (dev, m) in zip(axes, pay["machines"].items()):
        peak = m["achievable_gflops"] / 1e3          # TFLOP/s
        bw = m["achievable_gbytes_per_s"] / 1e3      # TB/s
        ridge = m["machine_balance_flops_per_byte"]

        ai = np.logspace(-1.1, 3, 500)
        roof = np.minimum(peak, bw * ai)
        ax.fill_between(ai, peak * 1e-4, roof, color="#f2f2f2", zorder=0)
        ax.plot(ai, roof, color=INK, lw=1.5, zorder=4)
        ax.axvline(ridge, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)

        ax.text(ridge * 0.8, peak * 4.2e-4, "memory-bound", fontsize=6.8,
                color=MUTED, ha="right", va="bottom")
        ax.text(ridge * 1.25, peak * 4.2e-4, "compute-bound", fontsize=6.8,
                color=MUTED, ha="left", va="bottom")
        ax.text(ridge * 0.88, peak * 1.6, f"ridge {ridge:.1f}", fontsize=6.6,
                color=MUTED, ha="right", va="bottom")

        for key, col, mk, lab in (("dense", C_DENSE, "o", "dense"),
                                  ("gathered", C_GATH, "s", "gathered")):
            k = "gather" if key == "gathered" else key
            xs = np.array([r[k]["ai"] for r in pay["rows"]])
            ys = np.minimum(peak, bw * xs)
            ax.plot(xs, ys, marker=mk, ms=5, lw=1.6, color=col, ls="-",
                    label=lab, markeredgecolor="white", markeredgewidth=0.8,
                    zorder=5)

        # Label the endpoints only -- never a number on every point.
        ns = [r["n_agents"] for r in pay["rows"]]
        dx = np.array([r["dense"]["ai"] for r in pay["rows"]])
        ax.annotate(f"$N$={ns[0]}", xy=(dx[0], min(peak, bw * dx[0])),
                    xytext=(10, -11), textcoords="offset points",
                    fontsize=6.6, color=C_DENSE, ha="left")
        ax.annotate(f"$N$={ns[-1]}", xy=(dx[-1], min(peak, bw * dx[-1])),
                    xytext=(7, 4), textcoords="offset points",
                    fontsize=6.6, color=C_DENSE, ha="left")
        gx = pay["rows"][-1]["gather"]["ai"]
        # Centred below the marker: anchored right it ran off the left axis,
        # since the gathered intensity sits close to the x-limit.
        ax.annotate("all $N$", xy=(gx, min(peak, bw * gx)),
                    xytext=(0, -14), textcoords="offset points",
                    fontsize=6.6, color=C_GATH, ha="center")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.22, 2e3)
        ax.set_ylim(peak * 2.2e-4, peak * 3.4)
        ax.set_xlabel("arithmetic intensity (FLOP/byte)")
        ax.set_ylabel("attainable TFLOP/s")
        if dev.startswith("cuda"):
            title = "GPU: both paths far below the ridge"
        else:
            title = "CPU: dense reaches it, gathered cannot"
        ax.set_title(title, fontsize=8.6, fontweight="bold", loc="left",
                     pad=16)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        # The empty corner differs by panel: on the GPU the roof rises far to
        # the right so upper-left is clear; on the CPU the ridge sits at 3.7
        # and the roof crosses that corner, leaving lower-right free.
        ax.legend(frameon=False, fontsize=7.2, borderpad=0.2, handlelength=1.4,
                  loc="upper left" if dev.startswith("cuda") else "lower right")

    fig.tight_layout(w_pad=2.4)
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
