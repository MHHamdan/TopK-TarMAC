"""Figure 1, the study at a glance: the communication module, where the retained
set is fixed, and the measurement protocol, on the four GPUs measured.

A schematic, not a plot of results, so it reads no artifact; it is drawn in the
paper's role colours (src/utils/figstyle.py: scores, select, aggregate, an
intermediate written and read back) at the width it is included at, and
refuses to emit text below the legibility floor.

Writes paper_submission_revision/figures/fig0_methodology.pdf.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import figstyle as fs  # noqa: E402

OUT = ROOT / "paper_submission_revision" / "figures" / "fig0_methodology.pdf"
W, H = fs.TEXT_WIDTH_IN, 3.80

# Panel and box colours: role colours from figstyle, plus a light tint of each.
ROLE = fs.ROLE_COLOR
TINT = {"score": "#E3E9F2", "select": "#FBEBD3", "aggregate": "#E2F0E9",
        "memory": "#F6E1DE", "neutral": "#E8F1F8", "grey": "#F0F0F0",
        "measure": "#EFE6F3", "white": "#FFFFFF"}
EDGE = {"comm": "#2E6DA4", "set": "#4A4A4A", "measure": "#7B3294",
        "A": "#C0651E", "B": "#2E8B6A", "grey": "#4A4A4A"}
INK, MUTED = "#1A1A1A", "#555555"
T_TITLE, T_SUB, T_HEAD = 7.0, 6.2, 7.0
#: A dense schematic: box titles at 7 pt as everywhere, secondary lines at 6.2 pt,
#: as in the figure it replaces. The floor checked here is therefore 6 pt, not
#: figstyle's 7 pt for plots.
MIN_PT = 6.0


def _register_inter() -> None:
    """Inter is the schematic's face; register a user-installed copy, which
    matplotlib's font cache does not always pick up."""
    from matplotlib import font_manager
    for d in (Path.home() / ".fonts", Path.home() / ".local/share/fonts"):
        for f in sorted(d.glob("**/Inter-*.ttf")) if d.exists() else ():
            font_manager.fontManager.addfont(str(f))


def setup() -> tuple[plt.Figure, plt.Axes]:
    _register_inter()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Source Sans 3", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Inter", "mathtext.it": "Inter:italic",
        "mathtext.bf": "Inter:bold",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)          # y grows downwards, in inches
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, face, edge, lw=1.0, r=0.04, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=face, ec=edge, lw=lw, ls=ls, zorder=z))


def node(ax, cx, y, w, title, sub=(), face=TINT["neutral"], edge=EDGE["comm"],
         ha="center", lw=1.0, title_size=T_TITLE):
    """A box with a title and muted sub-lines; returns its bottom edge."""
    h = 0.05 + 0.105 + 0.095 * len(sub) + 0.04
    x = cx - w / 2
    rbox(ax, x, y, w, h, face, edge, lw=lw, z=3)
    tx = cx if ha == "center" else x + 0.07
    ax.text(tx, y + 0.05 + 0.05, title, ha=ha, va="center", fontsize=title_size,
            color=INK, fontweight="medium", zorder=4)
    for i, s in enumerate(sub):
        ax.text(tx, y + 0.05 + 0.105 + 0.095 * i + 0.042, s, ha=ha, va="center",
                fontsize=T_SUB, color=MUTED, zorder=4)
    return y + h


def arrow(ax, x0, y0, x1, y1, color=INK, ls="-", lw=0.9, z=5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=7, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=z))


def tag(ax, x, y, text, face):
    ax.text(x, y, text, ha="left", va="center", fontsize=T_HEAD, color="white",
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.12", fc=face,
                      ec="none"))


def panel_communication(ax) -> float:
    x0, w = 0.03, 1.60
    rbox(ax, x0, 0.03, w, H - 0.06, "#F4F8FC", EDGE["comm"], lw=1.1, r=0.06)
    tag(ax, x0 + 0.06, 0.15, "1  Communication", EDGE["comm"])
    cx = x0 + w / 2
    for i, (xx, lab) in enumerate(((0.30, "$o_1$"), (0.53, "$o_2$"),
                                   (0.76, "$o_3$"), (1.30, "$o_N$"))):
        ax.add_patch(Circle((xx, 0.42), 0.095, fc=TINT["neutral"],
                            ec=EDGE["comm"], lw=0.9, zorder=3))
        ax.text(xx, 0.42, lab, ha="center", va="center", fontsize=T_SUB + 0.5,
                color=INK, zorder=4)
    ax.text(1.03, 0.42, "$\\cdots$", ha="center", va="center", fontsize=T_TITLE,
            color=INK)
    ax.text(cx, 0.62, "$N$ agents (MPE), observations", ha="center",
            va="center", fontsize=T_SUB, color=MUTED)
    bw, y, gap = 1.32, 0.76, 0.15
    arrow(ax, cx, 0.68, cx, y)
    stack = [("observation embedding", ("Linear, GELU",), TINT["neutral"],
              EDGE["comm"]),
             ("Q/K/V projection", ("$q_i$, $k_j$, $v_j$, head width $d$",),
              TINT["neutral"], EDGE["comm"]),
             ("peer selection", ("keep $k$ of $N{-}1$ peers",), TINT["select"],
              ROLE["select"]),
             ("attention aggregation", ("softmax over kept scores",
                                        "$m_i = \\alpha_i V$"),
              TINT["aggregate"], ROLE["aggregate"]),
             ("message", ("GELU, Linear $\\rightarrow m_i$",), TINT["neutral"],
              EDGE["comm"]),
             ("MAPPO", ("shared actor on $[o_i, m_i]$", "centralised critic"),
              TINT["neutral"], EDGE["comm"])]
    tops, bots = [], []
    for i, (t, s, f, e) in enumerate(stack):
        tops.append(y)
        b = node(ax, cx, y, bw, t, s, face=f, edge=e, lw=1.4 if i == 2 else 1.0)
        bots.append(b)
        if i < len(stack) - 1:
            arrow(ax, cx, b, cx, b + gap)
        y = b + gap
    sel_mid = (tops[2] + bots[2]) / 2
    # the circled "2" that points into panel 2
    ax.add_patch(Circle((cx + bw / 2 - 0.13, sel_mid - 0.055), 0.065,
                        fc="white", ec=ROLE["select"], lw=1.0, zorder=5))
    ax.text(cx + bw / 2 - 0.13, sel_mid - 0.055, "2", ha="center", va="center",
            fontsize=T_SUB, color=ROLE["select"], fontweight="bold", zorder=6)
    # bracket for the timed core: peer selection and aggregation
    bx, ya, yb = x0 + 0.085, tops[2] - 0.05, bots[3] + 0.05
    ax.plot([bx + 0.04, bx, bx, bx + 0.04], [ya, ya, yb, yb], color=MUTED,
            lw=0.7)
    ax.text(bx - 0.035, (ya + yb) / 2, "timed attention core", rotation=90,
            ha="center", va="center", fontsize=T_SUB, color=MUTED)
    ax.text(cx, bots[-1] + 0.15, "actions $\\rightarrow$ environment step",
            ha="center", va="center", fontsize=T_SUB, color=MUTED)
    return sel_mid


def panel_set(ax, sel_mid: float) -> float:
    x0, w = 1.72, 2.96
    rbox(ax, x0, 0.03, w, H - 0.06, "#FAFAFA", EDGE["set"], lw=1.1, r=0.06)
    tag(ax, x0 + 0.06, 0.15, "2  Where the retained set is fixed", EDGE["set"])
    arrow(ax, 1.63, sel_mid, x0 - 0.005, sel_mid, color=ROLE["select"],
          ls=(0, (3, 2)))
    # dense reference
    rbox(ax, x0 + 0.10, 0.32, w - 0.20, 0.28, "white", "#9A9A9A", lw=0.9, z=3)
    ax.text(x0 + 0.17, 0.41, "dense reference", ha="left", va="center",
            fontsize=T_TITLE, color=INK, fontweight="medium", zorder=4)
    ax.text(x0 + 0.17, 0.53, "all $N^2$ scores, aggregate over $N$", ha="left",
            va="center", fontsize=T_SUB, color=MUTED, zorder=4)
    ax.text(x0 + w - 0.17, 0.465, "$C_{\\mathrm{dense}} = 2N^2d$", ha="right",
            va="center", fontsize=T_TITLE + 0.5, color=INK, zorder=4)
    top, bot = 0.70, H - 0.33
    ax_, aw = x0 + 0.10, 1.40
    bx_, bw_ = x0 + 1.56, 1.30
    rbox(ax, ax_, top, aw, bot - top, "#FDF6EE", EDGE["A"], lw=1.0, r=0.05, z=2)
    rbox(ax, bx_, top, bw_, bot - top, "#EFF7F3", EDGE["B"], lw=1.0, r=0.05, z=2)
    tag(ax, ax_ + 0.05, top + 0.11, "A  score, then select", EDGE["A"])
    tag(ax, bx_ + 0.05, top + 0.11, "B  select, then score", EDGE["B"])

    # --- A: score, then select
    cA = ax_ + aw / 2
    y = top + 0.25
    b = node(ax, cA, y, 1.18, "all $N^2$ scores",
             ("$s_i = q_i K^{\\top}$ for every pair",), face=TINT["score"],
             edge=ROLE["score"])
    arrow(ax, cA, b, cA, b + 0.10)
    yk = b + 0.10
    bk = node(ax, cA, yk, 1.18, "top-$k$ of $s_i$", ("ranked after scoring",),
              face=TINT["select"], edge=ROLE["select"])
    topk_mid = (yk + bk) / 2
    arrow(ax, cA, bk, cA, bk + 0.10)
    yc = bk + 0.10
    kernels = [("masked", "discarded weights as zeros", TINT["aggregate"],
                ROLE["aggregate"]),
               ("gathered", "writes, re-reads $(B, N, k, d)$", TINT["memory"],
                ROLE["memory"]),
               ("fused", "score row held on chip", TINT["aggregate"],
                ROLE["aggregate"])]
    ch = 0.17 + 3 * 0.27 + 0.03
    rbox(ax, cA - 0.62, yc, 1.24, ch, "white", "#B5B5B5", lw=0.8, z=3)
    ax.text(cA, yc + 0.09, "one function, three kernels", ha="center",
            va="center", fontsize=T_SUB, color=MUTED, zorder=4)
    for i, (t, s, f, e) in enumerate(kernels):
        yy = yc + 0.17 + 0.27 * i
        rbox(ax, cA - 0.57, yy, 1.14, 0.24, f, e, lw=0.9, z=4)
        ax.text(cA - 0.52, yy + 0.07, t, ha="left", va="center",
                fontsize=T_SUB + 0.3, color=INK, fontweight="medium", zorder=5)
        ax.text(cA - 0.52, yy + 0.172, s, ha="left", va="center",
                fontsize=T_SUB, color=MUTED, zorder=5)
    yf = yc + ch + 0.13
    for i, s in enumerate(("$C_{\\mathrm{top}\\text{-}k} = N^2d + Nkd$",
                           "$C_{\\mathrm{dense}} / C_{\\mathrm{top}\\text{-}k}"
                           " = 2N/(N{+}k) < 2$")):
        ax.text(cA, yf + 0.14 * i, s, ha="center", va="center",
                fontsize=T_SUB + 0.3, color=INK)
    ax.text(cA, yf + 0.28, "gather byte screen: $kd \\leq N$", ha="center",
            va="center", fontsize=T_SUB, color=MUTED)

    # --- B: select, then score
    cB = bx_ + bw_ / 2
    y = top + 0.25
    b = node(ax, cB, y, 1.12, "agent positions",
             ("$(B, N, 2)$; no score formed",), face=TINT["grey"],
             edge=EDGE["grey"])
    arrow(ax, cB, b, cB, b + 0.10)
    ys = b + 0.10
    sh = 0.17 + 3 * 0.25 + 0.03
    rbox(ax, cB - 0.58, ys, 1.16, sh, TINT["select"], ROLE["select"], lw=1.0,
         z=3)
    ax.text(cB, ys + 0.09, "selector, cost $C_{\\mathrm{sel}}$", ha="center",
            va="center", fontsize=T_SUB + 0.3, color=INK, zorder=4)
    for i, (t, s) in enumerate((("all-pairs kNN", "$\\Theta(N^2)$, capped"),
                                ("grid kNN", "$\\Theta(Nk)$ + one sort"),
                                ("Z-order window",
                                 "$O(N \\log N)$ sort"))):
        yy = ys + 0.17 + 0.25 * i
        rbox(ax, cB - 0.53, yy, 1.06, 0.22, "white", ROLE["select"], lw=0.8,
             z=4)
        ax.text(cB - 0.48, yy + 0.065, t, ha="left", va="center",
                fontsize=T_SUB + 0.3, color=INK, fontweight="medium", zorder=5)
        ax.text(cB - 0.48, yy + 0.16, s, ha="left", va="center",
                fontsize=T_SUB, color=MUTED, zorder=5)
    be = ys + sh
    arrow(ax, cB, be, cB, be + 0.17)
    ax.text(cB + 0.05, be + 0.085, "indices $(B, N, k)$", ha="left", va="center",
            fontsize=T_SUB, color=MUTED)
    yfs = be + 0.17
    bfs = node(ax, cB, yfs, 1.20, "fixed-set fused attention",
               ("$Nk$ scores + $Nk$ aggregate",), face=TINT["aggregate"],
               edge=ROLE["aggregate"])
    fs_mid = (yfs + bfs) / 2
    ax.text(cB, bfs + 0.10, "top-$k$ set of A, given free", ha="center",
            va="center", fontsize=T_SUB, color=EDGE["A"], style="italic")
    ax.text(cB, bfs + 0.25, "$S = 2N^2d\\,/\\,(2Nkd + C_{\\mathrm{sel}})$",
            ha="center", va="center", fontsize=T_SUB + 0.3, color=INK)
    ax.text(cB, bfs + 0.39, "$\\rightarrow N/k$ only if "
            "$C_{\\mathrm{sel}} = o(Nkd)$", ha="center", va="center",
            fontsize=T_SUB, color=MUTED)
    # the fixed-set comparison: A's exact set handed to B's fused kernel
    xa, xb = cA + 0.59, cB - 0.60
    xm = ax_ + aw + 0.075
    ax.plot([xa, xm, xm], [topk_mid, topk_mid, fs_mid], color=ROLE["select"],
            lw=0.9, ls=(0, (3, 2)), zorder=6)
    arrow(ax, xm, fs_mid, xb, fs_mid, color=ROLE["select"], ls=(0, (3, 2)))

    # role legend
    ly = H - 0.17
    lx = x0 + 0.14
    for lab, face, edge in (("scores", TINT["score"], ROLE["score"]),
                            ("select", TINT["select"], ROLE["select"]),
                            ("aggregate", TINT["aggregate"], ROLE["aggregate"]),
                            ("intermediate written, re-read", TINT["memory"],
                             ROLE["memory"])):
        rbox(ax, lx, ly - 0.06, 0.15, 0.12, face, edge, lw=0.8, r=0.02, z=3)
        t = ax.text(lx + 0.21, ly, lab, ha="left", va="center", fontsize=T_SUB,
                    color=MUTED)
        lx += 0.30 + 0.052 * len(lab)
    return x0 + w


def panel_measurement(ax, left: float) -> None:
    x0 = left + 0.08
    w = W - x0 - 0.03
    rbox(ax, x0, 0.03, w, H - 0.06, "#F9F5FB", EDGE["measure"], lw=1.1, r=0.06)
    tag(ax, x0 + 0.06, 0.15, "3  Measurement", EDGE["measure"])
    arrow(ax, left + 0.005, 0.15, x0 - 0.005, 0.15, color=EDGE["measure"],
          lw=1.0)
    cx = x0 + w / 2
    ax.text(cx, 0.34, "every implementation of 2", ha="center", va="center",
            fontsize=T_SUB + 0.3, color=MUTED)
    ax.text(cx, 0.45, "+ dense, at each $(N, d, k)$", ha="center", va="center",
            fontsize=T_SUB + 0.3, color=MUTED)
    # the four GPUs
    gy = 0.55
    rbox(ax, x0 + 0.08, gy, w - 0.16, 0.55, "white", EDGE["measure"], lw=1.0,
         z=3)
    ax.text(cx, gy + 0.09, "on four NVIDIA GPUs", ha="center", va="center",
            fontsize=T_TITLE, color=INK, fontweight="medium", zorder=4)
    chips = (("RTX PRO 6000", "Blackwell, sm_120"), ("A100", "sm_80"),
             ("H100", "sm_90"), ("H200", "sm_90"))
    cw, chh = (w - 0.34) / 2, 0.16
    for i, (name, arch) in enumerate(chips):
        r, c = divmod(i, 2)
        xx = x0 + 0.13 + c * (cw + 0.08)
        yy = gy + 0.17 + r * (chh + 0.04)
        rbox(ax, xx, yy, cw, chh, TINT["measure"], EDGE["measure"], lw=0.8,
             r=0.03, z=4)
        ax.text(xx + cw / 2, yy + chh / 2, name, ha="center", va="center",
                fontsize=T_SUB, color=INK, zorder=5)
    y = gy + 0.55 + 0.11
    arrow(ax, cx, gy + 0.55, cx, y)
    bw = w - 0.16
    steps = [("equivalence check", ("against the reference, at every",
                                    "reported configuration")),
             ("best dense baseline", ("SDPA and Triton dense, tuned on",
                                      "each GPU (+ FlexAttention in Q4)")),
             ("interleaved timing", ("CUDA events or CUDA-graph replay;",
                                     "min over samples, repeated blocks")),
             ("artifacts", ("raw samples, ratio to best dense;",
                            "Nsight counters (H100, H200)")),
             ("tables, figures, numbers", ("generated from the artifacts,",
                                           "checked against a manifest"))]
    for i, (t, s) in enumerate(steps):
        b = node(ax, cx, y, bw, t, s, face=TINT["measure"],
                 edge=EDGE["measure"])
        if i < len(steps) - 1:
            arrow(ax, cx, b, cx, b + 0.10)
        y = b + 0.10


def main() -> int:
    fig, ax = setup()
    sel_mid = panel_communication(ax)
    right = panel_set(ax, sel_mid)
    panel_measurement(ax, right)
    bad = fs.check_min_font(fig, MIN_PT)
    if bad:
        raise ValueError(f"text below {MIN_PT} pt: {bad}")
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
