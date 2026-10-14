"""Figure 1, the study at a glance: the communication module, where the retained
set is fixed, and the measurement protocol, on the four GPUs measured.

The three panels are stacked vertically in reading order (1 -> 2 -> 3), each a
full-width band whose stages run left to right, so that no stage is squeezed
into a narrow column.

A schematic, not a plot of results, so it reads no artifact; it is drawn in the
paper's role colours (src/utils/figstyle.py: scores, select, aggregate, an
intermediate written and read back) at the width it is included at, refuses to
emit text below the legibility floor, and refuses text that overflows its box.

Writes paper_submission_revision/figures/fig0_methodology.pdf (and a PNG preview
next to it with --png).
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
W = fs.TEXT_WIDTH_IN

ROLE = fs.ROLE_COLOR
TINT = {"score": "#E3E9F2", "select": "#FBEBD3", "aggregate": "#E2F0E9",
        "memory": "#F6E1DE", "neutral": "#E8F1F8", "grey": "#F0F0F0",
        "measure": "#EFE6F3", "white": "#FFFFFF"}
EDGE = {"comm": "#2E6DA4", "set": "#4A4A4A", "measure": "#7B3294",
        "A": "#C0651E", "B": "#2E8B6A", "grey": "#4A4A4A"}
INK, MUTED = "#1A1A1A", "#555555"
T_TITLE, T_SUB, T_HEAD = 7.0, 6.4, 7.0
#: Box titles at 7 pt, secondary lines at 6.4 pt; the floor checked is 6 pt.
MIN_PT = 6.0
LINE = 0.10          # line pitch of secondary text, inches
GAP = 0.15           # horizontal gap between stages, inches

#: (box patch, [text artists]) pairs whose text must stay inside the box.
_CONTAINED: list = []


def _register_inter() -> None:
    """Inter is the schematic's face; register a user-installed copy, which
    matplotlib's font cache does not always pick up."""
    from matplotlib import font_manager
    for d in (Path.home() / ".fonts", Path.home() / ".local/share/fonts"):
        for f in sorted(d.glob("**/Inter-*.ttf")) if d.exists() else ():
            font_manager.fontManager.addfont(str(f))


def setup(h: float) -> tuple[plt.Figure, plt.Axes]:
    _register_inter()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Source Sans 3", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Inter", "mathtext.it": "Inter:italic",
        "mathtext.bf": "Inter:bold",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(W, h))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(h, 0)          # y grows downwards, in inches
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, face, edge, lw=1.0, r=0.04, ls="-", z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=face, ec=edge, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def node(ax, x, y, w, h, title, sub=(), face=TINT["neutral"], edge=EDGE["comm"],
         lw=1.0, z=3):
    """A box of fixed size with a title and muted sub-lines, centred as a block.
    Returns (left, top, right, bottom)."""
    p = rbox(ax, x, y, w, h, face, edge, lw=lw, z=z)
    titles = title if isinstance(title, tuple) else (title,)
    block = 0.115 * len(titles) + LINE * len(sub)
    ty = y + (h - block) / 2 + 0.06
    texts = []
    for t in titles:
        texts.append(ax.text(x + w / 2, ty, t, ha="center", va="center",
                             fontsize=T_TITLE, color=INK, fontweight="medium",
                             zorder=z + 1))
        ty += 0.115
    for s in sub:
        texts.append(ax.text(x + w / 2, ty, s, ha="center", va="center",
                             fontsize=T_SUB, color=MUTED, zorder=z + 1))
        ty += LINE
    _CONTAINED.append((p, texts))
    return x, y, x + w, y + h


def arrow(ax, x0, y0, x1, y1, color=INK, ls="-", lw=0.9, z=5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=7, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=z))


def path(ax, pts, color=INK, ls="-", lw=0.9, z=6):
    """Orthogonal poly-line with an arrow head on the last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=z,
            solid_capstyle="butt")
    arrow(ax, xs[-2], ys[-2], xs[-1], ys[-1], color=color, ls=ls, lw=lw, z=z)


def tag(ax, x, y, text, face):
    ax.text(x, y, text, ha="left", va="center", fontsize=T_HEAD, color="white",
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.12", fc=face,
                      ec="none"))


def marker(ax, x, y, text, color):
    ax.add_patch(Circle((x, y), 0.062, fc="white", ec=color, lw=1.0, zorder=7))
    ax.text(x, y, text, ha="center", va="center", fontsize=T_SUB, color=color,
            fontweight="bold", zorder=8)


def row(x0, x1, widths):
    """Left edges for boxes of the given relative widths filling [x0, x1]
    with GAP between them."""
    avail = x1 - x0 - GAP * (len(widths) - 1)
    scale = avail / sum(widths)
    xs, ws, x = [], [], x0
    for w in widths:
        xs.append(x)
        ws.append(w * scale)
        x += w * scale + GAP
    return xs, ws


# --------------------------------------------------------------------------
# panel 1: the communication module, left to right
# --------------------------------------------------------------------------
def panel_communication(ax, top: float) -> tuple[float, float]:
    h = 1.24
    x0, x1 = 0.03, W - 0.03
    rbox(ax, x0, top, x1 - x0, h, "#F4F8FC", EDGE["comm"], lw=1.1, r=0.06)
    tag(ax, x0 + 0.07, top + 0.13, "1  Communication module", EDGE["comm"])
    ry, rh = top + 0.36, 0.56
    xs, ws = row(x0 + 0.12, x1 - 0.12, (0.95, 0.80, 0.85, 1.0, 1.05, 0.80, 1.10))
    # agents and their observations
    ax_, aw = xs[0], ws[0]
    p = rbox(ax, ax_, ry, aw, rh, TINT["neutral"], EDGE["comm"], z=3)
    cy = ry + 0.17
    for xx, lab in ((0.17, "$o_1$"), (0.38, "$o_2$"), (aw - 0.17, "$o_N$")):
        ax.add_patch(Circle((ax_ + xx, cy), 0.085, fc="white", ec=EDGE["comm"],
                            lw=0.9, zorder=4))
        ax.text(ax_ + xx, cy, lab, ha="center", va="center",
                fontsize=T_SUB + 0.4, color=INK, zorder=5)
    ax.text(ax_ + (0.38 + aw - 0.17) / 2, cy, "$\\cdots$", ha="center",
            va="center", fontsize=T_TITLE, color=INK, zorder=5)
    t1 = ax.text(ax_ + aw / 2, ry + 0.37, "$N$ agents (MPE)", ha="center",
                 va="center", fontsize=T_TITLE, color=INK, fontweight="medium",
                 zorder=5)
    t2 = ax.text(ax_ + aw / 2, ry + 0.47, "observations", ha="center",
                 va="center", fontsize=T_SUB, color=MUTED, zorder=5)
    _CONTAINED.append((p, [t1, t2]))
    stages = [
        ("embedding", ("Linear, GELU",), TINT["neutral"], EDGE["comm"], 1.0),
        ("Q/K/V", ("$q_i$, $k_j$, $v_j$", "head width $d$"), TINT["neutral"],
         EDGE["comm"], 1.0),
        ("peer selection", ("keep $k$ of $N{-}1$", "peers"), TINT["select"],
         ROLE["select"], 1.4),
        ("aggregation", ("softmax over kept", "scores; $m_i = \\alpha_i V$"),
         TINT["aggregate"], ROLE["aggregate"], 1.0),
        ("message", ("GELU, Linear", "$\\rightarrow m_i$"), TINT["neutral"],
         EDGE["comm"], 1.0),
        ("MAPPO", ("shared actor", "on $[o_i, m_i]$;", "centralised critic"),
         TINT["neutral"], EDGE["comm"], 1.0),
    ]
    boxes = [(ax_, ry, ax_ + aw, ry + rh)]
    for (t, s, f, e, lw), x, w in zip(stages, xs[1:], ws[1:]):
        boxes.append(node(ax, x, ry, w, rh, t, s, face=f, edge=e, lw=lw))
    mid = ry + rh / 2
    for (l0, _, r0, _), (l1, _, _, _) in zip(boxes, boxes[1:]):
        arrow(ax, r0, mid, l1, mid)
    sel, agg = boxes[3], boxes[4]
    marker(ax, sel[2] - 0.06, sel[1] + 0.06, "2", ROLE["select"])
    # the timed attention core: peer selection and aggregation, bracketed above
    by = ry - 0.07
    ax.plot([sel[0], sel[0], agg[2], agg[2]], [by + 0.035, by, by, by + 0.035],
            color=MUTED, lw=0.7)
    ax.text((sel[0] + agg[2]) / 2, by - 0.075, "timed attention core",
            ha="center", va="center", fontsize=T_SUB, color=MUTED)
    # the environment loop, closing under the row
    ly = ry + rh + 0.15
    mappo, agents = boxes[-1], boxes[0]
    xm, xa = (mappo[0] + mappo[2]) / 2, (agents[0] + agents[2]) / 2
    path(ax, [(xm, ry + rh), (xm, ly), (xa, ly), (xa, ry + rh)], color=MUTED,
         ls=(0, (3, 2)), lw=0.8)
    ax.text((xm + xa) / 2, ly - 0.065, "actions $\\rightarrow$ environment step",
            ha="center", va="center", fontsize=T_SUB, color=MUTED,
            bbox=dict(fc="#F4F8FC", ec="none", pad=0.6), zorder=7)
    return top + h, (sel[0] + sel[2]) / 2


# --------------------------------------------------------------------------
# panel 2: where the retained set is fixed, A beside B
# --------------------------------------------------------------------------
def stack(ax, x, y, w, header, items, face, edge, inner_face, inner_edge):
    """A container with a header line and stacked two-line mini boxes."""
    ih, ig = 0.27, 0.05
    h = 0.20 + len(items) * (ih + ig) + 0.03
    p = rbox(ax, x, y, w, h, face, edge, lw=0.9, z=3)
    texts = [ax.text(x + w / 2, y + 0.11, header, ha="center", va="center",
                     fontsize=T_SUB + 0.3, color=INK, zorder=4)]
    for i, (t, s, f, e) in enumerate(items):
        yy = y + 0.20 + i * (ih + ig)
        q = rbox(ax, x + 0.06, yy, w - 0.12, ih, f or inner_face,
                 e or inner_edge, lw=0.8, z=4)
        tt = [ax.text(x + 0.12, yy + 0.085, t, ha="left", va="center",
                      fontsize=T_SUB + 0.3, color=INK, fontweight="medium",
                      zorder=5),
              ax.text(x + 0.12, yy + 0.19, s, ha="left", va="center",
                      fontsize=T_SUB, color=MUTED, zorder=5)]
        _CONTAINED.append((q, tt))
    _CONTAINED.append((p, texts))
    return x, y, x + w, y + h


def panel_set(ax, top: float, sel_x: float) -> float:
    h = 2.70
    x0, x1 = 0.03, W - 0.03
    rbox(ax, x0, top, x1 - x0, h, "#FAFAFA", EDGE["set"], lw=1.1, r=0.06)
    tag(ax, x0 + 0.07, top + 0.13, "2  Where the retained set is fixed",
        EDGE["set"])
    # the dense reference, on the header line
    rx0 = x0 + 2.55
    p = rbox(ax, rx0, top + 0.05, x1 - 0.10 - rx0, 0.18, "white", "#9A9A9A",
             lw=0.9, z=3)
    tt = [ax.text(rx0 + 0.08, top + 0.14, "dense reference:", ha="left",
                  va="center", fontsize=T_TITLE, color=INK, fontweight="medium",
                  zorder=4),
          ax.text(rx0 + 1.05, top + 0.14, "all $N^2$ scores, aggregate over $N$",
                  ha="left", va="center", fontsize=T_SUB, color=MUTED, zorder=4),
          ax.text(x1 - 0.18, top + 0.14, "$C_{\\mathrm{dense}} = 2N^2d$",
                  ha="right", va="center", fontsize=T_TITLE, color=INK,
                  zorder=4)]
    _CONTAINED.append((p, tt))

    sub_top, sub_bot = top + 0.32, top + h - 0.30
    mid = (x0 + x1) / 2
    ax_, aw = x0 + 0.10, mid - 0.06 - (x0 + 0.10)
    bx_, bw_ = mid + 0.06, x1 - 0.10 - (mid + 0.06)
    rbox(ax, ax_, sub_top, aw, sub_bot - sub_top, "#FDF6EE", EDGE["A"], lw=1.0,
         r=0.05, z=2)
    rbox(ax, bx_, sub_top, bw_, sub_bot - sub_top, "#EFF7F3", EDGE["B"], lw=1.0,
         r=0.05, z=2)
    tag(ax, ax_ + 0.06, sub_top + 0.13, "A  score, then select", EDGE["A"])
    tag(ax, bx_ + 0.06, sub_top + 0.13, "B  select, then score", EDGE["B"])
    ry = sub_top + 0.40

    # --- A: score, then select
    xs, ws = row(ax_ + 0.10, ax_ + aw - 0.10, (0.70, 0.64, 1.50))
    sc = node(ax, xs[0], ry, ws[0], 0.56, ("all $N^2$", "scores"),
              ("$s_i = q_i K^{\\top}$", "for every pair"), face=TINT["score"],
              edge=ROLE["score"])
    tk = node(ax, xs[1], ry, ws[1], 0.56, "top-$k$ of $s_i$",
              ("ranked after", "scoring"), face=TINT["select"],
              edge=ROLE["select"])
    ks = stack(ax, xs[2], ry - 0.10, ws[2], "one function, three kernels",
               [("masked", "discarded weights as zeros", TINT["aggregate"],
                 ROLE["aggregate"]),
                ("gathered", "writes, re-reads $(B, N, k, d)$", TINT["memory"],
                 ROLE["memory"]),
                ("fused", "score row held on chip", TINT["aggregate"],
                 ROLE["aggregate"])],
               "white", "#B5B5B5", None, None)
    m = ry + 0.28
    arrow(ax, sc[2], m, tk[0], m)
    arrow(ax, tk[2], m, ks[0], m)
    yb = max(ks[3], sc[3], tk[3]) + 0.07
    fy = yb + 0.25
    ax.text(ax_ + aw / 2, fy,
            "$C_{\\mathrm{top}\\text{-}k} = N^2d + Nkd$,   "
            "$C_{\\mathrm{dense}} / C_{\\mathrm{top}\\text{-}k} = 2N/(N{+}k) < 2$",
            ha="center", va="center", fontsize=T_SUB + 0.3, color=INK)
    ax.text(ax_ + aw / 2, fy + 0.14, "gather byte screen: $kd \\leq N$",
            ha="center", va="center", fontsize=T_SUB, color=MUTED)

    # --- B: select, then score
    xs, ws = row(bx_ + 0.10, bx_ + bw_ - 0.10, (0.86, 1.10, 0.86))
    po = node(ax, xs[0], ry, ws[0], 0.56, ("agent", "positions"),
              ("$(B, N, 2)$;", "no score formed"), face=TINT["grey"],
              edge=EDGE["grey"])
    se = stack(ax, xs[1], ry - 0.10, ws[1], "selector, cost $C_{\\mathrm{sel}}$",
               [("all-pairs kNN", "$\\Theta(N^2)$, capped", None, None),
                ("grid kNN", "$\\Theta(Nk)$ + one sort", None, None),
                ("Z-order window", "$O(N \\log N)$ sort", None, None)],
               TINT["select"], ROLE["select"], "white", ROLE["select"])
    fx = node(ax, xs[2], ry, ws[2], 0.56, ("fixed-set fused", "attention"),
              ("$Nk$ scores +", "$Nk$ aggregate"), face=TINT["aggregate"],
              edge=ROLE["aggregate"])
    arrow(ax, po[2], m, se[0], m)
    arrow(ax, se[2], m, fx[0], m)
    ax.text(bx_ + bw_ / 2, fy,
            "$S = 2N^2d\\,/\\,(2Nkd + C_{\\mathrm{sel}})$   "
            "$\\rightarrow N/k$ only if $C_{\\mathrm{sel}} = o(Nkd)$",
            ha="center", va="center", fontsize=T_SUB + 0.3, color=INK)
    ax.text(bx_ + bw_ / 2, fy + 0.14, "selector indices $(B, N, k)$ feed the "
            "fused kernel", ha="center", va="center", fontsize=T_SUB,
            color=MUTED)

    # the fixed-set comparison: A's exact top-k set handed to B's fused kernel,
    # routed below both rows
    xt, xf = (tk[0] + tk[2]) / 2, (fx[0] + fx[2]) / 2
    path(ax, [(xt, tk[3]), (xt, yb), (xf, yb), (xf, fx[3])],
         color=ROLE["select"], ls=(0, (3, 2)))
    ax.text((bx_ + xf) / 2 + 0.25, yb + 0.085, "fixed-set comparison: A's exact "
            "top-$k$ set, given free", ha="center", va="center",
            fontsize=T_SUB, color=EDGE["A"], style="italic", zorder=7)

    # role legend along the bottom of the panel
    ly = top + h - 0.15
    lx = x0 + 0.17
    for lab, face, edge in (("scores", TINT["score"], ROLE["score"]),
                            ("select", TINT["select"], ROLE["select"]),
                            ("aggregate", TINT["aggregate"], ROLE["aggregate"]),
                            ("intermediate written, re-read", TINT["memory"],
                             ROLE["memory"])):
        rbox(ax, lx, ly - 0.055, 0.15, 0.11, face, edge, lw=0.8, r=0.02, z=3)
        ax.text(lx + 0.21, ly, lab, ha="left", va="center", fontsize=T_SUB,
                color=MUTED)
        lx += 0.36 + 0.052 * len(lab)
    # the entry from panel 1's peer-selection step
    arrow(ax, sel_x, top - 0.11, sel_x, top + 0.005, color=ROLE["select"],
          ls=(0, (3, 2)))
    return top + h


# --------------------------------------------------------------------------
# panel 3: the measurement protocol, left to right
# --------------------------------------------------------------------------
def panel_measurement(ax, top: float) -> float:
    h = 1.28
    x0, x1 = 0.03, W - 0.03
    rbox(ax, x0, top, x1 - x0, h, "#F9F5FB", EDGE["measure"], lw=1.1, r=0.06)
    tag(ax, x0 + 0.07, top + 0.13, "3  Measurement", EDGE["measure"])
    ax.text(x0 + 1.25, top + 0.13, "every implementation of 2, plus dense, at "
            "each $(N, d, k)$", ha="left", va="center", fontsize=T_SUB + 0.3,
            color=MUTED)
    arrow(ax, W / 2, top - 0.11, W / 2, top + 0.005, color=EDGE["measure"])
    ry, rh = top + 0.34, 0.80
    xs, ws = row(x0 + 0.12, x1 - 0.12, (0.95, 0.92, 1.0, 1.0, 0.92, 1.0))
    # the four GPUs, one per line
    gx, gw = xs[0], ws[0]
    p = rbox(ax, gx, ry, gw, rh, "white", EDGE["measure"], lw=1.0, z=3)
    t = ax.text(gx + gw / 2, ry + 0.11, "four NVIDIA GPUs", ha="center",
                va="center", fontsize=T_TITLE, color=INK, fontweight="medium",
                zorder=4)
    _CONTAINED.append((p, [t]))
    chh = 0.125
    for i, name in enumerate(("RTX PRO 6000", "A100", "H100", "H200")):
        yy = ry + 0.205 + i * (chh + 0.022)
        q = rbox(ax, gx + 0.06, yy, gw - 0.12, chh, TINT["measure"],
                 EDGE["measure"], lw=0.8, r=0.03, z=4)
        _CONTAINED.append((q, [ax.text(gx + gw / 2, yy + chh / 2, name,
                                       ha="center", va="center",
                                       fontsize=T_SUB, color=INK, zorder=5)]))
    steps = [("equivalence", ("vs. the reference", "at every reported",
                              "configuration")),
             ("best dense", ("SDPA and Triton", "dense, tuned on",
                             "each GPU; + Flex-", "Attention in reduced",
                             "precision")),
             (("interleaved", "timing"), ("CUDA events or", "CUDA-graph replay;",
                                          "min over samples,",
                                          "repeated blocks")),
             ("artifacts", ("raw samples, ratio", "to best dense;",
                            "Nsight counters", "(H100, H200)")),
             (("tables, figures,", "numbers"), ("generated from the",
                                                "artifacts, checked",
                                                "against a manifest"))]
    boxes = [(gx, ry, gx + gw, ry + rh)]
    for (tt, s), x, w in zip(steps, xs[1:], ws[1:]):
        boxes.append(node(ax, x, ry, w, rh, tt, s, face=TINT["measure"],
                          edge=EDGE["measure"]))
    mid = ry + rh / 2
    for (_, _, r0, _), (l1, _, _, _) in zip(boxes, boxes[1:]):
        arrow(ax, r0, mid, l1, mid)
    return top + h


def check_contained(fig) -> list[str]:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for patch, texts in _CONTAINED:
        pb = patch.get_window_extent(r)
        for t in texts:
            tb = t.get_window_extent(r)
            if tb.x0 < pb.x0 - 0.5 or tb.x1 > pb.x1 + 0.5 or \
               tb.y0 < pb.y0 - 0.5 or tb.y1 > pb.y1 + 0.5:
                bad.append(t.get_text())
    return bad


def main(png: bool = False) -> int:
    gap = 0.22
    h1, h2, h3 = 1.24, 2.70, 1.28
    H = 0.03 + h1 + gap + h2 + gap + h3 + 0.03
    fig, ax = setup(H)
    b1, sel_x = panel_communication(ax, 0.03)
    b2 = panel_set(ax, b1 + gap, sel_x)
    panel_measurement(ax, b2 + gap)
    bad = fs.check_min_font(fig, MIN_PT)
    if bad:
        raise ValueError(f"text below {MIN_PT} pt: {bad}")
    over = check_contained(fig)
    if over:
        raise ValueError(f"text overflows its box: {over}")
    fig.savefig(OUT)
    if png:
        fig.savefig(OUT.with_suffix(".png"), dpi=220)
    print(f"wrote {OUT} ({W:.2f} x {H:.2f} in)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(png="--png" in sys.argv))
