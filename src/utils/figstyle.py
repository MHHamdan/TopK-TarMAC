"""One visual language for every figure in the paper.

Three rules this module exists to enforce, each of which was violated by the
figure set it replaces.

1. **Colour never carries a distinction on its own.** Every implementation has a
   colour, a marker and a dash pattern, and a mark type without a marker or dash
   channel -- a bar segment, a heatmap cell -- gets a hatch or a monotone
   luminance ordering instead. The figures then survive greyscale printing and
   the common colour-vision deficiencies.
2. **Implementation colours are reserved.** The six hues below encode
   implementations and nothing else. Stage decompositions and sequential
   magnitudes draw from separate ramps, so a reader who has learned that blue
   means dense is never shown a blue that means something else.
3. **Figures are generated at the width they are included at.** The scale factor
   is always one, so a point of text is a point of text in every figure.

Nothing here uses LaTeX markup: figures render with ``text.usetex`` off, so a
string such as ``\\texttt{...}`` would print its own backslashes. Use mathtext
for symbols.
"""

from __future__ import annotations

from typing import Final

import matplotlib as mpl
import numpy as np

# --- implementation identity ------------------------------------------------

IMPL_COLOR: Final[dict[str, str]] = {
    "dense_unfused": "#0072B2",   # blue
    "dense_fused": "#7B3294",     # purple
    "dense_triton": "#4A4A4A",    # dark grey: the matched-effort control
    "topk_masked": "#D55E00",     # vermillion
    "topk_gather": "#009E73",     # green
    "topk_fused": "#00707F",      # teal, kin to gathered: same algorithm
    "topk_fused_gather": "#00707F",
    "topk_fused_masked": "#B35A00",   # kin to masked, for the same reason
    "adaptive": "#8C6D1F",
    "random_k": "#999999",
    "radius_knn": "#B3446C",
    "morton_window": "#E69F00",
}

IMPL_MARKER: Final[dict[str, str]] = {
    "dense_unfused": "o",
    "dense_fused": "D",
    "dense_triton": "*",
    "topk_masked": "v",
    "topk_gather": "s",
    "topk_fused": "^",
    "topk_fused_gather": "^",
    "topk_fused_masked": "P",
    "adaptive": "X",
    "random_k": ">",
    "radius_knn": "<",
    "morton_window": "h",
}

IMPL_DASH: Final[dict[str, tuple]] = {
    "dense_unfused": (0, ()),
    "dense_fused": (0, (5, 1.5, 1, 1.5)),
    "dense_triton": (0, (1, 1)),
    "topk_masked": (0, (4, 2)),
    "topk_gather": (0, (6, 2)),
    "topk_fused": (0, (3, 1, 1, 1)),
    "topk_fused_gather": (0, (3, 1, 1, 1)),
    "topk_fused_masked": (0, (2, 1)),
    "adaptive": (0, (1, 1.5)),
    "random_k": (0, (1, 2)),
    "radius_knn": (0, (5, 2)),
    "morton_window": (0, (7, 2, 1, 2)),
}

IMPL_LABEL: Final[dict[str, str]] = {
    "dense_unfused": "dense, unfused",
    "dense_fused": "dense, fused (library)",
    "dense_triton": "dense, fused (ours)",
    "topk_masked": "top-k, masked",
    "topk_gather": "top-k, gathered",
    "topk_fused": "top-k, fused",
    "topk_fused_gather": "top-k, fused + gathered",
    "topk_fused_masked": "top-k, fused + masked",
    "adaptive": "learned gate",
    "random_k": "random-k",
    "radius_knn": "pre-score, nearest",
    "morton_window": "pre-score, Z-order",
}

# --- non-identity ramps -----------------------------------------------------

#: Monotone-luminance ramp for part-of-whole bars. Every pair differs by at
#: least 0.12 in relative luminance, adjacent or not, so a stack is readable in
#: greyscale where colour alone would fail.
STACK_RAMP: Final[list[str]] = [
    "#10243a", "#254c73", "#3f77a8", "#6a9fc7", "#9cc0dc", "#c6dcec", "#e6f0f7",
]

#: Sequential single hue for heatmaps, anchored at the meaningful floor.
HEAT_CMAP: Final[str] = "Blues"

RULE_COLOR: Final[str] = "#6B6B6B"
ANNOT_COLOR: Final[str] = "#1A1A1A"
MUTED_COLOR: Final[str] = "#6B6B6B"
ABSENT_FACE: Final[str] = "#F6F6F6"
ABSENT_EDGE: Final[str] = "#BDBDBD"

#: The one width figures are generated at, matching the TMLR text block.
TEXT_WIDTH_IN: Final[float] = 6.5

#: Nothing in a figure is set below this, in points, after inclusion.
MIN_PT: Final[float] = 7.0


def apply(base: float = 8.5) -> None:
    """Install the paper-wide style. ``base`` is the axis-label size in points."""
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans 3", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": base,
        "axes.titlesize": base + 0.5,
        "axes.labelsize": base,
        "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5,
        "legend.fontsize": MIN_PT + 0.5,
        "figure.titlesize": base + 0.5,
        "axes.grid": True,
        "grid.color": "#D8D8D8",
        "grid.alpha": 1.0,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "legend.handlelength": 2.6,
        "figure.dpi": 150,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style(impl: str) -> dict:
    """Line keyword arguments for one implementation: colour, marker, dashes."""
    return {"color": IMPL_COLOR.get(impl, "#333333"),
            "marker": IMPL_MARKER.get(impl, "o"),
            "linestyle": IMPL_DASH.get(impl, (0, ())),
            "label": IMPL_LABEL.get(impl, impl)}


def parity_line(ax, y: float = 1.0, label: str | None = None) -> None:
    """The reference at which two implementations cost the same."""
    ax.axhline(y, color=RULE_COLOR, linestyle=(0, (3, 2)), linewidth=1.0,
               zorder=1)
    if label:
        ax.annotate(label, xy=(0.99, y), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=MIN_PT,
                    color=RULE_COLOR)


def text_on(facecolor) -> str:
    """Black or white, whichever has more contrast against a cell's fill."""
    r, g, b = mpl.colors.to_rgb(facecolor)
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "#FFFFFF" if (1.05 / (lum + 0.05)) > ((lum + 0.05) / 0.05) else "#000000"


def check_min_font(fig, floor: float = MIN_PT) -> list[str]:
    """Names of any text artists set below the floor. Empty list is the goal."""
    bad = []
    for t in fig.findobj(mpl.text.Text):
        s = t.get_fontsize()
        if t.get_text().strip() and s and s < floor - 1e-6:
            bad.append(f"{t.get_text()[:30]!r} at {s:.1f} pt")
    return bad


def savefig(fig, path, width_in: float | None = None) -> None:
    """Write a vector PDF, refusing to emit text below the legibility floor."""
    if width_in is not None:
        assert abs(fig.get_size_inches()[0] - width_in) < 1e-6, (
            f"figure is {fig.get_size_inches()[0]:.2f} in but will be included "
            f"at {width_in:.2f} in; generate at the inclusion width")
    bad = check_min_font(fig)
    if bad:
        raise ValueError(f"text below {MIN_PT} pt: {bad}")
    fig.savefig(path)


def log2_ticks(ax, values) -> None:
    """Tick a log2 x-axis at exactly the measured values."""
    vals = sorted(set(values))
    ax.set_xscale("log", base=2)
    ax.set_xticks(vals)
    ax.set_xticklabels([str(v) for v in vals])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def shade_below(ax, y: float = 1.0, label: str | None = None) -> None:
    """Shade the region in which the sparse implementation would be winning."""
    ax.axhspan(ax.get_ylim()[0], y, color="#EAF2FA", zorder=0)
    if label:
        ax.annotate(label, xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=MIN_PT, color=MUTED_COLOR, ha="left", va="bottom")


def geo_mean(xs) -> float:
    a = np.asarray([x for x in xs if x and np.isfinite(x) and x > 0])
    return float(np.exp(np.mean(np.log(a)))) if a.size else float("nan")
