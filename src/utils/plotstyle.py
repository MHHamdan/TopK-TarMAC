"""Shared matplotlib styling for every paper figure.

Two rules this module exists to enforce:

1. **No LaTeX macros in matplotlib text.** Figures are rendered with
   ``text.usetex = False``, so a string like ``r"\\texttt{simple\\_spread}"``
   renders its backslashes and braces literally. Use :func:`env_label` for
   environment names and mathtext (``$N=6$``) for symbols.
2. **Colourblind-safe categorical colours.** ``tab10`` is not safe; the
   Okabe--Ito palette is.

Figures are written as vector PDF so they scale cleanly in a two-column
journal layout.
"""

from __future__ import annotations

from typing import Final

import matplotlib as mpl

#: Okabe--Ito colourblind-safe qualitative palette (Okabe & Ito 2008).
#: Ordered so the first three are maximally separable for the common
#: baseline / dense / sparse comparison.
OKABE_ITO: Final[list[str]] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

#: Stable colour per method arm, so a given arm keeps its colour across every
#: figure in the paper.
ARM_COLORS: Final[dict[str, str]] = {
    "mappo": OKABE_ITO[7],
    "dense": OKABE_ITO[0],
    "tarmac": OKABE_ITO[5],
    "topk_fixed": OKABE_ITO[2],
    "topk_adaptive": OKABE_ITO[1],
    "lagrangian": OKABE_ITO[3],
    "entmax": OKABE_ITO[4],
    "distance": OKABE_ITO[6],
    "random_k": "#666666",
}

#: Marker per arm, so figures stay readable in greyscale print.
ARM_MARKERS: Final[dict[str, str]] = {
    "mappo": "o",
    "dense": "s",
    "tarmac": "D",
    "topk_fixed": "^",
    "topk_adaptive": "v",
    "lagrangian": "P",
    "entmax": "X",
    "distance": "<",
    "random_k": ">",
}


def apply_style(base_font: float = 9.0) -> None:
    """Install the paper-wide matplotlib style.

    Args:
        base_font: Body font size in points. 9 pt suits a two-column
            IEEEtran layout at ``\\linewidth`` figure width.
    """
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": base_font,
        "axes.titlesize": base_font + 1,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        "legend.fontsize": base_font - 1,
        "figure.titlesize": base_font + 1,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "figure.dpi": 150,
        # Vector output; embed fonts as Type 42 so journals accept the PDF.
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def env_label(env_id: str) -> str:
    """Human-readable environment name safe for a matplotlib label.

    Never returns LaTeX markup. ``simple_spread_v3`` becomes
    ``"MPE simple_spread"`` -- the underscore is literal and renders fine
    outside mathtext.

    Args:
        env_id: Environment identifier, e.g. ``"simple_spread_v3"``.

    Returns:
        A display string containing no backslashes or brace groups.
    """
    stem = env_id.removesuffix("_v3").removesuffix("_v4").removesuffix("_v2")
    family = "MPE" if stem.startswith("simple") else ""
    return f"{family} {stem}".strip()


def arm_style(arm: str) -> dict[str, str]:
    """Colour and marker keyword arguments for a method arm.

    Args:
        arm: Arm key, e.g. ``"topk_adaptive"``.

    Returns:
        Kwargs suitable for splatting into a matplotlib plotting call.
    """
    return {
        "color": ARM_COLORS.get(arm, OKABE_ITO[0]),
        "marker": ARM_MARKERS.get(arm, "o"),
    }
