"""Render the Figure-1 schematic of TopK-TarMAC.

Output: paper/figures/fig1_schematic.pdf

Pure matplotlib (no TikZ) so the build does not need LaTeX-specific extras.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures" / "fig1_schematic.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)


def box(ax, xy, w, h, label, fc="#E0E7FF", ec="black", **kw):
    ax.add_patch(patches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=fc, edgecolor=ec, lw=1.2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, label,
            ha="center", va="center", fontsize=9, **kw)


def arrow(ax, x0, y0, x1, y1, **kw):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=1.2, **kw))


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    # Observations
    for i, y in enumerate([0.5, 1.6, 2.7, 3.8]):
        box(ax, (0.1, y), 1.0, 0.6, f"$o_{{i={i+1}}}$", fc="#FFF4D6")
    ax.text(0.6, 4.65, "Per-agent\nobservations", ha="center", fontsize=8)

    # Obs embedder
    box(ax, (1.5, 1.6), 1.2, 1.8, "Shared\nLinear", fc="#E0E7FF")

    # Q/K/V projection
    box(ax, (3.1, 1.6), 1.2, 1.8, "$W_{qkv}$\nLinear", fc="#E0E7FF")

    # Attention soft scores
    box(ax, (4.7, 1.6), 1.5, 1.8, r"softmax$(QK^\top / \sqrt{d})$" "\n(N x N)", fc="#FDE2E2")
    ax.text(5.45, 4.65, "Full soft\nattention", ha="center", fontsize=8)

    # k-gate
    box(ax, (4.7, 3.9), 1.5, 0.8, "$k$-gate (Gumbel)\nadaptive $k_i$", fc="#D6F5DC")
    arrow(ax, 5.45, 3.9, 5.45, 3.4)  # gate -> attention

    # Top-k mask + renormalize
    box(ax, (6.5, 1.6), 1.5, 1.8, r"top-$k$ mask + renorm" "\n($N \\times k$)",
        fc="#FAE3B0")
    arrow(ax, 6.2, 2.5, 6.5, 2.5)
    arrow(ax, 4.3, 2.5, 4.7, 2.5)
    arrow(ax, 2.7, 2.5, 3.1, 2.5)

    # Message
    box(ax, (8.3, 2.0), 1.5, 1.0, r"$m_i = \sum_{j \in T_i} \tilde\alpha_{ij} v_j$",
        fc="#E0E7FF")
    arrow(ax, 8.0, 2.5, 8.3, 2.5)

    # Actor + critic chain
    box(ax, (8.3, 0.5), 1.5, 1.0, "Actor\n$\\pi(a_i | o_i, m_i)$", fc="#E0E7FF")
    arrow(ax, 9.05, 2.0, 9.05, 1.5)
    arrow(ax, 1.1, 1.9, 8.3, 1.0, connectionstyle="arc3,rad=-0.2")  # obs to actor

    # Title
    ax.text(5.0, 4.55, "TopK-TarMAC", fontsize=12, weight="bold", ha="center")

    # Legend: highlight novel parts
    ax.add_patch(patches.Rectangle((0.05, 0.05), 9.9, 4.85, fill=False,
                                   linestyle="dashed", lw=0.8,
                                   edgecolor="#888888"))

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
