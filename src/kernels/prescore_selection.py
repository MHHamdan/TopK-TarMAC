"""Attention whose retained set is fixed before any attention score exists.

Proposition 2 of the manuscript states that a selector which fixes the retained
set without forming the complete score vector is not subject to the two-fold
ceiling, and that its operation-count ratio against dense attention approaches
N/k when the selector costs o(Nkd). That is an operation-count statement. This
module supplies the implementations needed to ask the separate empirical
question: whether the removed arithmetic appears as removed time.

Two selectors are provided, chosen because both correspond to mechanisms already
used in multi-agent communication rather than invented for the demonstration.

``radius_knn``
    Each agent retains its k spatially nearest peers, computed from
    two-dimensional positions. This is the rule underlying distance-based
    neighbourhood communication. Its selector is still pairwise -- 2N^2
    operations for the distances -- but pairwise in the position dimension
    rather than the head dimension, so it costs a factor of d/2 less than
    forming the scores. It escapes the ceiling because attention scores are then
    evaluated for the retained pairs only.

``morton_window``
    Agents are ordered along a Morton (Z-order) space-filling curve and each
    agent retains the k agents adjacent to it in that order. Sorting is
    O(N log N), so the selector is genuinely sub-quadratic and satisfies the
    o(Nkd) condition of Proposition 2 case (i) for any fixed k. Spatial locality
    along a Morton curve is approximate, which is the price paid for the cost.

Both share the same downstream path: gather the k retained keys and values,
evaluate the k scores per query, softmax over them, and contract. The score
computation is (B*N, 1, d) x (B*N, d, k), which is Nk rather than N^2 pairs --
the term Proposition 1 says a score-then-select rule cannot remove.

The selection stage is exposed separately from the attention stage so that the
selector cost can be charged rather than assumed away.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# selectors: positions -> retained peer indices, before any score exists
# ---------------------------------------------------------------------------

def select_radius_knn(pos: torch.Tensor, k: int) -> torch.Tensor:
    """k spatially nearest peers per agent. pos: (B, N, 2) -> (B, N, k) int64."""
    B, N, _ = pos.shape
    d2 = torch.cdist(pos, pos, p=2)
    d2 = d2.masked_fill(
        torch.eye(N, device=pos.device, dtype=torch.bool).unsqueeze(0),
        float("inf"))
    return d2.topk(k, dim=-1, largest=False, sorted=False).indices


#: Bit-spreading constants for the standard branch-free Morton interleave.
#: Spreading ten bits with five shift-and-mask steps replaces a per-bit loop
#: that would issue four elementwise kernels per bit; on the shapes measured
#: here that loop cost more than the sort it feeds, which made an earlier
#: version of this module measure its own launch overhead rather than the
#: selector.
_SPREAD = ((16, 0x0000FFFF), (8, 0x00FF00FF), (4, 0x0F0F0F0F),
           (2, 0x33333333), (1, 0x55555555))


def _spread_bits(v: torch.Tensor) -> torch.Tensor:
    """Insert a zero between consecutive bits of a non-negative integer."""
    for shift, mask in _SPREAD:
        v = (v | (v << shift)) & mask
    return v


def _morton_code(pos: torch.Tensor, bits: int = 10) -> torch.Tensor:
    """Interleave the bits of quantised 2-D coordinates. pos: (B, N, 2)."""
    lo = pos.amin(dim=1, keepdim=True)
    hi = pos.amax(dim=1, keepdim=True)
    span = (hi - lo).clamp_min(1e-6)
    grid = ((pos - lo) / span * (2 ** bits - 1)).long().clamp_(0, 2 ** bits - 1)
    return _spread_bits(grid[..., 0]) | (_spread_bits(grid[..., 1]) << 1)


def select_morton_window(pos: torch.Tensor, k: int) -> torch.Tensor:
    """k peers adjacent along a Z-order curve. Sub-quadratic in N."""
    B, N, _ = pos.shape
    code = _morton_code(pos)
    order = code.argsort(dim=-1)                      # (B, N) sorted positions
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(N, device=pos.device)
                  .expand(B, N))
    # Offsets +-1, +-2, ... around the agent's own rank, wrapping at the ends.
    half = (k + 1) // 2
    offs = torch.cat([torch.arange(1, half + 1, device=pos.device),
                      -torch.arange(1, k - half + 1, device=pos.device)])
    neigh_rank = (rank.unsqueeze(-1) + offs.view(1, 1, -1)) % N   # (B, N, k)
    return order.gather(1, neigh_rank.reshape(B, N * k)).view(B, N, k)


SELECTORS = {"radius_knn": select_radius_knn,
             "morton_window": select_morton_window}


# ---------------------------------------------------------------------------
# attention over a pre-selected peer set
# ---------------------------------------------------------------------------

def prescore_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       idx: torch.Tensor) -> torch.Tensor:
    """Score only the retained pairs. q,k,v: (B,N,d); idx: (B,N,kk)."""
    B, N, d = q.shape
    kk = idx.shape[-1]
    flat = idx.reshape(B, N * kk, 1).expand(B, N * kk, d)
    k_g = k.gather(1, flat).view(B, N, kk, d)
    v_g = v.gather(1, flat).view(B, N, kk, d)
    # (B,N,1,d) x (B,N,d,kk) -> (B,N,1,kk): Nk scored pairs, not N^2.
    scores = torch.matmul(q.unsqueeze(-2), k_g.transpose(-1, -2)) / (d ** 0.5)
    w = F.softmax(scores, dim=-1)
    return torch.matmul(w, v_g).squeeze(-2)


def prescore_full(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  pos: torch.Tensor, kk: int, selector: str) -> torch.Tensor:
    return prescore_attention(q, k, v, SELECTORS[selector](pos, kk))


def analytic_flops(selector: str, B: int, N: int, d: int, kk: int) -> dict:
    """Multiply--accumulates x2, split so the selector is charged separately.

    The dense reference is 4BN^2d. A score-then-select top-k path is
    2BN^2d + 2BNkd and so cannot fall below half the dense count; the pre-score
    paths below replace the N^2 score term with an Nk one and pay a selector
    instead, which is the substitution Proposition 2 describes.
    """
    if selector == "radius_knn":
        # squared euclidean over 2-D positions: 3 ops per pair, plus the top-k
        sel = 3 * B * N * N
    elif selector == "morton_window":
        # quantise and interleave 10 bit-pairs, then sort: O(N log N) compares
        sel = B * N * (40 + 2 * max(1, N.bit_length()))
    else:
        raise ValueError(selector)
    return {"selector": sel,
            "scores": 2 * B * N * kk * d,
            "aggregate": 2 * B * N * kk * d,
            "attention_total": 4 * B * N * kk * d,
            "total": sel + 4 * B * N * kk * d,
            "dense_total": 4 * B * N * N * d}
