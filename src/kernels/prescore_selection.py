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


def grid_knn_shape(N: int, k: int) -> tuple[int, int]:
    """Grid side G and per-cell capacity M for select_grid_knn.

    Cells are sized for about k agents each, so the k-th nearest neighbour lies
    within one cell width with high probability. The capacity allows three
    standard deviations of Poisson occupancy above the mean; agents beyond it
    are left out of the candidate set, which is what makes the selector
    approximate, and the benchmark measures how often that changes the
    retained set.
    """
    occ = max(2, k)
    G = max(1, int((N / occ) ** 0.5))
    mean = N / (G * G)
    M = int(mean + 3 * mean ** 0.5 + 2)
    return G, M


_GRID_CACHE: dict = {}


def select_grid_knn(pos: torch.Tensor, k: int) -> torch.Tensor:
    """k nearest peers among the 3x3 grid cells around each agent.

    A uniform grid replaces the all-pairs distance matrix. Agents are sorted
    once by cell, with cell id x*G + y, so the three cells (x', y-1..y+1) of
    each neighbouring column x' are contiguous in sorted order: each agent's
    candidates are three contiguous runs of at most 3M sorted agents, and it
    ranks only those. For positions of bounded density that is Theta(Nk)
    candidates rather than N^2, which moves the selector from Corollary 1(b)
    to linear cost. It is exact whenever the k-th nearest neighbour is no
    farther than the nearest edge of the 3x3 block and no cell overflowed;
    ``grid_knn_recall`` measures how often the retained set matches exact kNN.
    Where the neighbourhood would hold N or more candidates the grid saves
    nothing, and the exact all-pairs selector is used instead.
    """
    B, N, _ = pos.shape
    G, M = grid_knn_shape(N, k)
    L = 3 * M
    if 3 * L >= N:
        return select_radius_knn(pos, k)
    dev = pos.device
    key = (G, M, N, str(dev))
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = (torch.arange(N, device=dev),
                            torch.arange(L, device=dev),
                            torch.arange(-1, 2, device=dev))
    ar_n, ar_l, off = _GRID_CACHE[key]
    C = G * G
    lo = pos.amin(dim=1, keepdim=True)
    span = (pos.amax(dim=1, keepdim=True) - lo).clamp_min(1e-6)
    cxy = ((pos - lo) / span * G).long().clamp_(0, G - 1)          # (B,N,2)
    cell = cxy[..., 0] * G + cxy[..., 1]                            # (B,N)
    order = cell.argsort(dim=1)                                     # (B,N)
    # start[c] = first sorted position of cell c, for c in 0..C (C = end).
    counts = torch.zeros(B, C + 1, device=dev, dtype=torch.long)
    counts.scatter_add_(1, cell + 1, torch.ones_like(cell))
    start = counts.cumsum(1)                                        # (B,C+1)

    # For neighbour column x' = x-1..x+1: run from cell (x', max(y-1,0)) to
    # the end of cell (x', min(y+1,G-1)).
    x = cxy[..., 0].unsqueeze(-1) + off                             # (B,N,3)
    y = cxy[..., 1].unsqueeze(-1)
    ok_col = (x >= 0) & (x < G)
    xc = x.clamp(0, G - 1)
    c_lo = xc * G + (y - 1).clamp_min(0)
    c_hi = xc * G + (y + 1).clamp_max(G - 1) + 1
    s_lo = start.gather(1, c_lo.view(B, -1)).view(B, N, 3)
    s_hi = start.gather(1, c_hi.view(B, -1)).view(B, N, 3)
    s_hi = torch.minimum(s_hi, s_lo + L)                            # capacity
    idx = s_lo.unsqueeze(-1) + ar_l                                  # (B,N,3,L)
    valid = (idx < s_hi.unsqueeze(-1)) & ok_col.unsqueeze(-1)
    cand = order.gather(1, idx.clamp_max(N - 1).view(B, -1)).view(B, N, 3 * L)
    valid = valid.view(B, N, 3 * L) & (cand != ar_n.view(1, N, 1))
    pc = pos.gather(1, cand.view(B, -1, 1).expand(B, -1, 2)).view(B, N, 3 * L, 2)
    d2 = (pc - pos.unsqueeze(2)).square().sum(-1).masked_fill(~valid,
                                                              float("inf"))
    j = d2.topk(k, dim=-1, largest=False, sorted=False).indices
    return cand.gather(2, j)


def grid_knn_recall(pos: torch.Tensor, k: int) -> float:
    """Fraction of retained peers that exact kNN would also retain."""
    got = select_grid_knn(pos, k)
    ref = select_radius_knn(pos, k)
    hit = (got.unsqueeze(-1) == ref.unsqueeze(-2)).any(-1)
    return hit.float().mean().item()


SELECTORS = {"radius_knn": select_radius_knn,
             "grid_knn": select_grid_knn,
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
        # Euclidean distance over 2-D positions is 3 operations per pair, and
        # the selection is a top-k over the resulting N x N matrix. The top-k
        # performs comparisons rather than multiply-adds, so counting it in the
        # same total is a convention; we charge one comparison per candidate,
        # which is a lower bound for any selection algorithm and keeps the
        # selector from being credited with work it does not do.
        sel = 3 * B * N * N + B * N * N
    elif selector == "grid_knn":
        # One sort to bucket agents by cell, then per agent 9M candidates each
        # charged a 2-D distance (3 operations) and one comparison, with M the
        # per-cell capacity: Theta(Nk) at bounded density.
        _, M = grid_knn_shape(N, kk)
        if 9 * M >= N:          # falls back to the all-pairs selector
            sel = 3 * B * N * N + B * N * N
        else:
            sel = B * N * (4 * 9 * M + 2 * max(1, N.bit_length()))
    elif selector == "morton_window":
        # Quantise, interleave ten bit-pairs with five shift-and-mask steps,
        # then sort: O(N log N) comparisons. Comparisons and multiply-adds are
        # counted in one total here, which is a convention and not a physical
        # equivalence; the manuscript labels this column as an analytic model.
        sel = B * N * (30 + 2 * max(1, N.bit_length()))
    else:
        raise ValueError(selector)
    return {"selector": sel,
            "scores": 2 * B * N * kk * d,
            "aggregate": 2 * B * N * kk * d,
            "attention_total": 4 * B * N * kk * d,
            "total": sel + 4 * B * N * kk * d,
            "dense_total": 4 * B * N * N * d}
