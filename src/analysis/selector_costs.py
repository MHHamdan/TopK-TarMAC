"""Partial scalar ledger for the measured pre-score selector expressions.

This is not an executed FLOP/instruction counter. Sorting, top-k, indexing,
allocation and traffic remain uncharged; their nonnegative cost is explicit.
See Appendix A.4 of the manuscript. Historical JSON counts
are immutable and must not be mistaken for this versioned analysis.
"""
from __future__ import annotations

from math import isqrt, sqrt

MODEL_VERSION = "partial_scalar_ledger_v2"


def direct_distance_ops(p: int, *, squared: bool = False) -> int:
    """Subtract, square, reduce; optionally take a square root."""
    if p < 1:
        raise ValueError("p must be positive")
    return 3 * p - 1 + int(not squared)


def pair_ledger(N: int, p: int = 2) -> dict[str, int]:
    if N < 2 or p < 1:
        raise ValueError("N >= 2 and p >= 1 required")
    if N <= 25:
        return {"direct_distance_expression": N * N * direct_distance_ops(p)}
    # PyTorch v2.11 Distance.cpp::_euclidean_dist (default route for N>25).
    return {"two_norms": 2 * N * (2 * p - 1),
            "scale_one_input": N * p,
            "augmented_product": 2 * (p + 2) * N * N,
            "clamp_and_sqrt": 2 * N * N}


def morton_ledger(N: int, k: int) -> dict[str, int]:
    return {"coordinate_extrema": 4 * (N - 1), "span": 4,
            "quantise_cast_clamp": 12 * N,
            "bit_spreading": 30 * N, "combine_coordinates": 2 * N,
            "window_add_modulo": 2 * N * k}


def grid_shape(N: int, k: int) -> tuple[int, int]:
    """Same capacity rule as the measured implementation, without torch."""
    G = max(1, isqrt(N // max(2, k)))
    mean = N / (G * G)
    return G, int(mean + 3 * sqrt(mean) + 2)


def grid_ledger(N: int, k: int) -> tuple[dict[str, int], str]:
    G, M = grid_shape(N, k)
    if 9 * M >= N:
        return pair_ledger(N), "all_pairs_fallback"
    J = 9 * M * N
    return {"normalisation": 16 * N, "cell_ids": 2 * N,
            "histogram": N, "prefix_sum": G * G,
            "neighbor_columns_and_bounds": 43 * N,
            "candidate_index_add": J, "initial_validity": 2 * J,
            "candidate_index_clamp": J, "self_exclusion": 2 * J,
            "squared_distances": 5 * J, "mask_inversion": J}, "grid"


def partial_counts(selector: str, B: int, N: int, d: int, k: int,
                   p: int = 2) -> dict:
    if min(B, d, p) < 1 or N < 2 or not 1 <= k < N:
        raise ValueError("positive B,d,p and 1 <= k < N required")
    if selector == "radius_knn":
        ledger, route = pair_ledger(N, p), "matrix_distance" if N > 25 else "direct_distance"
    elif selector == "morton_window":
        if p != 2:
            raise ValueError("the measured Morton selector is two-dimensional")
        ledger, route = morton_ledger(N, k), "morton"
    elif selector == "grid_knn":
        if p != 2:
            raise ValueError("the measured grid selector is two-dimensional")
        ledger, route = grid_ledger(N, k)
    else:
        raise ValueError(selector)
    known = B * sum(ledger.values())
    scores = aggregate = 2 * B * N * k * d
    dense = 4 * B * N * N * d
    total = known + scores + aggregate
    return {"count_model": MODEL_VERSION, "selector_charge_complete": False,
            "route": route, "batch": B, "n_agents": N, "k": k,
            "head_dim": d, "position_dim": p,
            "scalar_ledger_per_batch_element": ledger,
            "uncharged": ["sorting", "ranking", "indexing_and_allocation", "memory_traffic"],
            "selector": known, "scores": scores, "aggregate": aggregate,
            "attention_total": scores + aggregate, "total": total,
            "dense_total": dense, "partial_count_ratio_upper": dense / total}
