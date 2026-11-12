"""Independent checks of primitive charges and the partial-count bound."""
import math

import pytest
import torch

from src.analysis.selector_costs import (direct_distance_ops, grid_ledger,
                                       grid_shape, morton_ledger, pair_ledger,
                                       partial_counts)


@pytest.mark.parametrize("p", [1, 2, 5, 16])
def test_direct_distance_enumeration(p):
    # Explicit primitive list, independent of the closed form.
    ops = ["subtract", "multiply"] * p + ["add"] * (p - 1)
    assert direct_distance_ops(p, squared=True) == len(ops)
    assert direct_distance_ops(p) == len(ops + ["sqrt"])


@pytest.mark.parametrize("N,p", [(48, 1), (192, 2), (512, 5)])
def test_matrix_distance_charge_and_expression(N, p):
    # Reconstruct the actual augmented operands on CPU, not timing them.
    gen = torch.Generator().manual_seed(N + p)
    x, y = (torch.randn(N, p, dtype=torch.float64, generator=gen) for _ in range(2))
    normx, normy = x.square().sum(-1, keepdim=True), y.square().sum(-1, keepdim=True)
    augx = torch.cat((-2 * x, normx, torch.ones_like(normx)), -1)
    augy = torch.cat((y, torch.ones_like(normy), normy), -1)
    result = (augx @ augy.T).clamp_min(0).sqrt()
    direct = (x[:, None] - y[None]).square().sum(-1).sqrt()
    assert torch.allclose(result, direct, atol=1e-7, rtol=1e-7)
    # Count each expression's entries explicitly.
    expected = sum(2 * (p + p - 1) for _ in range(N))
    expected += sum(p for _ in range(N))
    expected += sum(2 * (p + 2) + 2 for _ in range(N * N))
    assert sum(pair_ledger(N, p).values()) == expected
    assert expected == N * (5 * p - 2) + (2 * p + 6) * N**2


@pytest.mark.parametrize("N,k", [(48, 2), (192, 4), (512, 16)])
def test_morton_charge_enumeration(N, k):
    # Reduction/quantisation, five rounds on two coordinates, then window.
    expected = 4 * (N - 1) + 4
    for _ in range(N):
        expected += 2 * 6
        for _coordinate in range(2):
            for _round in range(5):
                expected += 3
        expected += 2
        expected += 2 * k
    assert sum(morton_ledger(N, k).values()) == expected == 48 * N + 2 * N * k


@pytest.mark.parametrize("N,k", [(48, 4), (192, 4), (512, 4), (512, 16)])
def test_grid_branch_and_charge(N, k):
    from src.kernels.prescore_selection import grid_knn_shape
    G, M = grid_shape(N, k)
    assert (G, M) == grid_knn_shape(N, k)
    ledger, route = grid_ledger(N, k)
    if 9 * M >= N:
        assert route == "all_pairs_fallback"
        assert ledger == pair_ledger(N)
    else:
        assert route == "grid"
        # Stage counts independently grouped by source expression.
        per_query = 16 + 2 + 1 + 3 + 9 + 6 + 8 + 11 + 6
        per_candidate = 1 + 2 + 1 + 2 + (2 + 2 + 1) + 1
        assert sum(ledger.values()) == per_query * N + G**2 + per_candidate * 9 * M * N


@pytest.mark.parametrize("selector", ["radius_knn", "morton_window", "grid_knn"])
@pytest.mark.parametrize("N,k,d", [(48, 2, 16), (192, 4, 32), (512, 16, 64)])
def test_partial_bound_not_complete_count(selector, N, k, d):
    a = partial_counts(selector, 1, N, d, k)
    b = partial_counts(selector, 7, N, d, k)
    assert not a["selector_charge_complete"]
    assert b["selector"] == 7 * a["selector"]
    assert b["total"] == 7 * a["total"]
    assert b["partial_count_ratio_upper"] == pytest.approx(a["partial_count_ratio_upper"])
    for omitted in [0, 1, 100000]:
        full = a["dense_total"] / (a["total"] + omitted)
        assert full <= a["partial_count_ratio_upper"] <= N / k


@pytest.mark.parametrize("gamma,p,k,d", [(1, 2, 4, 16), (3, 5, 2, 32)])
def test_conditional_allpairs_limit(gamma, p, k, d):
    N = 10**8
    ratio = 2 * N * N * d / (2 * N * k * d + gamma * N * N * p)
    assert ratio == pytest.approx(2 * d / (gamma * p), rel=1e-5)


def test_morton_nlogn_is_not_little_o_nkd_at_fixed_kd():
    k, d = 4, 16
    penalties = [math.log2(N) / (k * d) for N in [2**8, 2**16, 2**32]]
    assert penalties[0] < penalties[1] < penalties[2]
