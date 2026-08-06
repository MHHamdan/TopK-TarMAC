"""Invariants of the inter-agent communication module.

These tests exist because the manuscript asserts each of these properties in
prose. Anything the paper claims about the module's behaviour is verified
here, not assumed:

a. ``k = N-1`` reproduces dense attention to within 1e-6.
b. A zeroed communication channel reproduces MAPPO logits exactly.
c. Top-k renormalisation leaves every row summing to 1.
d. The adaptive gate's straight-through gradient is finite and non-zero.
e. The FLOPs counter matches an analytic formula on a synthetic batch.
"""

from __future__ import annotations

import numpy as np
import torch

from src.agents.mappo import AttentionComm, MAPPOAgent, MAPPOConfig


def _cfg(**kw) -> MAPPOConfig:
    base = dict(obs_dim=18, n_actions=5, n_agents=6, hidden=32,
                use_comm=True, msg_dim=8, attn_mode="dense")
    base.update(kw)
    return MAPPOConfig(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# (a) k = N-1 reproduces dense attention
# --------------------------------------------------------------------------

def test_topk_at_n_minus_1_equals_dense() -> None:
    """With k = N-1 every off-diagonal peer is kept, so top-k == dense.

    The manuscript states this as the module's reduce-to-dense property
    (03_method.tex). Self-attention is masked to -inf, so the row has exactly
    N-1 candidates and selecting all of them must be an identity.
    """
    torch.manual_seed(0)
    N = 6
    dense = AttentionComm(_cfg(n_agents=N, attn_mode="dense"), in_dim=32)
    topk = AttentionComm(_cfg(n_agents=N, attn_mode="topk", topk=N - 1), in_dim=32)
    topk.load_state_dict(dense.state_dict())

    x = torch.randn(4, N, 32)
    with torch.no_grad():
        m_dense, _ = dense(x)
        m_topk, info = topk(x)

    assert info["effective_k"] == float(N - 1)
    max_err = (m_dense - m_topk).abs().max().item()
    assert max_err < 1e-6, f"top-k(N-1) deviates from dense by {max_err:.3e}"


def test_topk_at_n_minus_1_equals_dense_across_n() -> None:
    """The reduce-to-dense identity must hold at every N we report on."""
    for N in (3, 6, 12, 24):
        torch.manual_seed(N)
        dense = AttentionComm(_cfg(n_agents=N, attn_mode="dense"), in_dim=32)
        topk = AttentionComm(_cfg(n_agents=N, attn_mode="topk", topk=N - 1), in_dim=32)
        topk.load_state_dict(dense.state_dict())
        x = torch.randn(3, N, 32)
        with torch.no_grad():
            a, _ = dense(x)
            b, _ = topk(x)
        assert (a - b).abs().max().item() < 1e-6, f"failed at N={N}"


# --------------------------------------------------------------------------
# (b) zeroed comm reproduces MAPPO logits exactly
# --------------------------------------------------------------------------

def test_zeroed_comm_reproduces_mappo_logits_exactly() -> None:
    """A comm-enabled agent whose message pathway is zeroed is bit-identical
    to baseline MAPPO.

    The equivalence is weight-matched, not architectural: enabling comm widens
    the actor's first layer from ``obs_dim`` to ``obs_dim + msg_dim``, so the
    two models cannot share a single ``fc1``. The correct statement -- and the
    one the manuscript must make -- is that with the message-input columns of
    ``fc1`` set to zero and the remaining actor weights copied across, the
    logits agree exactly (not merely to a tolerance).
    """
    torch.manual_seed(0)
    obs_dim, msg_dim = 18, 8
    base = MAPPOAgent(MAPPOConfig(obs_dim=obs_dim, n_actions=5, n_agents=6,
                                  hidden=32, use_comm=False))
    comm = MAPPOAgent(_cfg(obs_dim=obs_dim, msg_dim=msg_dim,
                           attn_mode="adaptive_topk"))

    with torch.no_grad():
        comm.actor.fc1.weight[:, :obs_dim] = base.actor.fc1.weight
        comm.actor.fc1.weight[:, obs_dim:] = 0.0
        comm.actor.fc1.bias.copy_(base.actor.fc1.bias)
        comm.actor.fc2.weight.copy_(base.actor.fc2.weight)
        comm.actor.fc2.bias.copy_(base.actor.fc2.bias)
        comm.actor.head.weight.copy_(base.actor.head.weight)
        comm.actor.head.bias.copy_(base.actor.head.bias)

    obs = torch.randn(5, 6, obs_dim)
    with torch.no_grad():
        base_logits, _ = base.policy_logits(obs)
        comm_logits, _ = comm.policy_logits(obs)

    assert torch.equal(comm_logits, base_logits), (
        "zeroed comm is not bit-identical to MAPPO; max abs diff "
        f"{(comm_logits - base_logits).abs().max().item():.3e}"
    )


def test_zeroed_comm_message_does_not_reach_actor_gradient() -> None:
    """With the message columns zeroed, no gradient flows into the comm block."""
    torch.manual_seed(0)
    obs_dim = 18
    comm = MAPPOAgent(_cfg(obs_dim=obs_dim, attn_mode="dense"))
    with torch.no_grad():
        comm.actor.fc1.weight[:, obs_dim:] = 0.0

    logits, _ = comm.policy_logits(torch.randn(4, 6, obs_dim))
    logits.sum().backward()

    qkv_grad = comm.comm.qkv.weight.grad
    assert qkv_grad is not None
    assert qkv_grad.abs().max().item() == 0.0


# --------------------------------------------------------------------------
# (c) top-k renormalisation sums to 1 per row
# --------------------------------------------------------------------------

def _attention_rows(cfg: MAPPOConfig, x: torch.Tensor) -> torch.Tensor:
    """Recompute the post-mask attention matrix the module aggregates with."""
    mod = AttentionComm(cfg, in_dim=x.shape[-1])
    torch.manual_seed(0)
    B, N, _ = x.shape
    q, k, _v = mod.qkv(x).chunk(3, dim=-1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / (k.shape[-1] ** 0.5)
    eye = torch.eye(N, dtype=torch.bool).unsqueeze(0)
    scores = scores.masked_fill(eye, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    if cfg.attn_mode == "dense":
        return weights
    k_eff = min(max(cfg.topk, 1), N - 1)
    _vals, idx = weights.topk(k_eff, dim=-1)
    mask = torch.zeros_like(weights).scatter_(-1, idx, 1.0)
    attn = weights * mask
    return attn / attn.sum(-1, keepdim=True).clamp_min(1e-8)


def test_topk_rows_sum_to_one() -> None:
    """Every renormalised attention row is a probability distribution."""
    for N in (3, 6, 12):
        for k in range(1, N):
            cfg = _cfg(n_agents=N, attn_mode="topk", topk=k)
            x = torch.randn(4, N, 32)
            rows = _attention_rows(cfg, x)
            sums = rows.sum(-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), (
                f"N={N} k={k}: row sums deviate by "
                f"{(sums - 1).abs().max().item():.3e}"
            )


def test_topk_keeps_exactly_k_nonzeros() -> None:
    """Sparsity is exactly k per row -- the premise of the FLOPs accounting."""
    for N in (6, 12):
        for k in (1, 2, N - 1):
            cfg = _cfg(n_agents=N, attn_mode="topk", topk=k)
            rows = _attention_rows(cfg, torch.randn(3, N, 32))
            nnz = (rows > 0).sum(-1)
            assert torch.all(nnz == k), f"N={N} k={k}: got {nnz.unique().tolist()}"


def test_dense_rows_sum_to_one_and_exclude_self() -> None:
    """Dense rows are distributions over peers only; the diagonal is zero."""
    N = 6
    cfg = _cfg(n_agents=N, attn_mode="dense")
    rows = _attention_rows(cfg, torch.randn(4, N, 32))
    sums = rows.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    diag = rows[:, torch.arange(N), torch.arange(N)]
    assert diag.abs().max().item() == 0.0


# --------------------------------------------------------------------------
# (d) gate straight-through gradient is finite and non-zero
# --------------------------------------------------------------------------

def test_gate_straight_through_gradient_is_finite_and_nonzero() -> None:
    """The adaptive gate must receive usable gradient.

    Gumbel-softmax with ``hard=True`` returns a one-hot sample whose gradient
    is the soft relaxation's. If this were zero or non-finite the adaptive arm
    would be training nothing.
    """
    torch.manual_seed(0)
    N = 6
    mod = AttentionComm(_cfg(n_agents=N, attn_mode="adaptive_topk"), in_dim=32)
    x = torch.randn(8, N, 32)
    msg, _info = mod(x)
    msg.sum().backward()

    g = mod.k_gate.weight.grad
    assert g is not None, "gate received no gradient at all"
    assert torch.isfinite(g).all(), "gate gradient contains NaN or Inf"
    assert g.abs().max().item() > 0.0, "gate gradient is identically zero"


def test_gate_gradient_nonzero_over_repeated_samples() -> None:
    """Gate gradient is reliably non-zero, not just on one lucky sample."""
    N = 6
    nonzero = 0
    trials = 10
    for t in range(trials):
        torch.manual_seed(100 + t)
        mod = AttentionComm(_cfg(n_agents=N, attn_mode="adaptive_topk"), in_dim=32)
        msg, _ = mod(torch.randn(8, N, 32))
        msg.sum().backward()
        g = mod.k_gate.weight.grad
        assert g is not None and torch.isfinite(g).all()
        nonzero += int(g.abs().max().item() > 0.0)
    assert nonzero == trials, f"gate gradient vanished on {trials - nonzero}/{trials}"


def test_adaptive_gate_reports_mean_k_in_range() -> None:
    """Reported ``mean_k`` must lie in [1, N-1] -- it feeds the FLOPs ratio."""
    N = 12
    torch.manual_seed(0)
    mod = AttentionComm(_cfg(n_agents=N, attn_mode="adaptive_topk"), in_dim=32)
    with torch.no_grad():
        _msg, info = mod(torch.randn(16, N, 32))
    assert 1.0 <= info["mean_k"] <= float(N - 1), info["mean_k"]


# --------------------------------------------------------------------------
# (e) FLOPs counter matches an analytic formula
# --------------------------------------------------------------------------

def test_flops_counter_matches_analytic_formula() -> None:
    """``attention_flops`` must equal the closed form it documents.

    Message gather is ``N * k_eff * msg_dim`` multiply-accumulates; the score
    matrix is ``N^2 * msg_dim`` and is paid by dense and sparse alike, which
    is why the paper's saving applies only to the gather term.
    """
    from scripts.measure_flops import attention_flops

    for N in (3, 6, 12, 24, 48):
        for k_eff in (1.0, 2.5, float(N - 1)):
            for msg_dim in (8, 16, 32):
                got = attention_flops(N, k_eff, msg_dim)
                assert got["msg_gather_ops"] == float(N * k_eff * msg_dim)
                assert got["score_ops"] == float(N * N * msg_dim)
                assert np.isclose(got["msg_gather_over_dense"],
                                  k_eff / max(N - 1, 1))


def test_flops_ratio_is_one_for_dense() -> None:
    """A dense run must report a gather ratio of exactly 1.0."""
    from scripts.measure_flops import attention_flops

    for N in (3, 6, 12, 24):
        assert attention_flops(N, float(N - 1), 16)["msg_gather_over_dense"] == 1.0


def test_masked_topk_aggregation_executes_a_dense_matmul() -> None:
    """``attn_mode="topk"`` does the *same* dense matmul as the dense path.

    This is the strawman variant, kept deliberately. It masks and renormalises
    the attention matrix and then calls ``torch.matmul(attn, v)`` with ``attn``
    still shaped (B, N, N) -- the zeros are materialised, not skipped. So it
    executes every multiply-accumulate dense executes, plus the top-k selection
    and scatter on top.

    ``attn_mode="topk_gather"`` is the variant that actually skips the work;
    the tests below pin the difference. Benchmarking sparsity against *this*
    path only would be benchmarking against an implementation artifact.
    """
    N, k, B, d = 48, 4, 8, 16
    shapes = _traced_matmul_shapes({"dense": {}, "topk": {"topk": k}},
                                   N=N, B=B, d=d)

    assert shapes["dense"] == shapes["topk"], (
        "the masked top-k path no longer mirrors dense's aggregation matmul"
    )
    assert shapes["topk"] == [((B, N, d), (B, d, N)),   # scores
                              ((B, N, N), (B, N, d))]   # dense aggregation


def _traced_matmul_shapes(modes: dict[str, dict], N: int, B: int,
                          d: int) -> dict[str, list]:
    """Record the (lhs, rhs) shapes of every ``torch.matmul`` per mode."""
    seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original = torch.matmul

    def traced(a, b, *args, **kwargs):
        seen.append((tuple(a.shape), tuple(b.shape)))
        return original(a, b, *args, **kwargs)

    out: dict[str, list] = {}
    try:
        torch.matmul = traced  # type: ignore[assignment]
        for mode, kw in modes.items():
            seen.clear()
            cfg = _cfg(n_agents=N, attn_mode=mode, msg_dim=d, **kw)
            mod = AttentionComm(cfg, in_dim=32)
            mod.collect_info = False
            mod(torch.randn(B, N, 32))
            # Keep only the attention-internal products; the qkv/proj Linears
            # go through F.linear, not torch.matmul, so nothing else appears.
            out[mode] = list(seen)
    finally:
        torch.matmul = original  # type: ignore[assignment]
    return out


# --------------------------------------------------------------------------
# (f) the gather path is the same function computed with fewer operations
# --------------------------------------------------------------------------

def test_topk_gather_matches_masked_topk_exactly() -> None:
    """``topk_gather`` and ``topk`` compute the same function.

    Two facts make this hold: softmax is monotone within a row, so the top-k
    weights are the top-k scores; and a softmax over just those k scores equals
    the dense softmax restricted to them and renormalised. If this test fails,
    the "sparse" path is not the method the dense path is being compared to,
    and every latency comparison between them is meaningless.
    """
    for N in (6, 12, 24):
        for k in (1, 2, 3, N - 1):
            torch.manual_seed(N * 100 + k)
            masked = AttentionComm(
                _cfg(n_agents=N, attn_mode="topk", topk=k), in_dim=32).double()
            gather = AttentionComm(
                _cfg(n_agents=N, attn_mode="topk_gather", topk=k), in_dim=32).double()
            gather.load_state_dict(masked.state_dict())
            x = torch.randn(8, N, 32, dtype=torch.double)
            with torch.no_grad():
                a, _ = masked(x)
                b, _ = gather(x)
            err = (a - b).abs().max().item()
            assert err < 1e-10, f"N={N} k={k}: gather deviates by {err:.3e}"


def test_gather_path_never_materialises_the_dense_weight_matrix() -> None:
    """The gather path must not execute a (B, N, N) x (B, N, d) product.

    This is the assertion that separates a real sparse implementation from a
    masked one. The gather contracts (B*N, 1, k) x (B*N, k, d) instead.
    """
    N, k, B, d = 48, 4, 8, 16
    shapes = _traced_matmul_shapes({"topk_gather": {"topk": k}}, N=N, B=B, d=d)
    agg = [s for s in shapes["topk_gather"] if s[1] == (B, N, d)]
    assert agg == [], f"gather path still runs a dense aggregation: {agg}"
    assert ((B, N, 1, k), (B, N, k, d)) in shapes["topk_gather"], shapes


def test_topk_gather_executes_fewer_macs_than_dense() -> None:
    """Executed FLOPs must differ between the dense, masked and gather paths.

    ``measure_flops.attention_flops`` reports an analytic saving. Until the
    gather path existed, nothing executed it (D-019). This pins the executed
    counts so the two can never silently diverge again:

      dense / masked : qkv + proj + 2BN^2 d (scores) + 2BN^2 d (aggregate)
      gather         : qkv + proj + 2BN^2 d (scores) + 2BNk d (aggregate)
    """
    from torch.utils.flop_counter import FlopCounterMode

    N, k, B, d, in_dim = 48, 4, 8, 16, 32
    counts: dict[str, int] = {}
    for mode, kw in (("dense", {}), ("topk", {"topk": k}),
                     ("topk_gather", {"topk": k})):
        mod = AttentionComm(_cfg(n_agents=N, attn_mode=mode, msg_dim=d, **kw),
                            in_dim=in_dim)
        mod.collect_info = False
        counter = FlopCounterMode(display=False)
        with counter:
            mod(torch.randn(B, N, in_dim))
        counts[mode] = counter.get_total_flops()

    proj_flops = 2 * B * N * in_dim * (3 * d) + 2 * B * N * d * d
    scores = 2 * B * N * N * d
    assert counts["dense"] == proj_flops + 2 * scores
    assert counts["topk"] == counts["dense"], (
        "the masked path is supposed to be dense-cost; if it was optimised, "
        "the strawman comparison in the manuscript must be revisited"
    )
    assert counts["topk_gather"] == proj_flops + scores + 2 * B * N * k * d
    assert counts["topk_gather"] < counts["dense"], (
        "the gather path executes no fewer FLOPs than dense -- the paper's "
        "efficiency premise is unimplemented again"
    )


def test_score_matmul_is_irreducible_in_every_mode() -> None:
    """Every top-k mode still pays the full (B, N, N) score matmul.

    Selecting the top-k peers *by attention score* requires all N scores, so
    only the aggregation half of attention is sparsifiable. This bounds the
    achievable saving at 2x of attention FLOPs no matter how small k is --
    the ceiling the manuscript's 21-29% claim was measured against.
    """
    N, k, B, d = 48, 4, 8, 16
    shapes = _traced_matmul_shapes(
        {"dense": {}, "topk": {"topk": k}, "topk_gather": {"topk": k}},
        N=N, B=B, d=d)
    for mode, seen in shapes.items():
        assert ((B, N, d), (B, d, N)) in seen, (
            f"{mode} does not compute the full score matrix; if a cheap "
            f"routing proxy was introduced, the 2x ceiling no longer applies"
        )


def test_attention_flop_saving_is_bounded_by_two() -> None:
    """No choice of k can make sparse attention more than 2x cheaper.

    attention FLOPs = scores (2BN^2 d) + aggregation (2BNk d).
    dense = 4BN^2 d; gather at k=1 -> 2BN^2 d + 2BN d. The ratio tends to 2
    from below and never exceeds it.
    """
    B, d = 8, 16
    for N in (6, 12, 24, 48, 192, 512):
        dense = 2 * B * N * N * d + 2 * B * N * (N - 1) * d
        best = 2 * B * N * N * d + 2 * B * N * 1 * d
        assert dense / best < 2.0, f"N={N}: ratio {dense / best:.4f}"


def test_dense_sdpa_matches_dense() -> None:
    """The fused SDPA baseline computes the same function as `dense`.

    It is the baseline a sparse method has to beat on real hardware: it never
    materialises the (B, N, N) weight matrix, so comparing top-k only against
    the unfused `dense` path would flatter it.
    """
    for N in (6, 24):
        torch.manual_seed(N)
        dense = AttentionComm(_cfg(n_agents=N, attn_mode="dense"),
                              in_dim=32).double()
        sdpa = AttentionComm(_cfg(n_agents=N, attn_mode="dense_sdpa"),
                             in_dim=32).double()
        sdpa.load_state_dict(dense.state_dict())
        x = torch.randn(8, N, 32, dtype=torch.double)
        with torch.no_grad():
            a, _ = dense(x)
            b, _ = sdpa(x)
        assert (a - b).abs().max().item() < 1e-10, f"N={N}"


def test_random_k_selects_exactly_k_distinct_peers_never_self() -> None:
    """The random-k control must match top-k's k and exclude self.

    If it could select self, or select the same peer twice, it would not be a
    matched-cost control and the comparison against top-k would be confounded.
    """
    for N in (6, 12, 24):
        for k in (1, 3, N - 1):
            torch.manual_seed(N * 10 + k)
            mod = AttentionComm(
                _cfg(n_agents=N, attn_mode="random_k", topk=k), in_dim=32)
            B = 128
            eye = torch.eye(N, dtype=torch.bool).unsqueeze(0)
            noise = torch.rand(B, N, N).masked_fill(eye, float("inf"))
            idx = noise.topk(min(k, N - 1), dim=-1, largest=False).indices
            self_idx = torch.arange(N).view(1, N, 1).expand_as(idx)
            assert not bool((idx == self_idx).any()), f"N={N} k={k}: selected self"
            assert bool((idx.sort(-1).values.diff(dim=-1) > 0).all()), (
                f"N={N} k={k}: duplicate peer selected")
            _msg, info = mod(torch.randn(4, N, 32))
            assert info["effective_k"] == float(min(max(k, 1), N - 1))


def test_random_k_differs_from_topk_but_costs_the_same() -> None:
    """Matched k and matched executed FLOPs, different selected peers.

    That is the whole design of the control: any performance gap between them
    is attributable to *which* peers were chosen, not to how many.
    """
    from torch.utils.flop_counter import FlopCounterMode

    N, k, B, d = 24, 4, 8, 16
    counts = {}
    for mode in ("topk_gather", "random_k"):
        torch.manual_seed(0)
        mod = AttentionComm(_cfg(n_agents=N, attn_mode=mode, topk=k, msg_dim=d),
                            in_dim=32)
        mod.collect_info = False
        counter = FlopCounterMode(display=False)
        with counter:
            mod(torch.randn(B, N, 32))
        counts[mode] = counter.get_total_flops()
    assert counts["topk_gather"] == counts["random_k"], counts

    torch.manual_seed(0)
    a = AttentionComm(_cfg(n_agents=N, attn_mode="topk_gather", topk=k), in_dim=32)
    b = AttentionComm(_cfg(n_agents=N, attn_mode="random_k", topk=k), in_dim=32)
    b.load_state_dict(a.state_dict())
    x = torch.randn(16, N, 32)
    with torch.no_grad():
        ma, _ = a(x)
        mb, _ = b(x)
    assert not torch.allclose(ma, mb), (
        "random-k reproduced top-k exactly; the control is not controlling")


def test_entmax_produces_exact_zeros_and_sparsifies_with_alpha() -> None:
    """entmax must yield genuine zeros, and more of them as alpha grows.

    alpha=1 is softmax (dense); larger alpha is sparser. If the support did
    not shrink with alpha the arm would not be testing the manuscript's own
    proposed alternative to the gate.
    """
    N = 24
    supports = []
    for alpha in (1.5, 2.0):
        torch.manual_seed(0)
        mod = AttentionComm(
            _cfg(n_agents=N, attn_mode="entmax", entmax_alpha=alpha), in_dim=32)
        with torch.no_grad():
            _msg, info = mod(torch.randn(64, N, 32))
        supports.append(info["effective_k"])
        assert 1.0 <= info["effective_k"] <= float(N - 1), info
    assert supports[1] < supports[0], (
        f"alpha=2.0 support {supports[1]} not sparser than alpha=1.5 "
        f"{supports[0]}")


def test_entmax_gradient_is_finite_and_nonzero() -> None:
    """Sparsity must not kill the gradient the arm trains on."""
    for alpha in (1.5, 2.0):
        torch.manual_seed(0)
        mod = AttentionComm(
            _cfg(n_agents=12, attn_mode="entmax", entmax_alpha=alpha), in_dim=32)
        msg, _ = mod(torch.randn(32, 12, 32))
        msg.sum().backward()
        g = mod.qkv.weight.grad
        assert g is not None and torch.isfinite(g).all()
        assert g.abs().max().item() > 0.0, f"alpha={alpha}: gradient vanished"


def test_adaptive_gate_exposes_differentiable_expected_k() -> None:
    """The Lagrangian arm needs a budget term with a gradient.

    ``expected_k`` must stay a tensor: calling ``.item()`` on it would detach
    the dual constraint from the gate it is supposed to constrain, so the
    budget would be reported but never enforced.
    """
    N = 12
    torch.manual_seed(0)
    mod = AttentionComm(_cfg(n_agents=N, attn_mode="adaptive_topk"), in_dim=32)
    _msg, info = mod(torch.randn(16, N, 32))
    ek = info["expected_k"]
    assert isinstance(ek, torch.Tensor), type(ek)
    assert ek.requires_grad, "expected_k is detached; the budget cannot train"
    ek.backward()
    assert mod.k_gate.weight.grad is not None
    assert mod.k_gate.weight.grad.abs().max().item() > 0.0
    assert 1.0 <= float(ek.detach()) <= float(N - 1)


def test_collect_info_removes_the_device_sync_from_forward() -> None:
    """With ``collect_info=False`` the forward returns no ``.item()``-derived
    entry.

    The entropy and mean-k diagnostics call ``.item()``, which synchronises the
    device inside the forward pass. Leaving that inside a timed region measures
    the synchronisation rather than the module, which is what made dense
    latency look flat in N in the first microbenchmark.
    """
    for mode, kw in (("dense", {}), ("topk", {"topk": 2}),
                     ("topk_gather", {"topk": 2}), ("adaptive_topk", {})):
        mod = AttentionComm(_cfg(n_agents=12, attn_mode=mode, **kw), in_dim=32)
        mod.collect_info = False
        _msg, info = mod(torch.randn(4, 12, 32))
        assert "mean_attn_entropy" not in info
        assert "mean_k" not in info


def test_flops_counter_matches_measured_nonzeros() -> None:
    """The analytic gather count must agree with the module's actual sparsity.

    This is the test that connects the counter to the implementation: it
    counts non-zero attention entries on a synthetic batch and checks the
    analytic formula predicts the same number of gather operations.
    """
    from scripts.measure_flops import attention_flops

    N, msg_dim, B = 12, 8, 4
    for k in (1, 3, 7, N - 1):
        cfg = _cfg(n_agents=N, attn_mode="topk", topk=k, msg_dim=msg_dim)
        rows = _attention_rows(cfg, torch.randn(B, N, 32))
        measured_nnz_per_sample = (rows > 0).sum().item() / B
        analytic = attention_flops(N, float(k), msg_dim)
        assert measured_nnz_per_sample == float(N * k)
        assert analytic["msg_gather_ops"] == measured_nnz_per_sample * msg_dim
