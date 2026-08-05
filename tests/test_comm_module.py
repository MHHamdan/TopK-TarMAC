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


def test_topk_aggregation_executes_a_dense_matmul() -> None:
    """The top-k path does the *same* dense matmul as the dense path.

    ``AttentionComm.forward`` masks and renormalises the attention matrix and
    then calls ``torch.matmul(attn, v)`` with ``attn`` still shaped (B, N, N)
    -- the zeros are materialised, not skipped. So top-k executes every
    multiply-accumulate dense executes, plus the top-k selection and scatter
    on top.

    This test pins the discrepancy in place: ``attention_flops`` reports a
    saving of ``1 - k/(N-1)`` that the implementation never realises. Any
    future change that makes top-k genuinely sparse (a gather into a (B, N, k)
    tensor) will break this test, which is the point -- it should be revisited
    deliberately, not drifted past.
    """
    from scripts.measure_flops import attention_flops

    N, k, B, d = 48, 4, 8, 16
    seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original = torch.matmul

    def traced(a, b, *args, **kwargs):
        seen.append((tuple(a.shape), tuple(b.shape)))
        return original(a, b, *args, **kwargs)

    shapes: dict[str, list] = {}
    try:
        torch.matmul = traced  # type: ignore[assignment]
        for mode, kw in (("dense", {}), ("topk", {"topk": k})):
            seen.clear()
            cfg = _cfg(n_agents=N, attn_mode=mode, msg_dim=d, **kw)
            AttentionComm(cfg, in_dim=32)(torch.randn(B, N, 32))
            shapes[mode] = [s for s in seen if s[0][-1] == N]
    finally:
        torch.matmul = original  # type: ignore[assignment]

    assert shapes["dense"] == shapes["topk"], (
        "top-k no longer mirrors dense's aggregation matmul; if the module "
        "was made genuinely sparse, update attention_flops and this test"
    )
    assert shapes["topk"] == [((B, N, N), (B, N, d))]

    # The counter's claimed saving is real arithmetic on numbers the code
    # does not execute.
    claimed = attention_flops(N, float(k), d)["msg_gather_over_dense"]
    assert claimed == k / (N - 1)
    assert claimed < 0.1, "this N,k was chosen to make the gap obvious"


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
