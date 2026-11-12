"""The Z-order (Morton) peer-selection arm of the communication module.

`attn_mode="zorder_k"` fixes each agent's k peers from positions alone, by
adjacency along a Morton curve, before any attention score exists, and then
scores only those k pairs. These tests pin that contract:

a. Every agent gets k distinct peers, never itself.
b. The peers are exactly the curve-order neighbours (+1, -1, +2, ... wrapping),
   checked against an independent pure-Python Morton ordering.
c. The message equals dense attention restricted to the selected peers.
d. An agent's message does not depend on any peer it did not select.
e. MAPPOAgent feeds the agents' self-observed positions, obs[..., 2:4].
"""

from __future__ import annotations

import pytest
import torch

from src.agents.mappo import SELF_POS, AttentionComm, MAPPOAgent, MAPPOConfig
from src.kernels.prescore_selection import select_morton_window


def _cfg(**kw) -> MAPPOConfig:
    base = dict(obs_dim=36, n_actions=5, n_agents=6, hidden=32,
                use_comm=True, msg_dim=8, attn_mode="zorder_k", topk=2)
    base.update(kw)
    return MAPPOConfig(**base)  # type: ignore[arg-type]


def _morton_ref(pos: torch.Tensor, bits: int = 10) -> list[int]:
    """Bit-by-bit Morton code of one (N, 2) position set, same quantisation."""
    lo, hi = pos.min(0).values, pos.max(0).values
    span = (hi - lo).clamp_min(1e-6)
    g = ((pos - lo) / span * (2 ** bits - 1)).long().clamp(0, 2 ** bits - 1)
    codes = []
    for x, y in g.tolist():
        c = 0
        for b in range(bits):
            c |= ((x >> b) & 1) << (2 * b)
            c |= ((y >> b) & 1) << (2 * b + 1)
        codes.append(c)
    return codes


@pytest.mark.parametrize("N,k", [(6, 1), (6, 2), (6, 4), (12, 3)])
def test_selector_k_distinct_peers_without_self(N: int, k: int) -> None:
    torch.manual_seed(N * 10 + k)
    pos = torch.rand(16, N, 2) * 2 - 1
    idx = select_morton_window(pos, k)
    assert idx.shape == (16, N, k)
    self_idx = torch.arange(N).view(1, N, 1)
    assert not (idx == self_idx).any(), "selector kept self"
    srt = idx.sort(-1).values
    assert (srt[..., 1:] != srt[..., :-1]).all(), "duplicate peer selected"


@pytest.mark.parametrize("N,k", [(6, 2), (6, 3), (12, 3)])
def test_selector_matches_curve_order_neighbours(N: int, k: int) -> None:
    torch.manual_seed(1)
    pos = torch.rand(8, N, 2) * 2 - 1
    idx = select_morton_window(pos, k)
    half = (k + 1) // 2
    offs = list(range(1, half + 1)) + [-o for o in range(1, k - half + 1)]
    for b in range(pos.shape[0]):
        codes = _morton_ref(pos[b])
        order = sorted(range(N), key=lambda i: codes[i])
        rank = {a: r for r, a in enumerate(order)}
        for i in range(N):
            want = {order[(rank[i] + o) % N] for o in offs}
            assert set(idx[b, i].tolist()) == want


def test_message_equals_dense_restricted_to_selected() -> None:
    torch.manual_seed(0)
    N, B = 6, 5
    comm = AttentionComm(_cfg(), in_dim=32).eval()
    x = torch.randn(B, N, 32)
    pos = torch.rand(B, N, 2) * 2 - 1
    with torch.no_grad():
        got, info = comm(x, pos)
        q, kk, v = comm.qkv(x).chunk(3, dim=-1)
        scores = q @ kk.transpose(-1, -2) / kk.shape[-1] ** 0.5
        keep = torch.zeros(B, N, N, dtype=torch.bool)
        keep.scatter_(-1, select_morton_window(pos, 2), True)
        w = torch.softmax(scores.masked_fill(~keep, float("-inf")), -1)
        ref = comm.proj(torch.nn.functional.gelu(w @ v))
    assert info["effective_k"] == 2.0
    assert (got - ref).abs().max().item() < 1e-6


def test_message_ignores_unselected_peers() -> None:
    torch.manual_seed(2)
    N = 6
    comm = AttentionComm(_cfg(), in_dim=32).eval()
    x = torch.randn(1, N, 32)
    pos = torch.rand(1, N, 2) * 2 - 1
    sel = select_morton_window(pos, 2)[0, 0].tolist()
    other = next(j for j in range(1, N) if j not in sel)
    x2 = x.clone()
    x2[0, other] += 10.0
    with torch.no_grad():
        m1, _ = comm(x, pos)
        m2, _ = comm(x2, pos)
    assert torch.allclose(m1[0, 0], m2[0, 0], atol=1e-6)


def test_requires_positions() -> None:
    comm = AttentionComm(_cfg(), in_dim=32)
    with pytest.raises(ValueError):
        comm(torch.randn(1, 6, 32))


def test_agent_uses_self_observed_positions() -> None:
    """Selection must follow obs[..., 2:4] and nothing else in the obs."""
    torch.manual_seed(3)
    agent = MAPPOAgent(_cfg()).eval()
    obs = torch.randn(4, 6, 36)
    seen = {}
    orig = agent.comm.forward

    def spy(x, pos=None):
        seen["pos"] = pos
        return orig(x, pos)

    agent.comm.forward = spy  # type: ignore[method-assign]
    with torch.no_grad():
        logits, _ = agent.policy_logits(obs)
    assert logits.shape == (4, 6, 5)
    assert torch.equal(seen["pos"], obs[..., SELF_POS])
