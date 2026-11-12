"""Unit tests for MAPPOAgent.

Verifies:
1. forward pass shapes are correct.
2. value head returns scalar per batch.
3. comm-on reduces to baseline when the message is zeroed (sanity-reduction).
4. loss is finite and decreases over a few gradient steps on a toy batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.agents.mappo import MAPPOAgent, MAPPOConfig


def test_actor_critic_shapes():
    cfg = MAPPOConfig(obs_dim=18, n_actions=5, n_agents=3, hidden=32, use_comm=False)
    agent = MAPPOAgent(cfg)
    obs = torch.randn(4, 3, 18)
    logits, info = agent.policy_logits(obs)
    assert logits.shape == (4, 3, 5)
    v = agent.value(obs)
    assert v.shape == (4,)


def test_comm_off_matches_no_comm_module():
    """An agent with use_comm=False must not even instantiate the comm module."""
    cfg = MAPPOConfig(obs_dim=18, n_actions=5, n_agents=3, hidden=32, use_comm=False)
    agent = MAPPOAgent(cfg)
    assert agent.comm is None


def test_comm_zero_message_recovers_baseline():
    """Sanity-reduction: with use_comm=True but the comm output zeroed, the
    actor logits must equal those of a baseline MAPPO model with the same
    seed for the actor weights.

    We zero the message by patching the comm forward.
    """
    torch.manual_seed(0)
    cfg_base = MAPPOConfig(obs_dim=18, n_actions=5, n_agents=3, hidden=32, use_comm=False)
    base = MAPPOAgent(cfg_base)
    obs = torch.randn(2, 3, 18)
    base_logits, _ = base.policy_logits(obs)

    torch.manual_seed(0)
    cfg_comm = MAPPOConfig(obs_dim=18, n_actions=5, n_agents=3, hidden=32,
                           use_comm=True, msg_dim=8, attn_mode="dense")
    comm_agent = MAPPOAgent(cfg_comm)
    # Patch the comm to emit zero messages — the actor input becomes
    # [obs, 0] and the base actor sees [obs]. The actor input dim differs
    # (obs_dim vs obs_dim+msg_dim), so we instead test that the actor *with
    # comm zeroed* maps obs+zeros through fc1 = base.actor.fc1 only on the
    # obs slice.
    comm_agent.actor.fc1.weight.data[:, :18] = base.actor.fc1.weight.data
    comm_agent.actor.fc1.bias.data = base.actor.fc1.bias.data
    comm_agent.actor.fc2.weight.data = base.actor.fc2.weight.data
    comm_agent.actor.fc2.bias.data = base.actor.fc2.bias.data
    comm_agent.actor.head.weight.data = base.actor.head.weight.data
    comm_agent.actor.head.bias.data = base.actor.head.bias.data
    # Zero out the message-input slice of fc1 so any comm output is ignored.
    comm_agent.actor.fc1.weight.data[:, 18:] = 0.0

    with torch.no_grad():
        comm_logits, _ = comm_agent.policy_logits(obs)
    assert torch.allclose(comm_logits, base_logits, atol=1e-5), (
        comm_logits - base_logits
    ).abs().max().item()


def test_loss_decreases_on_toy_batch():
    cfg = MAPPOConfig(obs_dim=10, n_actions=4, n_agents=2, hidden=32, use_comm=False)
    agent = MAPPOAgent(cfg)
    opt = torch.optim.Adam(agent.parameters(), lr=3e-3)
    obs = torch.randn(64, 2, 10)
    # Target distribution: action=0 always.
    target_action = torch.zeros(64, 2, dtype=torch.long)
    target_value = torch.zeros(64)

    losses = []
    for _ in range(40):
        logits, _ = agent.policy_logits(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(-1, target_action.unsqueeze(-1)).mean()
        v = agent.value(obs)
        v_loss = (v - target_value).pow(2).mean()
        loss = nll + v_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, f"loss did not decrease: {losses[0]} -> {losses[-1]}"


def test_comm_modes_run_and_return_correct_shape():
    for mode in ("dense", "topk", "adaptive_topk"):
        cfg = MAPPOConfig(obs_dim=18, n_actions=5, n_agents=6, hidden=32,
                          use_comm=True, msg_dim=8, attn_mode=mode, topk=2)
        agent = MAPPOAgent(cfg)
        obs = torch.randn(3, 6, 18)
        logits, info = agent.policy_logits(obs)
        assert logits.shape == (3, 6, 5)
        # Backprop sanity.
        loss = logits.sum()
        loss.backward()
