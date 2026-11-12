"""Multi-environment rollout buffer for MAPPO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutBatch:
    obs: torch.Tensor          # (T, E, N, obs_dim)
    actions: torch.Tensor      # (T, E, N)
    logp: torch.Tensor         # (T, E, N)
    values: torch.Tensor       # (T+1, E)
    rewards: torch.Tensor      # (T, E)         — team reward (sum over agents)
    dones: torch.Tensor        # (T, E)
    advantages: torch.Tensor   # (T, E)
    returns: torch.Tensor      # (T, E)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, E = rewards.shape
    advantages = torch.zeros(T, E, device=rewards.device)
    last_gae = torch.zeros(E, device=rewards.device)
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:T]
    return advantages, returns
