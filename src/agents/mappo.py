"""MAPPO with shared-parameter actor + centralised critic for cooperative MARL.

The actor takes per-agent observations and outputs a categorical action
distribution; parameters are shared across agents. The critic takes the
concatenated joint observation (CTDE assumption: all agent obs available at
training time) and outputs a scalar value of the joint state.

This is intentionally compact — under ~150 lines — so we can drop in the
proposed TopK-TarMAC attention-comm module in Phase 5 without re-plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MAPPOConfig:
    obs_dim: int
    n_actions: int
    n_agents: int
    hidden: int = 64
    use_orthogonal_init: bool = True

    # Toggle the attention-communication module. When False, this exactly
    # reproduces baseline MAPPO. Used as the reduce-to-baseline switch in
    # Phase 5.
    use_comm: bool = False
    # Communication module hyperparameters — only consulted when use_comm.
    msg_dim: int = 16
    n_heads: int = 1
    # Sparse attention mode. `dense` = full softmax over peers (TarMAC-like).
    # `topk` = hard top-k after the soft scores. `adaptive_topk` = top-k with k
    # produced by a per-agent gate, straight-through during training.
    attn_mode: str = "dense"
    topk: int = 0  # used only when attn_mode == "topk"


def _ortho_init(layer: nn.Linear, gain: float = np.sqrt(2.0)) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class SharedActor(nn.Module):
    """Per-agent policy with shared parameters across agents."""

    def __init__(self, cfg: MAPPOConfig) -> None:
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.obs_dim + (cfg.msg_dim if cfg.use_comm else 0)
        self.fc1 = nn.Linear(in_dim, cfg.hidden)
        self.fc2 = nn.Linear(cfg.hidden, cfg.hidden)
        self.head = nn.Linear(cfg.hidden, cfg.n_actions)
        if cfg.use_orthogonal_init:
            for layer in (self.fc1, self.fc2):
                _ortho_init(layer)
            _ortho_init(self.head, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.tanh(self.fc1(x))
        h = F.tanh(self.fc2(h))
        return self.head(h)


class CentralisedCritic(nn.Module):
    """Joint-state value head — concatenates all agent observations.

    Input shape: (B, N * obs_dim). Output: (B,) scalar value.
    """

    def __init__(self, cfg: MAPPOConfig) -> None:
        super().__init__()
        in_dim = cfg.n_agents * cfg.obs_dim
        self.fc1 = nn.Linear(in_dim, cfg.hidden * 2)
        self.fc2 = nn.Linear(cfg.hidden * 2, cfg.hidden * 2)
        self.head = nn.Linear(cfg.hidden * 2, 1)
        if cfg.use_orthogonal_init:
            for layer in (self.fc1, self.fc2):
                _ortho_init(layer)
            _ortho_init(self.head, gain=1.0)

    def forward(self, joint_obs: torch.Tensor) -> torch.Tensor:
        h = F.tanh(self.fc1(joint_obs))
        h = F.tanh(self.fc2(h))
        return self.head(h).squeeze(-1)


class AttentionComm(nn.Module):
    """Inter-agent attention communication block.

    Inputs: per-agent observation embeddings of shape (B, N, D_in).
    Outputs: per-agent message vectors of shape (B, N, msg_dim).

    Supports three modes (cfg.attn_mode):
        dense          — full softmax over all N peers (TarMAC-like).
        topk           — hard top-k after softmax, fixed k = cfg.topk.
        adaptive_topk  — top-k where k is selected per agent from a learned
                         gate via a Gumbel-softmax over {1..N-1}; straight-
                         through to keep gradients.

    The block is the *proposed* method; baseline MAPPO sets cfg.use_comm=False.
    """

    def __init__(self, cfg: MAPPOConfig, in_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(in_dim, 3 * cfg.msg_dim)
        self.proj = nn.Linear(cfg.msg_dim, cfg.msg_dim)
        # k-selector gate: input = per-agent obs embedding, output = logits over
        # K bins in {1..N-1}; only used when attn_mode == "adaptive_topk".
        self.k_bins = max(cfg.n_agents - 1, 1)
        self.k_gate = nn.Linear(in_dim, self.k_bins)
        if cfg.use_orthogonal_init:
            for layer in (self.qkv, self.proj, self.k_gate):
                _ortho_init(layer)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # x: (B, N, in_dim)
        B, N, _ = x.shape
        qkv = self.qkv(x)  # (B, N, 3*msg_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        # Scaled dot-product attention. Mask out self-attention.
        scores = torch.matmul(q, k.transpose(-1, -2)) / (k.shape[-1] ** 0.5)
        eye_mask = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
        scores = scores.masked_fill(eye_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, N, N)

        info: dict = {"mean_attn_entropy": (-(weights.clamp_min(1e-12) *
                                              weights.clamp_min(1e-12).log()).sum(-1)
                                            ).mean().item()}

        if self.cfg.attn_mode == "dense":
            attn = weights
        elif self.cfg.attn_mode == "topk":
            k_eff = min(max(self.cfg.topk, 1), N - 1)
            topk_vals, topk_idx = weights.topk(k_eff, dim=-1)
            mask = torch.zeros_like(weights)
            mask.scatter_(-1, topk_idx, 1.0)
            attn = weights * mask
            # Renormalise so rows still sum to 1.
            attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-8)
            info["effective_k"] = float(k_eff)
        elif self.cfg.attn_mode == "adaptive_topk":
            # Per-agent k selection.
            k_logits = self.k_gate(x)  # (B, N, K)
            k_soft = F.gumbel_softmax(k_logits, tau=1.0, hard=True)  # (B, N, K)
            # Map each row to a length-N "keep mask" of the top-k entries by
            # weight. Implementation: compute cumulative selection over sorted
            # weight indices, then dot-product with k_soft to pick the chosen
            # cutoff.
            sorted_w, sorted_idx = weights.sort(dim=-1, descending=True)  # (B,N,N)
            # cum_keep[b,i,j,m] = 1 iff position j is within the top (m+1)
            # weights for agent i. Use broadcasting on m ∈ {1..K}.
            m_idx = torch.arange(N, device=x.device).view(1, 1, 1, N)  # (1,1,1,N)
            k_idx = (torch.arange(self.k_bins, device=x.device) + 1).view(1, 1, self.k_bins, 1)
            cum_keep = (m_idx < k_idx).float()  # (1,1,K,N)
            keep_for_each_k = cum_keep.expand(B, N, self.k_bins, N)
            keep_sorted = (k_soft.unsqueeze(-1) * keep_for_each_k).sum(dim=-2)  # (B,N,N)
            # Scatter back to original index space.
            keep = torch.zeros_like(weights)
            keep.scatter_(-1, sorted_idx, keep_sorted)
            attn = weights * keep
            attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-8)
            info["mean_k"] = float((k_soft *
                                    (torch.arange(self.k_bins, device=x.device).float() + 1)
                                    ).sum(-1).mean().item())
        else:
            raise ValueError(f"unknown attn_mode {self.cfg.attn_mode!r}")

        msg = torch.matmul(attn, v)  # (B, N, msg_dim)
        msg = self.proj(F.gelu(msg))
        return msg, info


class MAPPOAgent(nn.Module):
    """MAPPO actor-critic with an optional attention-comm module.

    The comm module is *insertable* between the obs embedding and the actor.
    With use_comm=False, it is bypassed and the architecture equals the
    baseline MAPPO described in Yu et al. 2022.
    """

    def __init__(self, cfg: MAPPOConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Optional obs embedding fed into the comm module. With comm off this
        # stays unused.
        self.obs_embed = nn.Linear(cfg.obs_dim, cfg.hidden)
        if cfg.use_orthogonal_init:
            _ortho_init(self.obs_embed)
        self.comm = AttentionComm(cfg, in_dim=cfg.hidden) if cfg.use_comm else None
        self.actor = SharedActor(cfg)
        self.critic = CentralisedCritic(cfg)

    def policy_logits(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """obs: (B, N, obs_dim). Returns logits (B, N, n_actions)."""
        info: dict = {}
        if self.comm is not None:
            emb = F.gelu(self.obs_embed(obs))
            msg, info = self.comm(emb)
            actor_in = torch.cat([obs, msg], dim=-1)
        else:
            actor_in = obs
        # Flatten agents into batch for shared-parameter forward.
        B, N, _ = actor_in.shape
        logits = self.actor(actor_in.reshape(B * N, -1)).reshape(B, N, -1)
        return logits, info

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, N, obs_dim) — flattened to (B, N*obs_dim) for the critic."""
        B = obs.shape[0]
        return self.critic(obs.reshape(B, -1))
