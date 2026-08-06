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
    # Sparse attention mode.
    #   dense         — full softmax over peers, weights materialised as (B,N,N),
    #                   aggregation is a dense (B,N,N)x(B,N,d) matmul.
    #   dense_sdpa    — same maths as `dense`, but fused through
    #                   F.scaled_dot_product_attention so the (B,N,N) weight
    #                   matrix is never materialised. This is the baseline a
    #                   sparse method actually has to beat.
    #   topk          — hard top-k applied as a *mask* on the dense weights.
    #                   Numerically sparse, computationally dense: the zeros are
    #                   materialised and the same dense aggregation matmul runs.
    #   topk_gather   — the same function computed sparsely: top-k on the raw
    #                   scores, a k-wide softmax, and an index_select/gather of
    #                   the selected values into (B,N,k,d) contracted as
    #                   (B*N,1,k)x(B*N,k,d). Executes N/k fewer aggregation MACs
    #                   than `dense`; identical output to `topk`.
    #   random_k      — k peers chosen uniformly at random instead of by
    #                   attention score, aggregated through the same gather
    #                   path. The control for "does the selection rule learn
    #                   anything": matched k, matched cost, no information in
    #                   the choice of peers. A top-k arm that does not beat
    #                   this is not doing what the paper says it does.
    #   entmax        — alpha-entmax attention (alpha=2 is sparsemax), which
    #                   produces exact zeros from the normalisation itself
    #                   rather than from a hard cutoff. Tests the manuscript's
    #                   own diagnosis that the gate is the wrong mechanism.
    #   adaptive_topk — top-k with k produced by a per-agent Gumbel gate.
    attn_mode: str = "dense"
    topk: int = 0  # used only when attn_mode in {"topk", "topk_gather", "random_k"}
    # alpha for attn_mode == "entmax": 1.0 -> softmax, 1.5 -> entmax-1.5,
    # 2.0 -> sparsemax. Larger alpha gives sparser attention.
    entmax_alpha: float = 1.5


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

    Supports five modes (cfg.attn_mode); see MAPPOConfig.attn_mode for the
    computational distinction between `topk` (masked, dense-cost) and
    `topk_gather` (genuinely sparse aggregation). `topk` and `topk_gather`
    compute the same function and agree to floating-point tolerance.

    The block is the *proposed* method; baseline MAPPO sets cfg.use_comm=False.

    `collect_info` gates the diagnostic dictionary. The entropy/mean-k entries
    call `.item()`, which forces a device synchronisation inside the forward
    pass; leaving that in a timed region measures the sync, not the module.
    Latency benchmarks must set it False. Training leaves it True.
    """

    def __init__(self, cfg: MAPPOConfig, in_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.collect_info = True
        self.qkv = nn.Linear(in_dim, 3 * cfg.msg_dim)
        self.proj = nn.Linear(cfg.msg_dim, cfg.msg_dim)
        # k-selector gate: input = per-agent obs embedding, output = logits over
        # K bins in {1..N-1}; only used when attn_mode == "adaptive_topk".
        self.k_bins = max(cfg.n_agents - 1, 1)
        self.k_gate = nn.Linear(in_dim, self.k_bins)
        if cfg.use_orthogonal_init:
            for layer in (self.qkv, self.proj, self.k_gate):
                _ortho_init(layer)

    def _k_eff(self, N: int) -> int:
        return min(max(self.cfg.topk, 1), N - 1)

    def _forward_sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      N: int) -> tuple[torch.Tensor, dict]:
        """Dense attention through the fused SDPA kernel.

        Mathematically identical to the `dense` branch, but the (B, N, N)
        weight matrix is never written to memory, so this is the honest
        dense baseline on modern hardware. The attention weights do not
        exist as a tensor here, so no entropy diagnostic is available.
        """
        mask = torch.zeros(N, N, device=q.device, dtype=q.dtype)
        mask.masked_fill_(torch.eye(N, device=q.device, dtype=torch.bool),
                          float("-inf"))
        # SDPA wants a head dimension: (B, 1, N, d).
        msg = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), attn_mask=mask,
        ).squeeze(1)
        return msg, {}

    def _forward_topk_gather(self, q: torch.Tensor, k: torch.Tensor,
                             v: torch.Tensor, eye_mask: torch.Tensor,
                             B: int, N: int) -> tuple[torch.Tensor, dict]:
        """Genuinely sparse top-k aggregation.

        Two facts make this equivalent to masking the dense weights:
          1. softmax is monotone within a row, so the top-k entries of the
             weights are the top-k entries of the raw scores;
          2. softmax over just those k scores equals the dense softmax
             restricted to them and renormalised.

        So this returns the same numbers as `topk` while contracting a
        (B*N, 1, k) x (B*N, k, d) product instead of (B, N, N) x (B, N, d).
        The score matrix itself is *not* avoidable: selecting the top-k peers
        by attention score requires all N scores. Only the value aggregation
        — half of the attention FLOPs — is sparsifiable.
        """
        d = k.shape[-1]
        scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
        scores = scores.masked_fill(eye_mask, float("-inf"))
        k_eff = self._k_eff(N)
        top_scores, top_idx = scores.topk(k_eff, dim=-1)   # (B, N, k)
        w = F.softmax(top_scores, dim=-1)                  # (B, N, k)

        # Gather the selected value rows: (B, N, k, d). No zeros materialised.
        idx = top_idx.reshape(B, N * k_eff, 1).expand(B, N * k_eff, d)
        v_g = v.gather(1, idx).view(B, N, k_eff, d)
        msg = torch.matmul(w.unsqueeze(-2), v_g).squeeze(-2)  # (B, N, d)

        info: dict = {}
        if self.collect_info:
            info["effective_k"] = float(k_eff)
            info["mean_attn_entropy"] = (
                -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(-1)
            ).mean().item()
        return msg, info

    def _forward_random_k(self, q: torch.Tensor, k: torch.Tensor,
                          v: torch.Tensor, eye_mask: torch.Tensor,
                          B: int, N: int) -> tuple[torch.Tensor, dict]:
        """Aggregate over k peers chosen uniformly at random.

        Identical cost and identical k to `topk_gather`; the only difference is
        that the *identity* of the selected peers carries no information. This
        is the control that separates "attention selects the right peers" from
        "aggregating over any k peers is enough", which the fully observable
        benchmark cannot distinguish on its own.
        """
        d = k.shape[-1]
        scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
        scores = scores.masked_fill(eye_mask, float("-inf"))
        k_eff = self._k_eff(N)

        # Uniform sampling without replacement, self excluded: rank uniform
        # noise and take the k smallest, with self forced to the back.
        noise = torch.rand(B, N, N, device=q.device)
        noise = noise.masked_fill(eye_mask, float("inf"))
        idx_sel = noise.topk(k_eff, dim=-1, largest=False).indices  # (B,N,k)

        sel_scores = scores.gather(-1, idx_sel)
        w = F.softmax(sel_scores, dim=-1)
        idx = idx_sel.reshape(B, N * k_eff, 1).expand(B, N * k_eff, d)
        v_g = v.gather(1, idx).view(B, N, k_eff, d)
        msg = torch.matmul(w.unsqueeze(-2), v_g).squeeze(-2)

        info: dict = {"effective_k": float(k_eff)}
        return msg, info

    def _forward_entmax(self, q: torch.Tensor, k: torch.Tensor,
                        v: torch.Tensor, eye_mask: torch.Tensor,
                        N: int) -> tuple[torch.Tensor, dict]:
        """alpha-entmax attention: sparsity from the normalisation itself.

        Unlike top-k, the support is chosen by the transformation rather than
        by a hard cutoff, so k is data-dependent without needing a gate --
        which is precisely the alternative mechanism the manuscript's own
        limitations section proposes. Solved by bisection on the threshold,
        which is exact in the limit and differentiable through the survivors.
        """
        d = k.shape[-1]
        scores = torch.matmul(q, k.transpose(-1, -2)) / (d ** 0.5)
        alpha = self.cfg.entmax_alpha
        z = scores * (alpha - 1.0)
        # Self is excluded by pushing it below any achievable threshold.
        neg = torch.finfo(z.dtype).min / 4
        z = z.masked_fill(eye_mask, neg)

        hi = z.max(dim=-1, keepdim=True).values
        lo = hi - 1.0
        for _ in range(24):
            tau = (hi + lo) / 2
            p_sum = (z - tau).clamp_min(0).pow(1.0 / (alpha - 1.0)).sum(-1,
                                                                        keepdim=True)
            lo = torch.where(p_sum > 1.0, tau, lo)
            hi = torch.where(p_sum > 1.0, hi, tau)
        tau = (hi + lo) / 2
        attn = (z - tau).clamp_min(0).pow(1.0 / (alpha - 1.0))
        attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-8)

        msg = torch.matmul(attn, v)
        info: dict = {}
        if self.collect_info:
            # The support size is the arm's own k_eff, measured not assumed.
            info["effective_k"] = float((attn > 0).sum(-1).float().mean().item())
        return msg, info

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # x: (B, N, in_dim)
        B, N, _ = x.shape
        qkv = self.qkv(x)  # (B, N, 3*msg_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Sparse and fused paths return before the dense (B, N, N) weight
        # matrix is ever built.
        if self.cfg.attn_mode == "dense_sdpa":
            msg, info = self._forward_sdpa(q, k, v, N)
            return self.proj(F.gelu(msg)), info
        eye_mask = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
        if self.cfg.attn_mode == "topk_gather":
            msg, info = self._forward_topk_gather(q, k, v, eye_mask, B, N)
            return self.proj(F.gelu(msg)), info
        if self.cfg.attn_mode == "random_k":
            msg, info = self._forward_random_k(q, k, v, eye_mask, B, N)
            return self.proj(F.gelu(msg)), info
        if self.cfg.attn_mode == "entmax":
            msg, info = self._forward_entmax(q, k, v, eye_mask, N)
            return self.proj(F.gelu(msg)), info

        # Scaled dot-product attention. Mask out self-attention.
        scores = torch.matmul(q, k.transpose(-1, -2)) / (k.shape[-1] ** 0.5)
        scores = scores.masked_fill(eye_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, N, N)

        info: dict = {}
        if self.collect_info:
            info["mean_attn_entropy"] = (-(weights.clamp_min(1e-12) *
                                           weights.clamp_min(1e-12).log()).sum(-1)
                                         ).mean().item()

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
            # Must follow the attention dtype, not hardcode float32: under
            # bf16/fp16 the promoted float32 product would not scatter into a
            # half-precision `keep`, and the arm would simply fail to run.
            cum_keep = (m_idx < k_idx).to(weights.dtype)  # (1,1,K,N)
            keep_for_each_k = cum_keep.expand(B, N, self.k_bins, N)
            keep_sorted = (k_soft.unsqueeze(-1) * keep_for_each_k).sum(dim=-2)  # (B,N,N)
            # Scatter back to original index space.
            keep = torch.zeros_like(weights)
            keep.scatter_(-1, sorted_idx, keep_sorted)
            attn = weights * keep
            attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-8)
            k_vals = (torch.arange(self.k_bins, device=x.device,
                                   dtype=k_soft.dtype) + 1)
            # Differentiable expected k. The Lagrangian arm needs a budget
            # term with a gradient, so this stays a tensor -- calling .item()
            # here would both detach it and synchronise the device.
            info["expected_k"] = (k_soft * k_vals).sum(-1).mean()
            if self.collect_info:
                info["mean_k"] = float(info["expected_k"].item())
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
