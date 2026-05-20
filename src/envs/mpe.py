"""MPE simple_spread wrapper with a tensor-friendly, agent-stacked interface.

Returns numpy arrays of shape (N, obs_dim) and reward shape (N,). Termination is
the OR across agents (mpe2 episodes end together).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpe2 import simple_spread_v3


@dataclass
class MPEConfig:
    n_agents: int = 3
    max_cycles: int = 25
    local_ratio: float = 0.5
    continuous_actions: bool = False
    seed: int = 0


class MPESimpleSpread:
    """Stacked-tensor wrapper around `mpe2.simple_spread_v3.parallel_env`.

    The underlying env returns dicts keyed by `agent_<i>`; we stack them in
    insertion order to expose a consistent (N, *) layout.
    """

    def __init__(self, cfg: MPEConfig) -> None:
        self.cfg = cfg
        self._env = simple_spread_v3.parallel_env(
            N=cfg.n_agents,
            max_cycles=cfg.max_cycles,
            local_ratio=cfg.local_ratio,
            continuous_actions=cfg.continuous_actions,
        )
        self._agents: list[str] | None = None
        self.obs_dim: int
        self.n_actions: int
        self._probe()

    def _probe(self) -> None:
        obs, _ = self._env.reset(seed=self.cfg.seed)
        self._agents = list(obs.keys())
        self.obs_dim = obs[self._agents[0]].shape[0]
        self.n_actions = int(self._env.action_space(self._agents[0]).n)

    @property
    def n_agents(self) -> int:
        return self.cfg.n_agents

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, _info = self._env.reset(seed=seed)
        self._agents = list(obs.keys())
        return np.stack([obs[a] for a in self._agents], axis=0).astype(np.float32)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        assert actions.shape == (self.cfg.n_agents,), actions.shape
        act_dict = {a: int(act) for a, act in zip(self._agents, actions.tolist())}
        obs, rewards, terms, truncs, info = self._env.step(act_dict)
        if not obs:
            # mpe2 returns empty dicts when the episode terminated; recreate
            # zeros so the shapes stay consistent.
            obs_arr = np.zeros((self.cfg.n_agents, self.obs_dim), dtype=np.float32)
            r_arr = np.array([rewards.get(a, 0.0) for a in self._agents], dtype=np.float32)
            done = True
            return obs_arr, r_arr, done, info
        obs_arr = np.stack([obs[a] for a in self._agents], axis=0).astype(np.float32)
        r_arr = np.array([rewards[a] for a in self._agents], dtype=np.float32)
        done = bool(any(terms.values()) or any(truncs.values()))
        return obs_arr, r_arr, done, info

    def close(self) -> None:
        self._env.close()
