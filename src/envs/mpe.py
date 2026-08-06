"""MPE wrappers with a tensor-friendly, agent-stacked interface.

Returns numpy arrays of shape (N, obs_dim) and reward shape (N,). Termination is
the OR across agents (mpe2 episodes end together).

Two families live here, and the distinction is the whole point of D-011:

`MPESimpleSpread` is the historical benchmark. It is fully observable and
locally solvable, so "sparsification is free" and "the channel was never used"
make identical predictions on it and no sparsification result measured there is
interpretable. Setting `visibility_radius` turns it into a genuinely
partially-observable task -- each agent sees only peers inside the radius and
must communicate to recover the rest -- while keeping N a free parameter, which
is what the k-sparsification arms need.

`MPESimpleReference` is the literature-standard communication-critical control:
neither agent can observe its own goal, only its partner's, so the task is
unsolvable without a working channel. It is fixed at N=2, so it can establish
that the channel works but cannot test sparsification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mpe2 import simple_reference_v3, simple_spread_v3


@dataclass
class MPEConfig:
    n_agents: int = 3
    max_cycles: int = 25
    local_ratio: float = 0.5
    continuous_actions: bool = False
    seed: int = 0
    # None keeps the task fully observable (the historical behaviour, bit for
    # bit). A float masks every peer farther away than this, in MPE world
    # units, out of the observation -- see `_apply_visibility`.
    visibility_radius: float | None = None


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

    # ------------------------------------------------------------------
    # partial observability
    # ------------------------------------------------------------------
    @property
    def _peer_slice(self) -> slice:
        """Indices of the other-agent relative positions in a spread obs.

        simple_spread lays each observation out as
            [self_vel(2), self_pos(2), landmark_rel(2L), peer_rel(2(N-1)),
             comm(2(N-1))]
        with L = N landmarks, giving obs_dim = 6N. The peer block therefore
        starts at 4 + 2N and runs for 2(N-1) entries.
        """
        n = self.cfg.n_agents
        start = 4 + 2 * n
        return slice(start, start + 2 * (n - 1))

    def _apply_visibility(self, obs: np.ndarray) -> np.ndarray:
        """Zero out peers beyond `visibility_radius`.

        The peer block holds *relative* positions, so a peer's distance is the
        norm of its own 2-vector and the mask needs no global state. Zeroing
        rather than deleting keeps obs_dim fixed across timesteps, which the
        stacked-tensor interface requires; a zeroed relative position is
        indistinguishable from a peer at the agent's exact location, which is
        an accepted and documented aliasing (agents are volume-excluded in
        simple_spread, so exact coincidence does not occur in practice).
        """
        r = self.cfg.visibility_radius
        if r is None:
            return obs
        n = self.cfg.n_agents
        block = obs[:, self._peer_slice].reshape(n, n - 1, 2)
        dist = np.linalg.norm(block, axis=-1)          # (N, N-1)
        block[dist > r] = 0.0
        obs[:, self._peer_slice] = block.reshape(n, 2 * (n - 1))
        return obs

    def visible_fraction(self, obs: np.ndarray) -> float:
        """Fraction of peers currently visible -- the knob's actual effect.

        Reported alongside returns so the difficulty of the partially
        observable variant is a measured quantity, not an assumed one.
        """
        n = self.cfg.n_agents
        if n < 2:
            return 1.0
        block = obs[:, self._peer_slice].reshape(n, n - 1, 2)
        return float((np.linalg.norm(block, axis=-1) > 0).mean())

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, _info = self._env.reset(seed=seed)
        self._agents = list(obs.keys())
        arr = np.stack([obs[a] for a in self._agents], axis=0).astype(np.float32)
        return self._apply_visibility(arr)

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
        obs_arr = self._apply_visibility(obs_arr)
        r_arr = np.array([rewards[a] for a in self._agents], dtype=np.float32)
        done = bool(any(terms.values()) or any(truncs.values()))
        return obs_arr, r_arr, done, info

    def close(self) -> None:
        self._env.close()


class MPESimpleReference:
    """Stacked-tensor wrapper around `mpe2.simple_reference_v3.parallel_env`.

    The communication-critical positive control required by D-011. Each of the
    two agents is rewarded for reaching a target landmark that only its
    *partner* can observe, so the task is not solvable without the channel
    carrying information. If dense attention-comm does not beat comm-free MAPPO
    here, the comm module is not working and no result about sparsifying it
    means anything.

    Both agents are homogeneous -- obs (21,), Discrete(50) -- so the same
    stacked (N, *) interface as `MPESimpleSpread` applies unchanged. N is fixed
    at 2 by the environment, which is why this control cannot itself carry the
    k-sparsification arms; use `MPESimpleSpread(visibility_radius=...)` for
    those.
    """

    def __init__(self, cfg: MPEConfig) -> None:
        self.cfg = cfg
        self._env = simple_reference_v3.parallel_env(
            local_ratio=cfg.local_ratio,
            max_cycles=cfg.max_cycles,
            continuous_actions=cfg.continuous_actions,
        )
        self._agents: list[str] | None = None
        self.obs_dim: int
        self.n_actions: int
        obs, _ = self._env.reset(seed=cfg.seed)
        self._agents = list(obs.keys())
        self.obs_dim = obs[self._agents[0]].shape[0]
        self.n_actions = int(self._env.action_space(self._agents[0]).n)

    @property
    def n_agents(self) -> int:
        return 2

    def reset(self, seed: int | None = None) -> np.ndarray:
        obs, _info = self._env.reset(seed=seed)
        self._agents = list(obs.keys())
        return np.stack([obs[a] for a in self._agents], axis=0).astype(np.float32)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        assert actions.shape == (2,), actions.shape
        act_dict = {a: int(act) for a, act in zip(self._agents, actions.tolist())}
        obs, rewards, terms, truncs, info = self._env.step(act_dict)
        if not obs:
            obs_arr = np.zeros((2, self.obs_dim), dtype=np.float32)
            r_arr = np.array([rewards.get(a, 0.0) for a in self._agents],
                             dtype=np.float32)
            return obs_arr, r_arr, True, info
        obs_arr = np.stack([obs[a] for a in self._agents], axis=0).astype(np.float32)
        r_arr = np.array([rewards[a] for a in self._agents], dtype=np.float32)
        done = bool(any(terms.values()) or any(truncs.values()))
        return obs_arr, r_arr, done, info

    def close(self) -> None:
        self._env.close()


ENVS = {"simple_spread": MPESimpleSpread, "simple_reference": MPESimpleReference}
