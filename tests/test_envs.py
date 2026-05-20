"""Phase-2 env wrapper tests."""

from __future__ import annotations

import numpy as np

from src.envs.mpe import MPEConfig, MPESimpleSpread


def test_mpe_shapes_n3():
    env = MPESimpleSpread(MPEConfig(n_agents=3, max_cycles=10, seed=0))
    obs = env.reset()
    assert obs.shape == (3, env.obs_dim)
    actions = np.zeros(3, dtype=np.int64)
    next_obs, r, done, info = env.step(actions)
    assert next_obs.shape == (3, env.obs_dim)
    assert r.shape == (3,)
    env.close()


def test_mpe_scales_to_12():
    env = MPESimpleSpread(MPEConfig(n_agents=12, max_cycles=5, seed=0))
    obs = env.reset()
    assert obs.shape == (12, env.obs_dim)
    for _ in range(5):
        actions = np.random.randint(0, env.n_actions, size=12)
        obs, r, done, info = env.step(actions)
        if done:
            obs = env.reset()
    env.close()
