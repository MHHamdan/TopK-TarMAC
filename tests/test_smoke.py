"""Phase-0 smoke tests — must pass before any phase moves forward.

Verifies that the core MARL stack imports and that we can do a single env rollout step.
Each test is fast (sub-second) and runs without GPU.
"""

from __future__ import annotations


def test_import_torch():
    import torch  # noqa: F401

    assert hasattr(__import__("torch"), "Tensor")


def test_import_numpy_pandas_matplotlib():
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401
    import matplotlib  # noqa: F401


def test_mpe2_simple_spread_rollout():
    """Single-step MPE simple_spread rollout via the `mpe2` package.

    PettingZoo 1.25+ split MPE out into the `mpe2` package — this test pins the API
    we will use for actual experiments (parallel_env, dict obs/reward).
    """
    from mpe2 import simple_spread_v3

    env = simple_spread_v3.parallel_env(N=3, max_cycles=5, continuous_actions=False)
    obs, info = env.reset(seed=0)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    next_obs, rewards, terms, truncs, infos = env.step(actions)

    assert set(obs.keys()) == set(next_obs.keys())
    assert all(isinstance(r, (int, float)) for r in rewards.values())
    env.close()


def test_supersuit_pad():
    """Verify supersuit wrappers are importable (used for vectorization later)."""
    import supersuit as ss  # noqa: F401


def test_gymnasium_classic():
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=0)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs is not None
    env.close()


def test_seeding_utility_present():
    """The seeding utility will be filled in by Phase 4; placeholder import keeps
    Phase 0 honest about what exists today."""
    import importlib

    mod = importlib.import_module("src.utils.seeding")
    assert callable(mod.set_global_seed)
