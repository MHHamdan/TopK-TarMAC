"""Measure random-policy baseline for each N. Used as the comparator floor
in the headline table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.envs.mpe import MPEConfig, MPESimpleSpread


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "random_baseline.json"


def main(N_list=(3, 6, 12), n_episodes: int = 40) -> None:
    out = {}
    for N in N_list:
        returns = []
        rng = np.random.RandomState(0)
        for seed in range(n_episodes):
            env = MPESimpleSpread(MPEConfig(n_agents=N, max_cycles=25, seed=seed))
            obs = env.reset()
            total = 0.0
            for _ in range(25):
                a = rng.randint(0, env.n_actions, size=N)
                obs, r, done, _ = env.step(a)
                total += float(r.sum())
            returns.append(total)
            env.close()
        arr = np.array(returns)
        out[str(N)] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "n_episodes": n_episodes,
        }
        print(f"Random N={N}: {arr.mean():.2f} +/- {arr.std():.2f} (n={n_episodes})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
