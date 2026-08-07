"""Characterise the partially observable simple_spread variant.

A custom environment variant is a reviewer flag, and rightly so: it is the one
place where a favourable result could be manufactured by choosing a knob. This
measures exactly what the knob does, across the range of N the paper trains
on, so the modification can be judged rather than taken on trust.

The modification, stated precisely:

    simple_spread's observation is
        [self_vel(2), self_pos(2), landmark_rel(2L), peer_rel(2(N-1)),
         comm(2(N-1))]     with L = N landmarks, so obs_dim = 6N.
    For each agent i and peer j, if ||peer_rel[i,j]||_2 > R then peer_rel[i,j]
    is set to (0,0). Nothing else changes: no reward change, no action-space
    change, no change to landmark observations, and obs_dim is unchanged so
    the same network shape applies.

Zeroing rather than deleting keeps obs_dim fixed, at the cost of aliasing a
hidden peer with one at the observer's exact position. simple_spread agents
are volume-excluded, so exact coincidence does not arise; near-coincidence is
possible and is a documented limitation.

**The confound this script exists to expose.** simple_spread places N agents
and N landmarks in a world whose size does not grow with N. Agent density
therefore rises with N, so a *fixed* radius R hides a different fraction of
peers at each N. Any comparison across N in this environment is confounded by
that unless the visibility fraction is reported alongside, which is what the
table below is for.

Writes results/po_env_characterisation.json and .md.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from src.envs.mpe import MPEConfig, MPESimpleSpread

ROOT = Path(__file__).resolve().parent.parent


def measure(n: int, radius: float | None, episodes: int, seed0: int,
            random_actions: bool = True) -> dict[str, Any]:
    """Visibility fraction over whole episodes under a random policy.

    A random policy is the right probe: it is policy-independent, so the
    number characterises the environment rather than any particular agent.
    """
    env = MPESimpleSpread(MPEConfig(n_agents=n, max_cycles=25, seed=seed0,
                                    visibility_radius=radius))
    rng = np.random.default_rng(seed0)
    per_step: list[float] = []
    per_episode: list[float] = []
    for ep in range(episodes):
        obs = env.reset(seed=seed0 + ep)
        vals = [env.visible_fraction(obs)]
        done = False
        while not done:
            acts = (rng.integers(0, env.n_actions, size=n) if random_actions
                    else np.zeros(n, dtype=int))
            obs, _r, done, _i = env.step(acts)
            vals.append(env.visible_fraction(obs))
        per_step.extend(vals)
        per_episode.append(statistics.fmean(vals))
    env.close()
    return {
        "n_agents": n,
        "radius": radius,
        "obs_dim": env.obs_dim,
        "episodes": episodes,
        "mean_visible_fraction": statistics.fmean(per_step),
        "std_over_episodes": (statistics.pstdev(per_episode)
                              if len(per_episode) > 1 else 0.0),
        "min_episode_mean": min(per_episode),
        "max_episode_mean": max(per_episode),
        "mean_visible_peers": statistics.fmean(per_step) * (n - 1),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-list", nargs="+", type=int, default=[3, 6, 9, 12])
    p.add_argument("--radii", nargs="+", type=float,
                   default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default=str(ROOT / "results" / "po_env_characterisation.json"))
    p.add_argument("--md", default=str(ROOT / "results" / "po_env_characterisation.md"))
    args = p.parse_args()

    grid = []
    for n in args.n_list:
        for r in args.radii:
            row = measure(n, r, args.episodes, args.seed)
            grid.append(row)
            print(f"N={n:<3} R={r:<5} visible {row['mean_visible_fraction']:.3f} "
                  f"({row['mean_visible_peers']:.2f} of {n - 1} peers)  "
                  f"+/-{row['std_over_episodes']:.3f}", flush=True)
    # Full observability reference.
    full = [measure(n, None, args.episodes, args.seed) for n in args.n_list]

    payload = {
        "modification": {
            "env": "mpe2.simple_spread_v3",
            "obs_layout": "[self_vel(2), self_pos(2), landmark_rel(2N), "
                          "peer_rel(2(N-1)), comm(2(N-1))], obs_dim = 6N",
            "rule": "peer_rel[i,j] <- (0,0) if ||peer_rel[i,j]||_2 > R",
            "unchanged": ["reward", "action space", "landmark observations",
                          "obs_dim", "episode length", "local_ratio"],
            "aliasing_caveat": "a hidden peer is encoded identically to a peer "
                               "at the observer's exact position; agents are "
                               "volume-excluded so exact coincidence does not "
                               "occur, near-coincidence is possible",
            "code": "src/envs/mpe.py MPESimpleSpread._apply_visibility",
            "radius_used_in_paper": 1.0,
        },
        "probe_policy": "uniform random actions, 40 episodes x 25 steps",
        "grid": grid,
        "full_observability_reference": full,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# The partially observable simple_spread variant", "",
             "Generated by `scripts/characterise_po_env.py`. Probe policy is "
             "uniform random, so these numbers describe the environment, not a "
             "trained agent.", "",
             "## Modification", "",
             "```", "obs = [self_vel(2), self_pos(2), landmark_rel(2N), "
             "peer_rel(2(N-1)), comm(2(N-1))]   # obs_dim = 6N",
             "peer_rel[i,j] <- (0,0)   if   ||peer_rel[i,j]||_2 > R", "```", "",
             "Unchanged: reward, action space, landmark observations, obs_dim, "
             "episode length, `local_ratio`. Implementation: "
             "`src/envs/mpe.py`, `MPESimpleSpread._apply_visibility`. "
             "**R = 1.0 in every reported run.**", "",
             "## Mean fraction of peers visible", "",
             "| N | " + " | ".join(f"R={r}" for r in args.radii)
             + " | fully observable |",
             "|" + "---|" * (len(args.radii) + 2)]
    for n in args.n_list:
        cells = []
        for r in args.radii:
            row = next(g for g in grid if g["n_agents"] == n and g["radius"] == r)
            cells.append(f"{row['mean_visible_fraction']:.3f}")
        f = next(g for g in full if g["n_agents"] == n)
        lines.append(f"| {n} | " + " | ".join(cells)
                     + f" | {f['mean_visible_fraction']:.3f} |")

    lines += ["", "## Visible peers in absolute terms, at the R=1.0 used", "",
              "| N | peers total | peers visible | fraction | sd over episodes |",
              "|---|---|---|---|---|"]
    for n in args.n_list:
        row = next(g for g in grid if g["n_agents"] == n and g["radius"] == 1.0)
        lines.append(f"| {n} | {n - 1} | {row['mean_visible_peers']:.2f} | "
                     f"{row['mean_visible_fraction']:.3f} | "
                     f"{row['std_over_episodes']:.3f} |")

    lines += ["", "## The confound, stated explicitly", "",
              "simple_spread places N agents and N landmarks in a world whose "
              "extent does not grow with N, so agent density rises with N and a "
              "**fixed radius hides a different fraction of peers at each N**. "
              "The N=6 and N=12 arms therefore differ in task difficulty as "
              "well as in agent count, and any apparent N-trend in return "
              "confounds the two. The table above is what makes that visible; "
              "it is the reason no claim in this work is based on comparing "
              "returns across N in this environment. The N-scaling claims come "
              "from the training-free microbenchmark instead (D-027).", "",
              "A radius chosen per N to equalise the visible fraction would "
              "remove the confound and is the right design for a future "
              "N-scaling study; it was not used here because the training "
              "budget does not support an N sweep in any case.", ""]
    Path(args.md).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}\nwrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
