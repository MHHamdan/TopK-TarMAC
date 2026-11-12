"""Descriptive summary of the exploratory Z-order arm (POST HOC, NOT PRE-REGISTERED).

`red_po6_zorder_k2` was added after every registered arm had been run and
inspected, in response to review. It is outside results/PREREGISTRATION.md and
outside the C1-C6 Holm family, so nothing here is a confirmatory test: the
output is mean +/- sd per arm and 95% bootstrap intervals of the difference in
means, computed exactly as scripts/summarise_reduced_design.py does (seeded
Random(0), 10,000 independent resamples of each arm, 2.5/97.5 percentiles).

It also measures what the selector actually picks: replaying each trained
Z-order policy greedily on the final-evaluation episodes (seeds 10000..10063,
as train_mappo.evaluate uses), it reports the fraction of Z-order-selected
peers that are among each agent's true k nearest neighbours (Euclidean, true
positions), against the k/(N-1) a uniformly random choice achieves.

Usage: .venv/bin/python scripts/summarise_zorder.py
Writes results/zorder_learning.json and results/zorder_learning.md.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarise_reduced_design import (  # noqa: E402
    boot_ci, cohens_d, final_returns, hodges_lehmann, perm_test)
from src.agents.mappo import SELF_POS, MAPPOAgent, MAPPOConfig  # noqa: E402
from src.envs.mpe import MPEConfig, MPESimpleSpread  # noqa: E402
from src.kernels.prescore_selection import (  # noqa: E402
    select_morton_window, select_radius_knn)

R = ROOT / "results"
ARM = "red_po6_zorder_k2"
COMPARATORS = [
    ("attention top-k (gather), k=2", "red_po6_gather_k2"),
    ("random-k, k=2", "red_po6_random_k2"),
    ("no communication", "red_po6_mappo"),
    ("dense attention", "red_po6_dense"),
]


def diff_ci(a: list[float], b: list[float]) -> tuple[float, float]:
    """Same estimator as summarise_reduced_design.analyse."""
    rng = random.Random(0)
    diffs = sorted(
        statistics.fmean(rng.choices(a, k=len(a)))
        - statistics.fmean(rng.choices(b, k=len(b)))
        for _ in range(10_000))
    return diffs[250], diffs[9750]


def knn_overlap(seed_dir: Path, n_episodes: int = 64,
                base_seed: int = 10_000) -> dict[str, float]:
    ck = torch.load(seed_dir / "final.pt", map_location="cpu", weights_only=False)
    mcfg = MAPPOConfig(**ck["model_cfg"])
    agent = MAPPOAgent(mcfg)
    agent.load_state_dict(ck["agent"])
    agent.eval()
    agent.comm.collect_info = False
    k = min(max(mcfg.topk, 1), mcfg.n_agents - 1)
    hits = tot = vis = 0
    vis_knn = 0
    with torch.no_grad():
        for ep in range(n_episodes):
            env = MPESimpleSpread(MPEConfig(n_agents=mcfg.n_agents,
                                            visibility_radius=1.0,
                                            seed=base_seed + ep))
            obs = env.reset()
            done = False
            while not done:
                o = torch.from_numpy(obs).float().unsqueeze(0)
                pos = o[..., SELF_POS]
                z = select_morton_window(pos, k)
                nn_ = select_radius_knn(pos, k)
                hits += (z.unsqueeze(-1) == nn_.unsqueeze(-2)).any(-1).sum().item()
                tot += z.numel()
                dist = torch.cdist(pos, pos)
                vis += (dist.gather(-1, z) <= 1.0).sum().item()
                vis_knn += (dist.gather(-1, nn_) <= 1.0).sum().item()
                logits, _ = agent.policy_logits(o)
                obs, _, done, _ = env.step(logits.argmax(-1).squeeze(0).numpy())
            env.close()
    return {"knn_fraction": hits / tot,
            "selected_within_radius": vis / tot,
            "knn_within_radius": vis_knn / tot,
            "n_selections": tot}


def main() -> int:
    z = final_returns(ARM)
    out: dict = {
        "arm": ARM,
        "status": "exploratory, post hoc, NOT pre-registered; descriptive only",
        "n_seeds": len(z),
        "final_return_per_seed": z,
        "final_return_mean": statistics.fmean(z),
        "final_return_sd": statistics.stdev(z),
        "final_return_boot95": boot_ci(z),
        "wall_time_s_per_seed": [
            json.loads((R / ARM / f"seed_{s}" / "final_eval.json").read_text())
            ["wall_time_s"] for s in range(len(z))],
        "comparators": {}, "contrasts": {},
    }
    for label, arm in COMPARATORS:
        b = final_returns(arm)
        out["comparators"][arm] = {
            "label": label, "n_seeds": len(b),
            "mean": statistics.fmean(b), "sd": statistics.stdev(b),
            "boot95": boot_ci(b)}
        out["contrasts"][f"{ARM} - {arm}"] = {
            "mean_diff": statistics.fmean(z) - statistics.fmean(b),
            "boot95_diff": diff_ci(z, b),
            "hodges_lehmann": hodges_lehmann(z, b),
            "cohens_d": cohens_d(z, b),
            "perm_p_uncorrected_descriptive": perm_test(z, b),
        }

    per_seed = [knn_overlap(R / ARM / f"seed_{s}") for s in range(len(z))]
    n_agents, k = 6, 2
    out["selection"] = {
        "per_seed": per_seed,
        "knn_fraction_mean": statistics.fmean(p["knn_fraction"] for p in per_seed),
        "knn_fraction_chance_random_k": k / (n_agents - 1),
        "selected_within_radius_mean": statistics.fmean(
            p["selected_within_radius"] for p in per_seed),
        "knn_within_radius_mean": statistics.fmean(
            p["knn_within_radius"] for p in per_seed),
        "protocol": "greedy rollouts of each trained Z-order policy on the 64 "
                    "final-eval episodes (env seeds 10000-10063); true "
                    "positions; ties impossible in continuous MPE",
    }
    (R / "zorder_learning.json").write_text(json.dumps(out, indent=2) + "\n")

    s = out["selection"]
    md = [
        "# Z-order peer selection in the learning study (exploratory)", "",
        "**Post hoc, NOT pre-registered.** Added after all registered arms were "
        "run and inspected, in response to review. Outside "
        "`results/PREREGISTRATION.md` and the C1-C6 Holm family; descriptive "
        "only. Generated by `scripts/summarise_zorder.py`.", "",
        "Arm `red_po6_zorder_k2`: partially observable simple_spread, N=6, "
        "visibility radius 1.0, k=2, 2M steps, 5 seeds; every hyperparameter "
        "identical to `red_po6_gather_k2` and `red_po6_random_k2` "
        "(`configs/reduced/red_po6_zorder_k2.yaml`). Peers are fixed by "
        "adjacency along a Morton curve of agent positions before any score "
        "is formed; only the k retained pairs are scored.", "",
        "Positions: each agent's own absolute position, `obs[..., 2:4]`, which "
        "the environment gives every agent and the visibility mask never "
        "zeroes. The selector pools all N self-reported positions, so a peer "
        "outside an agent's visibility radius can be selected by position -- "
        "the same channel through which every arm's keys and values already "
        "carry each agent's full observation. No simulator state outside the "
        "agents' observations is used.", "",
        "| Arm | seeds | final return mean ± sd | 95% bootstrap CI |",
        "|---|---|---|---|",
        f"| `{ARM}` (Z-order) | {len(z)} | {out['final_return_mean']:.2f} ± "
        f"{out['final_return_sd']:.2f} | [{out['final_return_boot95'][0]:.2f}, "
        f"{out['final_return_boot95'][1]:.2f}] |"]
    for arm, c in out["comparators"].items():
        md.append(f"| `{arm}` ({c['label']}) | {c['n_seeds']} | {c['mean']:.2f} "
                  f"± {c['sd']:.2f} | [{c['boot95'][0]:.2f}, "
                  f"{c['boot95'][1]:.2f}] |")
    md += ["", "| Difference (Z-order − B) | mean | 95% bootstrap CI | "
           "Hodges-Lehmann | Cohen's d | exact perm p (uncorrected) |",
           "|---|---|---|---|---|---|"]
    for name, c in out["contrasts"].items():
        md.append(f"| {name.split(' - ')[1]} | {c['mean_diff']:+.2f} | "
                  f"[{c['boot95_diff'][0]:+.2f}, {c['boot95_diff'][1]:+.2f}] | "
                  f"{c['hodges_lehmann']:+.2f} | {c['cohens_d']:+.2f} | "
                  f"{c['perm_p_uncorrected_descriptive']:.3f} |")
    md += ["", "## What the selector picks", "",
           f"Fraction of Z-order-selected peers that are among the agent's true "
           f"2 nearest neighbours: **{s['knn_fraction_mean']:.3f}** (per seed: "
           + ", ".join(f"{p['knn_fraction']:.3f}" for p in per_seed)
           + f"); a uniformly random choice gives {s['knn_fraction_chance_random_k']:.3f}. "
           f"Selected peers inside the visibility radius: "
           f"{s['selected_within_radius_mean']:.3f} (true 2-NN: "
           f"{s['knn_within_radius_mean']:.3f}). {s['protocol']}.", ""]
    (R / "zorder_learning.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
