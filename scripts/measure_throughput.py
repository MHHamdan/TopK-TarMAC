"""Measure rollout throughput (env-steps/sec) per agent count and arm.

Feeds the GPU-hour projections in ``results/compute_budget.md``. Rollout
collection, not the gradient update, is what dominates wall-clock in this
codebase, so this is what a budget must be built from.

Writes ``results/throughput.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.agents.mappo import MAPPOAgent, MAPPOConfig
from src.envs.mpe import MPEConfig, MPESimpleSpread
from src.training.train_mappo import VecEnv

ROOT = Path(__file__).resolve().parent.parent

_MODES = {"mappo": "dense", "dense": "dense", "topk": "adaptive_topk"}


def probe(n_agents: int, arm: str, device: str, n_envs: int = 32,
          rollout_len: int = 25, iters: int = 2) -> float:
    """Env-steps per second for one (agent count, arm) pair.

    Args:
        n_agents: Number of agents in the environment.
        arm: One of ``mappo``, ``dense``, ``topk``.
        device: Torch device string.
        n_envs: Parallel environments, matching the training config.
        rollout_len: Steps per rollout segment.
        iters: Rollout segments to time.

    Returns:
        Measured environment steps per second.
    """
    def make(seed: int) -> MPESimpleSpread:
        return MPESimpleSpread(MPEConfig(n_agents=n_agents, max_cycles=25, seed=seed))

    venv = VecEnv(make, n_envs, base_seed=0)
    cfg = MAPPOConfig(
        obs_dim=venv.obs_dim, n_actions=venv.n_actions, n_agents=n_agents,
        hidden=64, use_comm=(arm != "mappo"), msg_dim=16,
        attn_mode=_MODES[arm],
    )
    agent = MAPPOAgent(cfg).to(device)
    obs = torch.from_numpy(venv.reset()).float().to(device)

    with torch.no_grad():
        for _ in range(2):
            agent.policy_logits(obs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0, steps = time.time(), 0
    for _ in range(iters):
        for _t in range(rollout_len):
            with torch.no_grad():
                logits, _ = agent.policy_logits(obs)
                act = torch.distributions.Categorical(logits=logits).sample()
                agent.value(obs)
            nxt, _r, _d = venv.step(act.cpu().numpy())
            obs = torch.from_numpy(nxt).float().to(device)
            steps += n_envs
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return steps / (time.time() - t0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--n-list", nargs="+", type=int, default=[3, 6, 12, 24])
    p.add_argument("--arms", nargs="+", default=["mappo", "dense", "topk"])
    p.add_argument("--iters", type=int, default=2)
    p.add_argument("--out", default=str(ROOT / "results" / "throughput.json"))
    args = p.parse_args()

    out: dict[str, float] = {}
    print(f"{'N':>4} {'arm':<7} {'env-steps/s':>13}")
    for n in args.n_list:
        for arm in args.arms:
            sps = probe(n, arm, args.device, iters=args.iters)
            out[f"N{n}_{arm}"] = round(sps, 1)
            print(f"{n:>4} {arm:<7} {sps:>13.0f}", flush=True)

    Path(args.out).write_text(json.dumps({
        "device": args.device,
        "n_envs": 32,
        "rollout_len": 25,
        "env_steps_per_sec": out,
    }, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
