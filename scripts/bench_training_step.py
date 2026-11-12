"""Where a multi-agent training step actually spends its time.

Every latency result in this study is measured with the training loop removed,
which is the right way to compare attention implementations but leaves open what
the difference is worth end to end. This script closes that gap by timing one
complete MAPPO iteration -- environment stepping, action selection, the PPO
update and the optimiser step -- with the communication module swapped between
implementations and everything else held fixed.

The measurement is deliberately coarse. Phase boundaries are separated by device
synchronisations, so the numbers include synchronisation cost and cannot be
compared with the microbenchmark's kernel-level figures; they answer only the
question of what share of a training step the communication module occupies, and
therefore what an improvement to it could be worth.

Writes results/training_step_timing.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.mappo import MAPPOAgent, MAPPOConfig  # noqa: E402
from src.envs.mpe import MPEConfig, MPESimpleSpread  # noqa: E402
from src.training.train_mappo import VecEnv  # noqa: E402
from src.training.rollout import compute_gae  # noqa: E402

ARMS = {
    "no_comm": {"use_comm": False},
    "dense": {"use_comm": True, "attn_mode": "dense"},
    "dense_sdpa": {"use_comm": True, "attn_mode": "dense_sdpa"},
    "topk_masked": {"use_comm": True, "attn_mode": "topk"},
    "topk_gather": {"use_comm": True, "attn_mode": "topk_gather"},
}


class Phase:
    """Wall-clock timer that synchronises the device at both ends."""

    def __init__(self, store: dict[str, float], name: str, device: str):
        self.store, self.name, self.device = store, name, device

    def __enter__(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        self.store[self.name] = self.store.get(self.name, 0.0) + (
            time.perf_counter() - self.t0) * 1e3
        return False


def one_iteration(agent: MAPPOAgent, optim: Adam, venv: VecEnv,
                  obs_t: torch.Tensor, cfg: MAPPOConfig, device: str,
                  T: int, E: int, ppo_epochs: int, n_minibatches: int
                  ) -> tuple[dict[str, float], torch.Tensor]:
    """One rollout plus one PPO update, with each phase timed separately."""
    ph: dict[str, float] = {}
    N = cfg.n_agents
    buf_obs = torch.zeros(T, E, N, cfg.obs_dim, device=device)
    buf_act = torch.zeros(T, E, N, dtype=torch.long, device=device)
    buf_logp = torch.zeros(T, E, N, device=device)
    buf_val = torch.zeros(T + 1, E, device=device)
    buf_rew = torch.zeros(T, E, device=device)
    buf_done = torch.zeros(T, E, device=device)

    for t in range(T):
        with Phase(ph, "policy_forward", device), torch.no_grad():
            logits, _ = agent.policy_logits(obs_t)
            dist = Categorical(logits=logits)
            act = dist.sample()
            buf_logp[t] = dist.log_prob(act)
            buf_val[t] = agent.value(obs_t)
        buf_obs[t] = obs_t
        buf_act[t] = act
        with Phase(ph, "environment", device):
            obs_np, r_np, done_np = venv.step(act.cpu().numpy(),
                                              reward_scale=1.0 / N)
        obs_t = torch.from_numpy(obs_np).float().to(device)
        buf_rew[t] = torch.from_numpy(r_np).to(device)
        buf_done[t] = torch.from_numpy(done_np).to(device)

    with Phase(ph, "advantage", device), torch.no_grad():
        buf_val[T] = agent.value(obs_t)
        advs, rets = compute_gae(buf_rew, buf_val, buf_done, gamma=0.99,
                                 lam=0.95)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    flat_obs = buf_obs.reshape(T * E, N, cfg.obs_dim)
    flat_act = buf_act.reshape(T * E, N)
    flat_logp_old = buf_logp.reshape(T * E, N)
    flat_advs = advs.reshape(T * E)
    flat_rets = rets.reshape(T * E)
    n_samples = T * E
    mb = n_samples // n_minibatches
    idxs = np.arange(n_samples)

    for _ in range(ppo_epochs):
        np.random.shuffle(idxs)
        for s in range(0, n_samples, mb):
            sel = torch.from_numpy(idxs[s:s + mb]).long().to(device)
            with Phase(ph, "update_forward", device):
                logits, _ = agent.policy_logits(flat_obs[sel])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(flat_act[sel])
                ratio = (logp - flat_logp_old[sel]).exp()
                adv = flat_advs[sel].unsqueeze(-1).expand(-1, N)
                pl = -torch.min(ratio * adv,
                                ratio.clamp(0.8, 1.2) * adv).mean()
                vl = (agent.value(flat_obs[sel]) - flat_rets[sel]).pow(2).mean()
                loss = pl + 0.5 * vl - 0.01 * dist.entropy().mean()
            with Phase(ph, "update_backward", device):
                optim.zero_grad(set_to_none=True)
                loss.backward()
            with Phase(ph, "optimiser", device):
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optim.step()
    return ph, obs_t


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--n-agents", nargs="+", type=int, default=[6, 12, 24])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--rollout-len", type=int, default=100)
    p.add_argument("--ppo-epochs", type=int, default=10)
    p.add_argument("--n-minibatches", type=int, default=4)
    p.add_argument("--iterations", type=int, default=5,
                   help="timed iterations, after --warmup untimed ones")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--k-frac", type=float, default=0.25)
    p.add_argument("--out", default=str(ROOT / "results" /
                                        "training_step_timing.json"))
    args = p.parse_args()

    device = args.device
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    rows: list[dict[str, Any]] = []
    t_start = time.time()

    for N in args.n_agents:
        for arm in args.arms:
            torch.manual_seed(0)
            np.random.seed(0)
            k = max(1, int(args.k_frac * (N - 1)))
            make_env = (lambda i, N=N: MPESimpleSpread(
                MPEConfig(n_agents=N, seed=i, visibility_radius=1.0)))
            venv = VecEnv(make_env, args.n_envs, base_seed=0)
            cfg = MAPPOConfig(obs_dim=venv.obs_dim, n_actions=venv.n_actions,
                              n_agents=N, topk=k, **ARMS[arm])
            agent = MAPPOAgent(cfg).to(device)
            for m in agent.modules():
                if hasattr(m, "collect_info"):
                    m.collect_info = False
            optim = Adam(agent.parameters(), lr=3e-4, eps=1e-5)
            obs_t = torch.from_numpy(venv.reset()).float().to(device)

            # Two discarded iterations, not one: the first pays for lazy CUDA
            # module loading and autotuning caches that the later ones inherit,
            # and with a single discard the implementation that happens to run
            # first carries that cost into its reported time.
            per_iter: list[dict[str, float]] = []
            for it in range(args.iterations + args.warmup):
                ph, obs_t = one_iteration(
                    agent, optim, venv, obs_t, cfg, device,
                    args.rollout_len, args.n_envs, args.ppo_epochs,
                    args.n_minibatches)
                if it >= args.warmup:
                    per_iter.append(ph)

            keys = sorted(per_iter[0])
            summary = {k2: statistics.median([p[k2] for p in per_iter])
                       for k2 in keys}
            total = sum(summary.values())
            env_steps = args.rollout_len * args.n_envs
            row = {
                "n_agents": N, "arm": arm, "k": k,
                "n_envs": args.n_envs, "rollout_len": args.rollout_len,
                "ppo_epochs": args.ppo_epochs,
                "phase_ms": summary,
                "phase_share": {k2: summary[k2] / total for k2 in keys},
                "iteration_ms": total,
                "env_steps_per_iteration": env_steps,
                "env_steps_per_s": env_steps / (total * 1e-3),
                "iterations_timed": len(per_iter),
                "iteration_ms_spread_pct": 100.0 * (
                    max(sum(p.values()) for p in per_iter)
                    - min(sum(p.values()) for p in per_iter)) / min(
                    sum(p.values()) for p in per_iter),
            }
            rows.append(row)
            print(f"N={N:>3} {arm:<12} iter {total:9.1f} ms  "
                  f"{row['env_steps_per_s']:8.0f} env-steps/s  "
                  f"env {100*row['phase_share']['environment']:4.1f}%  "
                  f"fwd {100*row['phase_share']['policy_forward']:4.1f}%  "
                  f"upd-fwd {100*row['phase_share']['update_forward']:4.1f}%  "
                  f"bwd {100*row['phase_share']['update_backward']:4.1f}%  "
                  f"opt {100*row['phase_share']['optimiser']:4.1f}%  "
                  f"(+/-{row['iteration_ms_spread_pct']:.1f}%)", flush=True)
            del agent, optim, venv
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    payload = {
        "device": device,
        "device_name": (torch.cuda.get_device_name(device)
                        if device.startswith("cuda") else "cpu"),
        "torch": torch.__version__,
        "timing": "wall_clock_with_device_sync_at_phase_boundaries",
        "iterations_per_arm": args.iterations,
        "elapsed_s": round(time.time() - t_start, 1),
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
