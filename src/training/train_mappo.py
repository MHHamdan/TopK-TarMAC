"""MAPPO training loop on stacked-tensor MPE envs.

Designed to be small enough to read end-to-end. Logs to a CSV under
results/<run_name>/metrics.csv and writes the final checkpoint to
results/<run_name>/final.pt.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from src.agents.mappo import MAPPOAgent, MAPPOConfig
from src.training.rollout import compute_gae
from src.utils.seeding import set_global_seed


@dataclass
class TrainerConfig:
    # Algorithm
    total_steps: int = 1_000_000
    n_envs: int = 16
    rollout_len: int = 100
    ppo_epochs: int = 10
    n_minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    # Absolute clip on the per-update value change. Set high enough that it
    # never bites on our reward scale; the policy clip plus advantage
    # normalisation give the trust region.
    value_clip: float = 100.0
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    advantage_norm: bool = True
    # Per-agent reward = team_reward / N_agents. Keeps scale ~constant as we
    # sweep N.
    per_agent_reward: bool = True

    # Logging
    log_interval_steps: int = 2_000  # write a row this often
    eval_interval_steps: int = 20_000
    eval_episodes: int = 32
    device: str = "cuda"
    run_name: str = "mappo_default"
    out_dir: str = "results"


class VecEnv:
    """Trivial Python-level vec env: stacks rollouts from N processes serially.

    For MPE simple_spread on CPU this is fast enough — measured >5k env-steps
    per second at N_agents=12 with 8 envs.
    """

    def __init__(self, make_env: Callable[[int], "VecEnvCompatible"], n_envs: int, base_seed: int):
        self.envs = [make_env(base_seed + i) for i in range(n_envs)]
        self.n_envs = n_envs
        self.n_agents = self.envs[0].n_agents
        self.obs_dim = self.envs[0].obs_dim
        self.n_actions = self.envs[0].n_actions
        self._returns = np.zeros(n_envs, dtype=np.float32)
        self._last_completed: list[float] = []

    def reset(self, seeds: list[int] | None = None) -> np.ndarray:
        seeds = seeds or [None] * self.n_envs
        obs = np.stack([env.reset(s) for env, s in zip(self.envs, seeds)], axis=0)
        self._returns[:] = 0.0
        return obs  # (E, N, obs_dim)

    def step(self, actions: np.ndarray, reward_scale: float = 1.0
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        next_obs_list, rewards_list, dones_list = [], [], []
        for i, env in enumerate(self.envs):
            next_obs, r, done, _ = env.step(actions[i])
            team_r = float(r.sum())  # reported episode return uses raw team sum
            self._returns[i] += team_r
            if done:
                self._last_completed.append(float(self._returns[i]))
                self._returns[i] = 0.0
                next_obs = env.reset()
            next_obs_list.append(next_obs)
            rewards_list.append(team_r * reward_scale)  # what the algo trains on
            dones_list.append(done)
        return (np.stack(next_obs_list, 0),
                np.array(rewards_list, dtype=np.float32),
                np.array(dones_list, dtype=np.float32))

    def pop_completed_returns(self, max_n: int = 64) -> list[float]:
        out = self._last_completed[-max_n:]
        self._last_completed = []
        return out


class VecEnvCompatible:
    """Anything with n_agents, obs_dim, n_actions, reset(seed), step(actions)."""
    n_agents: int
    obs_dim: int
    n_actions: int

    def reset(self, seed: int | None = None) -> np.ndarray: ...
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]: ...


def evaluate(agent: MAPPOAgent, make_env: Callable[[int], VecEnvCompatible],
             n_episodes: int, device: str, base_seed: int = 10_000) -> dict[str, float]:
    agent.eval()
    returns = []
    with torch.no_grad():
        for ep in range(n_episodes):
            env = make_env(base_seed + ep)
            obs = env.reset()
            ep_return = 0.0
            done = False
            while not done:
                obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)  # (1,N,D)
                logits, _ = agent.policy_logits(obs_t)
                # Greedy at eval — Yu et al. 2022 follow this convention.
                actions = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
                obs, r, done, _ = env.step(actions)
                ep_return += float(r.sum())
            returns.append(ep_return)
            env.close()
    agent.train()
    arr = np.array(returns)
    return {"eval_return_mean": float(arr.mean()), "eval_return_std": float(arr.std())}


def train(make_env: Callable[[int], VecEnvCompatible],
          model_cfg: MAPPOConfig,
          trainer_cfg: TrainerConfig,
          seed: int) -> str:
    set_global_seed(seed)

    out_dir = Path(trainer_cfg.out_dir) / trainer_cfg.run_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"model": asdict(model_cfg), "trainer": asdict(trainer_cfg),
                    "seed": seed}, indent=2)
    )

    device = trainer_cfg.device if torch.cuda.is_available() else "cpu"
    venv = VecEnv(make_env, trainer_cfg.n_envs, base_seed=seed * 1000)
    assert venv.n_agents == model_cfg.n_agents
    assert venv.obs_dim == model_cfg.obs_dim
    assert venv.n_actions == model_cfg.n_actions

    agent = MAPPOAgent(model_cfg).to(device)
    optim = Adam(agent.parameters(), lr=trainer_cfg.lr, eps=1e-5)

    obs = venv.reset()  # (E, N, obs_dim)
    obs_t = torch.from_numpy(obs).float().to(device)

    T = trainer_cfg.rollout_len
    E = trainer_cfg.n_envs
    N = model_cfg.n_agents

    # Pre-allocate rollout buffers
    buf_obs = torch.zeros(T, E, N, model_cfg.obs_dim, device=device)
    buf_act = torch.zeros(T, E, N, dtype=torch.long, device=device)
    buf_logp = torch.zeros(T, E, N, device=device)
    buf_val = torch.zeros(T + 1, E, device=device)
    buf_rew = torch.zeros(T, E, device=device)
    buf_done = torch.zeros(T, E, device=device)

    csv_path = out_dir / "metrics.csv"
    csv_f = csv_path.open("w", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=[
        "global_step", "wall_time_s", "ep_return_mean", "ep_return_std",
        "policy_loss", "value_loss", "entropy", "approx_kl",
        "eval_return_mean", "eval_return_std",
    ])
    writer.writeheader()

    global_step = 0
    last_log_step = 0
    last_eval_step = 0
    start = time.time()
    last_eval_metrics: dict[str, float] = {}

    while global_step < trainer_cfg.total_steps:
        # Collect a rollout of T steps.
        for t in range(T):
            with torch.no_grad():
                logits, _info = agent.policy_logits(obs_t)
                dist = Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)
                v = agent.value(obs_t)
            buf_obs[t] = obs_t
            buf_act[t] = act
            buf_logp[t] = logp
            buf_val[t] = v
            obs_np, r_np, done_np = venv.step(
                act.cpu().numpy(),
                reward_scale=(1.0 / N) if trainer_cfg.per_agent_reward else 1.0,
            )
            obs_t = torch.from_numpy(obs_np).float().to(device)
            buf_rew[t] = torch.from_numpy(r_np).to(device)
            buf_done[t] = torch.from_numpy(done_np).to(device)
            global_step += E

        # Bootstrap final value.
        with torch.no_grad():
            buf_val[T] = agent.value(obs_t)
        advs, rets = compute_gae(
            buf_rew, buf_val, buf_done,
            gamma=trainer_cfg.gamma, lam=trainer_cfg.gae_lambda,
        )

        if trainer_cfg.advantage_norm:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Flatten time × env for PPO updates; keep agent dim.
        flat_obs = buf_obs.reshape(T * E, N, model_cfg.obs_dim)
        flat_act = buf_act.reshape(T * E, N)
        flat_logp_old = buf_logp.reshape(T * E, N)
        flat_advs = advs.reshape(T * E)
        flat_rets = rets.reshape(T * E)
        flat_val_old = buf_val[:T].reshape(T * E)

        n_samples = T * E
        mb_size = n_samples // trainer_cfg.n_minibatches
        idxs = np.arange(n_samples)

        epoch_metrics: dict[str, list[float]] = {"pl": [], "vl": [], "ent": [], "kl": []}
        for _ in range(trainer_cfg.ppo_epochs):
            np.random.shuffle(idxs)
            for start_idx in range(0, n_samples, mb_size):
                mb = idxs[start_idx:start_idx + mb_size]
                mb_t = torch.from_numpy(mb).long().to(device)
                obs_mb = flat_obs[mb_t]
                act_mb = flat_act[mb_t]
                logp_old_mb = flat_logp_old[mb_t]
                adv_mb = flat_advs[mb_t]
                ret_mb = flat_rets[mb_t]
                val_old_mb = flat_val_old[mb_t]

                logits, _ = agent.policy_logits(obs_mb)
                dist = Categorical(logits=logits)
                logp_new = dist.log_prob(act_mb)
                entropy = dist.entropy().mean()
                # Per-agent ratio, average advantage broadcast across agents.
                ratio = (logp_new - logp_old_mb).exp()
                adv_mb_n = adv_mb.unsqueeze(-1).expand(-1, N)
                surr1 = ratio * adv_mb_n
                surr2 = torch.clamp(ratio,
                                    1.0 - trainer_cfg.clip_ratio,
                                    1.0 + trainer_cfg.clip_ratio) * adv_mb_n
                policy_loss = -torch.min(surr1, surr2).mean()

                v_new = agent.value(obs_mb)
                v_clipped = val_old_mb + (v_new - val_old_mb).clamp(
                    -trainer_cfg.value_clip, trainer_cfg.value_clip
                )
                v_loss1 = (v_new - ret_mb).pow(2)
                v_loss2 = (v_clipped - ret_mb).pow(2)
                value_loss = torch.max(v_loss1, v_loss2).mean()

                loss = (policy_loss
                        + trainer_cfg.value_coef * value_loss
                        - trainer_cfg.entropy_coef * entropy)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), trainer_cfg.max_grad_norm)
                optim.step()

                with torch.no_grad():
                    approx_kl = (logp_old_mb - logp_new).mean().clamp_min(0.0)

                epoch_metrics["pl"].append(policy_loss.item())
                epoch_metrics["vl"].append(value_loss.item())
                epoch_metrics["ent"].append(entropy.item())
                epoch_metrics["kl"].append(approx_kl.item())

        # Logging.
        if global_step - last_log_step >= trainer_cfg.log_interval_steps:
            completed = venv.pop_completed_returns()
            ep_mean = float(np.mean(completed)) if completed else float("nan")
            ep_std = float(np.std(completed)) if completed else float("nan")
            row = {
                "global_step": global_step,
                "wall_time_s": time.time() - start,
                "ep_return_mean": ep_mean,
                "ep_return_std": ep_std,
                "policy_loss": float(np.mean(epoch_metrics["pl"])),
                "value_loss": float(np.mean(epoch_metrics["vl"])),
                "entropy": float(np.mean(epoch_metrics["ent"])),
                "approx_kl": float(np.mean(epoch_metrics["kl"])),
                **last_eval_metrics,
            }
            writer.writerow(row)
            csv_f.flush()
            last_log_step = global_step

        if global_step - last_eval_step >= trainer_cfg.eval_interval_steps:
            last_eval_metrics = evaluate(agent, make_env,
                                         n_episodes=trainer_cfg.eval_episodes,
                                         device=device)
            last_eval_step = global_step

    # Final eval + checkpoint.
    final_eval = evaluate(agent, make_env,
                          n_episodes=trainer_cfg.eval_episodes * 2,
                          device=device)
    torch.save({
        "agent": agent.state_dict(),
        "model_cfg": asdict(model_cfg),
        "trainer_cfg": asdict(trainer_cfg),
        "seed": seed,
        "final_eval": final_eval,
    }, out_dir / "final.pt")
    (out_dir / "final_eval.json").write_text(json.dumps(final_eval, indent=2))
    csv_f.close()
    return str(out_dir)
