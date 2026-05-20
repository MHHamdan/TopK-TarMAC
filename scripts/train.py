"""Entry-point for a single training run."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.mappo import MAPPOConfig  # noqa: E402
from src.envs.mpe import MPEConfig, MPESimpleSpread  # noqa: E402
from src.training.train_mappo import TrainerConfig, train  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config file path.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-name", type=str, default=None,
                   help="Override run name from config.")
    p.add_argument("--total-steps", type=int, default=None,
                   help="Override total training steps.")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["env"]
    model_cfg_raw = cfg["model"]
    trainer_cfg_raw = cfg["trainer"]

    def make_env(seed: int) -> MPESimpleSpread:
        return MPESimpleSpread(MPEConfig(
            n_agents=env_cfg["n_agents"],
            max_cycles=env_cfg.get("max_cycles", 25),
            continuous_actions=False,
            local_ratio=env_cfg.get("local_ratio", 0.5),
            seed=seed,
        ))

    probe = make_env(0)
    model_cfg = MAPPOConfig(
        obs_dim=probe.obs_dim,
        n_actions=probe.n_actions,
        n_agents=probe.n_agents,
        hidden=model_cfg_raw.get("hidden", 64),
        use_comm=model_cfg_raw.get("use_comm", False),
        msg_dim=model_cfg_raw.get("msg_dim", 16),
        n_heads=model_cfg_raw.get("n_heads", 1),
        attn_mode=model_cfg_raw.get("attn_mode", "dense"),
        topk=model_cfg_raw.get("topk", 0),
    )
    probe.close()

    trainer_cfg = TrainerConfig(
        total_steps=args.total_steps or trainer_cfg_raw["total_steps"],
        n_envs=trainer_cfg_raw.get("n_envs", 8),
        rollout_len=trainer_cfg_raw.get("rollout_len", 25),
        ppo_epochs=trainer_cfg_raw.get("ppo_epochs", 10),
        n_minibatches=trainer_cfg_raw.get("n_minibatches", 1),
        lr=float(trainer_cfg_raw.get("lr", 7e-4)),
        gamma=float(trainer_cfg_raw.get("gamma", 0.99)),
        gae_lambda=float(trainer_cfg_raw.get("gae_lambda", 0.95)),
        clip_ratio=float(trainer_cfg_raw.get("clip_ratio", 0.2)),
        value_clip=float(trainer_cfg_raw.get("value_clip", 0.2)),
        entropy_coef=float(trainer_cfg_raw.get("entropy_coef", 0.01)),
        value_coef=float(trainer_cfg_raw.get("value_coef", 1.0)),
        max_grad_norm=float(trainer_cfg_raw.get("max_grad_norm", 10.0)),
        eval_episodes=int(trainer_cfg_raw.get("eval_episodes", 32)),
        log_interval_steps=int(trainer_cfg_raw.get("log_interval_steps", 2_000)),
        eval_interval_steps=int(trainer_cfg_raw.get("eval_interval_steps", 20_000)),
        device=trainer_cfg_raw.get("device", "cuda"),
        run_name=args.run_name or cfg.get("run_name", "mappo_default"),
        out_dir=trainer_cfg_raw.get("out_dir", "results"),
    )

    print(json.dumps({
        "config": args.config,
        "seed": args.seed,
        "run_name": trainer_cfg.run_name,
        "obs_dim": model_cfg.obs_dim, "n_actions": model_cfg.n_actions,
        "n_agents": model_cfg.n_agents,
    }, indent=2))
    out = train(make_env, model_cfg, trainer_cfg, seed=args.seed)
    print("done:", out)


if __name__ == "__main__":
    main()
