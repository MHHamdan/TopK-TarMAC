"""Measure per-forward FLOPs and effective-k of the AttentionComm module.

Used by make_figures.py to render the FLOPs/return Pareto plot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.agents.mappo import MAPPOAgent, MAPPOConfig

ROOT = Path(__file__).resolve().parent.parent


def attention_flops(N: int, k_eff: float, msg_dim: int = 16) -> dict:
    """Rough operation count for the AttentionComm forward.

    - QKV projection: N * (3 * msg_dim * embed_in)
    - Scores: N * N * msg_dim
    - Softmax + topk: O(N^2)
    - Output: nnz(attn) * msg_dim  where nnz = N * k_eff
    - Projection: N * msg_dim^2

    We focus on the dense vs sparse delta — the dominant term is the score
    matrix (N^2) for dense, and the (N * k_eff) message gather for sparse.
    """
    # The score *matrix* is always computed (we still softmax over N) so it
    # is identical dense vs sparse. What differs is the message *gather*:
    # dense aggregates N values per agent; topk aggregates k_eff.
    msg_gather = N * k_eff * msg_dim
    scores = N * N * msg_dim
    return {
        "msg_gather_ops": float(msg_gather),
        "score_ops": float(scores),
        "msg_gather_over_dense": float(k_eff / max(N - 1, 1)),
    }


def measure_run_effective_k(run: str, n_episodes: int = 16) -> dict:
    """Load a trained checkpoint and measure the mean effective-k it uses."""
    p = ROOT / "results" / run
    # Use first seed.
    seeds = sorted(p.glob("seed_*"))
    if not seeds:
        return {"error": f"no seeds in {p}"}
    ckpt = torch.load(seeds[0] / "final.pt", map_location="cpu", weights_only=False)
    cfg_dict = ckpt["model_cfg"]
    cfg = MAPPOConfig(**cfg_dict)
    agent = MAPPOAgent(cfg)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    if not cfg.use_comm:
        return {"run": run, "use_comm": False, "k_eff": float(cfg.n_agents - 1)}

    # Roll out a few episodes; record mean attention non-zero count.
    from src.envs.mpe import MPEConfig, MPESimpleSpread
    env = MPESimpleSpread(MPEConfig(n_agents=cfg.n_agents, max_cycles=25, seed=0))
    nnz_list = []
    with torch.no_grad():
        for ep in range(n_episodes):
            obs = env.reset(seed=ep + 10_000)
            done = False
            while not done:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0)
                # Trace nonzero count via attention weights.
                emb = torch.nn.functional.gelu(agent.obs_embed(obs_t))
                msg, info = agent.comm(emb)
                if "mean_k" in info:
                    nnz_list.append(info["mean_k"])
                elif "effective_k" in info:
                    nnz_list.append(info["effective_k"])
                else:
                    nnz_list.append(float(cfg.n_agents - 1))
                logits, _ = agent.policy_logits(obs_t)
                acts = logits.argmax(-1).squeeze(0).cpu().numpy()
                obs, _, done, _ = env.step(acts)
    env.close()

    k_eff = float(np.mean(nnz_list)) if nnz_list else float(cfg.n_agents - 1)
    return {
        "run": run, "use_comm": True, "n_agents": cfg.n_agents,
        "attn_mode": cfg.attn_mode, "k_eff": k_eff,
        "flops": attention_flops(cfg.n_agents, k_eff, cfg.msg_dim),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default=str(ROOT / "results" / "flops_summary.json"))
    args = p.parse_args()
    rows = [measure_run_effective_k(r) for r in args.runs]
    Path(args.out).write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
