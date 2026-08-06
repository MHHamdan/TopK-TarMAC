"""Emit the Phase-C reduced-design configs.

The design is capped at 12 training arms and N <= 12 agents. Everything that
scales in N moved to the training-free microbenchmark (N up to 512), because
training throughput here is environment-bound and cannot test a cost claim
(D-018), while an N=24 cell alone costs ~197 GPU-h (D-017).

The 12 arms, in the order they must be run -- each priority tier is only
interpretable if the tier above it came out the right way:

  P1 positive control (4)  Does the channel do anything at all? On the
                           historical fully-observable simple_spread, "sparse
                           is free" and "the channel is unused" predict the
                           same thing, so no sparsification result there is
                           interpretable (D-011). Two independent controls:
                           simple_reference (channel logically necessary,
                           fixed N=2) and partially-observable simple_spread
                           (channel useful, N free).
  P2 random-k control (2)  Does the *selection rule* carry information, or is
                           aggregating over any k peers enough? Matched k,
                           matched FLOPs, peers chosen at random.
  P3 fixed-k grid (3)      How does return vary with k, holding the selection
                           rule fixed. The n=0 ablation cells of D-12.
  P4 mechanism arms (3)    Lagrangian budget (the only route to a positive
                           headline) and entmax (the manuscript's own proposed
                           alternative to the gate).

Usage: .venv/bin/python scripts/make_reduced_design_configs.py
Writes configs/reduced/*.yaml and configs/reduced/MANIFEST.json
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "configs" / "reduced"

# Steps per run. 2M is the reduced-design budget; the historical runs used
# 800k and did not separate arms, so this is a deliberate increase per cell
# paid for by dropping the N sweep.
STEPS = 2_000_000
SEEDS = 5

# Radius 1.0 leaves ~33% of peers visible at N=6 (measured, not assumed --
# see MPESimpleSpread.visible_fraction). Small enough that the channel has
# something to carry, large enough that the task stays learnable.
VIS_RADIUS = 1.0

BASE_TRAINER = {
    "total_steps": STEPS,
    "n_envs": 32,
    "rollout_len": 25,
    "ppo_epochs": 10,
    "n_minibatches": 1,
    "lr": 7.0e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "value_clip": 100.0,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "per_agent_reward": True,
    "advantage_norm": True,
    "eval_episodes": 32,
    "log_interval_steps": 4_000,
    "eval_interval_steps": 40_000,
    "device": "cuda",
    "out_dir": "results",
}


def arm(name: str, tier: int, rationale: str, env: dict, model: dict,
        trainer: dict | None = None) -> dict:
    return {
        "run_name": name,
        "tier": tier,
        "rationale": rationale,
        "env": env,
        "model": model,
        "trainer": {**BASE_TRAINER, **(trainer or {})},
    }


def spread(n: int, po: bool = True) -> dict:
    e = {"name": "simple_spread", "n_agents": n, "max_cycles": 25,
         "local_ratio": 0.5}
    if po:
        e["visibility_radius"] = VIS_RADIUS
    return e


REFERENCE = {"name": "simple_reference", "n_agents": 2, "max_cycles": 25,
             "local_ratio": 0.5}

NO_COMM = {"hidden": 64, "use_comm": False}


def comm(mode: str, **kw) -> dict:
    return {"hidden": 64, "use_comm": True, "msg_dim": 16,
            "attn_mode": mode, **kw}


ARMS = [
    # ---- P1: is there a channel effect to sparsify at all? ---------------
    arm("red_ref_mappo", 1,
        "Comm-free floor on a task that is unsolvable without communication. "
        "If red_ref_dense does not beat this, the comm module is broken and "
        "nothing downstream is interpretable.",
        REFERENCE, NO_COMM),
    arm("red_ref_dense", 1,
        "Dense attention-comm on the communication-critical control. This is "
        "the positive control D-011 makes a precondition for any headline.",
        REFERENCE, comm("dense")),
    arm("red_po6_mappo", 1,
        "Comm-free floor on partially-observable simple_spread, N=6. Unlike "
        "simple_reference this scales in N, so it carries the k arms.",
        spread(6), NO_COMM),
    arm("red_po6_dense", 1,
        "Dense attention-comm, N=6. The channel effect that the sparsification "
        "arms are allowed to give up some of.",
        spread(6), comm("dense")),

    # ---- P2: does the selection rule carry information? ------------------
    arm("red_po6_gather_k2", 2,
        "Genuinely sparse top-k (gather path), k=2 of 5 peers. Paired with "
        "red_po6_random_k2 at identical k and identical executed FLOPs.",
        spread(6), comm("topk_gather", topk=2)),
    arm("red_po6_random_k2", 2,
        "Random-k control at matched k=2. Any gap to red_po6_gather_k2 is "
        "attributable to WHICH peers attention picked, not how many.",
        spread(6), comm("random_k", topk=2)),

    # ---- P3: the fixed-k grid (D-12's missing ablation cells) ------------
    arm("red_po6_gather_k1", 3, "Fixed-k grid, k=1 of 5.",
        spread(6), comm("topk_gather", topk=1)),
    arm("red_po6_gather_k4", 3, "Fixed-k grid, k=4 of 5 (near-dense).",
        spread(6), comm("topk_gather", topk=4)),
    arm("red_po12_gather_k3", 3,
        "Fixed-k at N=12, k=3 (25% of peers) -- the k/N ratio the manuscript "
        "reports on, at the largest N the training budget allows.",
        spread(12), comm("topk_gather", topk=3)),

    # ---- P4: mechanism arms ---------------------------------------------
    arm("red_po6_lagrangian", 4,
        "Adaptive gate under a Lagrangian budget targeting kappa=25% of peers. "
        "Dual ascent on the multiplier, so the budget is met rather than "
        "traded against return at an arbitrary fixed weight.",
        spread(6), comm("adaptive_topk"),
        {"lagrangian_k": True, "kappa_frac": 0.25, "lambda_lr": 0.01}),
    arm("red_po6_entmax", 4,
        "alpha-entmax attention, alpha=1.5: sparsity from the normalisation "
        "instead of a hard cutoff. Tests the manuscript's own diagnosis that "
        "the learned-k gate is the wrong mechanism.",
        spread(6), comm("entmax", entmax_alpha=1.5)),
    arm("red_po12_dense", 4,
        "Dense reference at N=12, so red_po12_gather_k3 has a same-N baseline.",
        spread(12), comm("dense")),
]


def to_yaml(d: dict, indent: int = 0) -> str:
    pad = "  " * indent
    out = []
    for key, val in d.items():
        if isinstance(val, dict):
            out.append(f"{pad}{key}:")
            out.append(to_yaml(val, indent + 1))
        elif isinstance(val, bool):
            out.append(f"{pad}{key}: {str(val).lower()}")
        elif isinstance(val, str):
            out.append(f"{pad}{key}: {val}")
        else:
            out.append(f"{pad}{key}: {val}")
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert len(ARMS) <= 12, f"{len(ARMS)} arms exceeds the 12-arm cap"
    for a in ARMS:
        n = a["env"]["n_agents"]
        assert n <= 12, f"{a['run_name']}: N={n} exceeds the N<=12 cap"

    manifest = []
    for a in ARMS:
        body = {k: v for k, v in a.items() if k not in ("tier", "rationale")}
        header = (f"# {a['run_name']}  (priority tier {a['tier']})\n"
                  f"# {a['rationale']}\n"
                  f"# Generated by scripts/make_reduced_design_configs.py --"
                  f" edit there, not here.\n\n")
        (OUT / f"{a['run_name']}.yaml").write_text(header + to_yaml(body) + "\n")
        manifest.append({
            "run_name": a["run_name"], "tier": a["tier"],
            "rationale": a["rationale"],
            "env": a["env"]["name"], "n_agents": a["env"]["n_agents"],
            "visibility_radius": a["env"].get("visibility_radius"),
            "attn_mode": a["model"].get("attn_mode", "none"),
            "topk": a["model"].get("topk"),
            "total_steps": a["trainer"]["total_steps"],
            "seeds": SEEDS,
        })
    (OUT / "MANIFEST.json").write_text(json.dumps(
        {"n_arms": len(ARMS), "seeds_per_arm": SEEDS,
         "steps_per_run": STEPS, "arms": manifest}, indent=2) + "\n")
    print(f"wrote {len(ARMS)} configs to {OUT}")
    for m in manifest:
        print(f"  tier {m['tier']}  {m['run_name']:<22} "
              f"{m['env']}/N={m['n_agents']:<3} {m['attn_mode']}"
              + (f" k={m['topk']}" if m["topk"] else ""))


if __name__ == "__main__":
    main()
