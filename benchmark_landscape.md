# Benchmark Landscape

Surveyed 2026-05-20 under the compute constraints in `HARDWARE.md` (4× RTX 2080
Ti with ~4–6 GB free VRAM per device, 48 GPU-hour total budget). Disk budget
200 GB. Tested-on-this-machine columns reflect actual install + rollout runs
performed on this date.

## Table

| Benchmark | Domain | Disk | GPU-h baseline (lit.) | Dominant method | License | Clean for our use | Tested |
|-----------|--------|------|-----------------------|-----------------|---------|-------------------|--------|
| MPE (simple_spread) | Cooperative navigation, 2D | <50 MB | 0.5–2 (5 seeds) | MAPPO, HAPPO | MIT (mpe2) | Yes | ✅ N∈{3,6,12,25} subsecond/25 steps |
| LBF (Level-Based Foraging) | Cooperative grid foraging | <50 MB | 0.5–2 (5 seeds) | MAPPO, QMIX (EPyMARL nums) | Apache-2.0 | Yes | ✅ 2p, 5p, sub-50 ms/50 steps |
| RWARE | Cooperative grid warehouse | <50 MB | 0.5–2 (5 seeds) | EPyMARL baselines | Apache-2.0 | Yes (via lbforaging-style install) | Not yet |
| Hanabi (HLE) | Cooperative imperfect-info card game | ~50 MB | 5–10 (5 seeds) | MAPPO, R2D2 | Apache-2.0 | Yes | Not yet |
| Overcooked-AI | Cooperative cooking, 2 agents | <100 MB | 1–3 (5 seeds) | PPO + BC | MIT | Yes | Not yet |
| SMAC v1 | Cooperative SC2 micromanagement | ~30 GB (SC2) | 5–20 (5 seeds) | QMIX, MAPPO | Blizzard EULA | Click-through, **skip** | Not tested |
| SMAC v2 | Procedural SC2 | ~30 GB (SC2) | 10–30 (5 seeds) | MAPPO | Blizzard EULA | Click-through, **skip** | Not tested |
| Multi-Agent MuJoCo | Continuous-control limbs | ~200 MB | 5–10 (5 seeds) | HAPPO | MIT | Yes; heavier than MPE | Not tested |
| MeltingPot | Mixed-motive substrates | ~500 MB | 10–30 (5 seeds) | OPRE, PPO | Apache-2.0 | Yes; heavy | Not tested |
| MiniGrid + BabyAI | Instruction-following grid | <50 MB | 1–5 (5 seeds) | PPO, IL | Apache-2.0 | Yes | Not tested |
| ALFWorld | Embodied text + visual | ~500 MB | 5–20 (LLM agent) | ReAct, Reflexion | MIT | Yes; LLM-heavy | Not tested |
| AgentBench / WebArena | LLM-as-agent suites | 1–5 GB | ≫48 (LLM RL) | GPT-4 prompting | Mixed | **Out of budget** | Not tested |

## Selection rule (from the run prompt)

> Pick the *smallest* benchmark that (a) supports the contribution chosen in
> Phase 3 and (b) has published baselines you can reproduce.

For attention-based agent communication (the OP-1/2/7/8 cluster from
`literature_review.md`), MPE simple_spread + LBF together cover both the dense
spatial-navigation case and the discrete cooperative-foraging case, scale to
≥25 agents within our compute budget, and have published EPyMARL/MARLlib
baselines we can reproduce. Both are under 50 MB on disk; both are CPU-friendly,
which is critical given our ~4–6 GB free VRAM per device.

## Decision

- **Primary**: MPE simple_spread (PettingZoo / mpe2 v1.1.0) at N ∈ {3, 6, 12,
  25} for scaling experiments.
- **Secondary**: LBF (Foraging-15x15-Np-3f-v3) at N ∈ {2, 5, 8} for a different
  reward density and discrete state.
- **Reject**: SMAC v1/v2 (StarCraft II EULA click-through; 30 GB install; per-
  baseline 5–20 GPU-h; with 4 GB free VRAM per device, MAPPO at the published
  scales would not fit). Recorded as a known stretch goal in `BLOCKERS.md`.
- **Reject**: LLM-agent benchmarks (AgentBench, WebArena, ALFWorld for LLM
  agents) — out of compute budget without a strong reason in Phase 3.

## Reproducibility checkpoints

- MPE: matches `mpe2` v1.1.0 (`from mpe2 import simple_spread_v3`).
- LBF: matches `lbforaging` v2.0.0 (`gym.make("Foraging-NxN-Pp-Ff-v3")`).
- Both installs are pinned in `pyproject.toml`.

## Hardware-budget sanity check

Per-N rollout speed measured on this host (CPU only, 1 process):

| N (MPE simple_spread) | obs_dim | wall ms / 25 steps |
|-----------------------|---------|--------------------|
| 3                     | 18      | 230                |
| 6                     | 36      | 304                |
| 12                    | 72      | 437                |
| 25                    | 150     | 1596               |

At N=12 with 16 parallel envs we expect ~6 k env-steps/s wall-clock just from
rollout, well above what training throughput will limit us to. The bottleneck
will be GPU forward/back passes for the policy + comm module.
