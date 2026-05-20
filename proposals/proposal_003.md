# Proposal 003 — Budgeted Communication: A Constrained-MDP Formulation of Multi-Agent Channels

## Research question
Cooperative MARL communication papers minimise task return only; the
communication itself is free. In any real distributed system, channel
bandwidth has cost. Can we cast comm as a constrained MDP with a
bits-per-step budget B, and obtain a smooth Pareto trade-off between
task return and channel cost — *learned end-to-end*?

## Hypothesis
For a learned attention-comm module with a per-step KL-budget on the message
distribution (or equivalently a Lagrangian on bits-per-step), the
resulting policy frontier is monotone in B and recovers MAPPO at B=0,
dense-comm at B=∞. At intermediate B, the channel learns to be
informationally efficient — bits-per-step vs return is convex on
MPE simple_spread.

## Why it's novel
- TarMAC / IC3Net / ATOC / DGN — all in our review — minimise task return
  with no explicit channel cost.
- DPO [`rafailov2023dpo`] introduces a KL-budget in the LLM RLHF setting,
  not MARL.
- ATOC [`jiang2018atoc`] has a binary gate (on/off), not a continuous
  bits-per-step budget.

The novelty is the constrained-MDP formulation and the Pareto frontier
visualisation — channel-cost-aware MARL.

## Why it fits Mohammed's profile
Channel-aware attention is the direct extension of MARL-IoTP toward
real-world deployment constraints. A bandwidth-budget framing also opens
the door to applied MARL papers in the IoT/edge stream.

## Why it fits the target team
The team's distributed agentic-RL focus values explicit cost-of-rollout
accounting. A method that lets engineers dial channel cost vs return is
operationally useful and frames neatly against vLLM-omni's bandwidth /
serving-cost trade-offs.

## Experimental plan
- **Env:** MPE simple_spread (primary, N=6), LBF (secondary, 5p).
- **Method:** MAPPO + TarMAC-style soft attention channel, with a
  Lagrangian on the per-agent message KL divergence to a fixed prior
  (Gaussian or uniform-categorical). Lagrangian multiplier dual-ascent.
- **B sweep:** {0, 0.1, 0.3, 1.0, 3.0, ∞} (where ∞ disables the
  constraint).
- **Seeds:** 5 per (env, B).
- **Ablations:**
  1. Replace Lagrangian with a hard truncation (top-k); plot the same
     Pareto curve.
  2. Replace KL on the message dist with L2 on the message magnitude.
  3. Reduce-to-baseline at B=0 (verify recovery of MAPPO).
- **Success criterion:** monotone Pareto curve at p < 0.05 over seeds; the
  Lagrangian version dominates the hard-truncation version at the same
  bits-per-step.

## Compute estimate
6 B's × 5 seeds × 2 envs × 1M steps × 25 min ≈ 25 GPU-h. Ablations 3 ×
5 × 25 min ≈ 6 GPU-h. Total ≈ 31 GPU-h. Within budget.

## Risk register
1. **No interesting Pareto** (the optimum is always B=0 or B=∞). Acceptable:
   the paper then becomes about *when* communication is worth its cost,
   with a clean diagnostic in MPE.
2. **Lagrangian dual-ascent is unstable in practice.** Mitigation: PID-style
   multiplier update (Stooke et al. 2020 reproducible config); fall back
   to a sweep of fixed λ's.
3. **KL on message distribution is degenerate** (e.g., gate collapses).
   Mitigation: use Gumbel-softmax with temperature anneal; add the L2
   variant as a robustness check.

## Expected venue
ICML 2027 main track; NeurIPS 2026 MARL workshop as a stepping-stone.
