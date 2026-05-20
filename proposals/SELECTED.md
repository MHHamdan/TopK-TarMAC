# Selected proposal

## Ranking

Scores out of 5 in each axis (higher = better); ties broken by feasibility.

| Proposal | Novelty | Feasibility (compute) | Fit to Mohammed | Fit to target team | Total |
|----------|---------|-----------------------|-----------------|--------------------|-------|
| 001 — TopK-TarMAC (adaptive-k sparse comm)     | 4 | 4 | 5 | 5 | **18** |
| 002 — Two-Locus Attention (mixer + channel)    | 3 | 4 | 5 | 4 | **16** |
| 003 — Budgeted Communication (constrained MDP) | 4 | 3 | 4 | 4 | **15** |

- **Feasibility note:** all three fit the 48 GPU-h budget on paper, but
  Proposal 003 has the tightest margin (31 GPU-h estimate is fragile under
  dual-ascent instability and an extra ablation). Proposal 001 is the safest
  in compute and has the clearest scaling story.
- **Novelty note:** Proposals 001 and 003 are tied — both have no exact prior
  art in the 64-entry review. Proposal 002 is a controlled ablation rather
  than a new mechanism; novelty score reflects that, though the *answer* is
  not trivial.
- **Fit to Mohammed:** All three extend MARL-IoTP attention communication.
  Proposal 001 is the most direct line (sparsify the same attention block).
- **Fit to target team:** 001 wins on bandwidth-cost framing; 003 also wins
  here but with more theoretical apparatus that less directly maps to
  distributed-rollout infra concerns.

## Recommendation

**Proposal 001 — TopK-TarMAC.**

The one-paragraph defence: it asks a sharply scoped question that has a
clean operational answer, sits exactly on the MARL-IoTP design surface,
has a published reference baseline we can reproduce (MAPPO and TarMAC),
and produces both a *story* (sparsifying attention preserves cooperation)
and a *number* (FLOPs reduction at fixed return). The compute budget has
margin for the adaptive-k Pareto sweep, and the reduce-to-baseline ablation
is a one-flag switch. If we get strong scaling at N=25, we have an ICLR-class
result; if scaling is weak but the FLOPs trade-off is favourable, we still
have a NeurIPS-workshop paper. Proposal 002 will be partially absorbed:
the 2L architecture comparison runs as a free ablation in Phase 7 at N=12.

## Pause point

Per the run prompt's Phase 3 checkpoint, this file requests human review.
If no input arrives within 24 hours (run is unattended), the autonomous
agent will proceed with Proposal 001 and record that decision in
`DECISIONS.md`.
