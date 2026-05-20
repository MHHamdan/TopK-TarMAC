# Proposal 002 — Two-Locus Attention: Composing Attention-as-Mixer with Attention-as-Channel in Cooperative MARL

## Research question
Cooperative MARL has placed attention at two distinct loci: the *value mixer*
(Qatten-style, combining per-agent utilities into Q_tot) and the *communication
channel* (TarMAC-style, weighting incoming messages). Are these two loci
substitutes or complements? Does combining them in a single architecture
provide additive gains, or does one strictly dominate?

## Hypothesis
H1: On MPE simple_spread and LBF, MAPPO + dense soft-attention communication
+ Qatten-style attention mixer (henceforth **Two-Locus MAPPO**, 2L-MAPPO)
strictly improves return over MAPPO + attention-comm-only and MAPPO +
attention-mixer-only.
H2: The mixer-attention contribution scales with partial-observability;
the channel-attention contribution scales with N.

## Why it's novel
- Qatten [`yang2020qatten`] uses attention only as mixer; no comm channel.
- TarMAC [`das2019tarmac`] uses attention only as channel; on top of a
  monolithic critic.
- MAT [`wen2022mat`] uses attention in the policy encoder but not as a
  value mixer in the QMIX sense and not as a between-agent comm channel.
- No paper in our 64-entry review composes both loci in a controlled
  ablation that isolates each contribution.

The novelty is empirical: a clean 2×2 ablation that disentangles the two
attention loci and asks whether they are complementary.

## Why it fits Mohammed's profile
The proposal is a direct compositional study of attention in MARL — sitting
exactly on the MARL-IoTP design surface. Producing the first principled
answer about *where* attention pays off is the kind of contribution that
papers cite and engineers reach for.

## Why it fits the target team
Distributed MARL infra benefits from knowing which attention computation
*must* run on the inner loop (rollout) vs the outer loop (training). If
mixer-attention dominates, only the centralised critic needs to attend; if
channel-attention dominates, attention has to run per-step in the rollout
process and is a real cost driver.

## Experimental plan
- **Env:** MPE simple_spread (primary), LBF Foraging (secondary).
- **N:** 6 (default) and 12 (scaling).
- **Architecture matrix (the headline 2×2):**
  | Mixer       | Channel       | Method name |
  |-------------|---------------|-------------|
  | scalar      | none          | MAPPO (baseline) |
  | scalar      | dense attn    | MAPPO+TarMAC |
  | attention   | none          | MAPPO+Qatten |
  | attention   | dense attn    | 2L-MAPPO (proposed) |
- **Seeds:** 5 per cell.
- **Ablations:** zero-out each attention locus separately (reduce-to-
  baseline checks), and a "shared key/value" condition where the two attentions
  reuse the same key projection.
- **Observability sweep:** partial-obs radius ∈ {0.3, 0.5, 0.7, full}; tests
  whether mixer-attention scales with partial obs.
- **Success criterion:** H1 holds at p < 0.05 on paired bootstrap CI; the
  observability sweep shows the predicted differential effect.

## Compute estimate
4 cells × 5 seeds × 2 N's × MPE 1M steps ≈ 40 runs × 25 min ≈ 17 GPU-h.
Observability sweep adds 4 × 5 = 20 runs × 25 min ≈ 8 GPU-h. LBF mirror at
N=5 adds 4 × 5 × 25 min ≈ 8 GPU-h. Total ≈ 33 GPU-h. Within budget.

## Risk register
1. **Composability is null** (H1 false). Acceptable — this is a publishable
   negative result; we then write up *why* with attention-pattern analysis.
2. **MPE rewards are too local** for mixer-attention to matter. Mitigation:
   LBF + the partial-obs sweep.
3. **Two attention modules over-parameterise the small policy** and overfit.
   Mitigation: parameter-matched comparison (match total params with a wider
   single-locus baseline).

## Expected venue
NeurIPS 2026 MARL workshop; ICML 2027 main track if scaling at N=12 holds.
