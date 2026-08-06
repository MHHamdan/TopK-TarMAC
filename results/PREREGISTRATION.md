# Pre-registered analysis plan — reduced-design tiers 2–4

**Written 2026-08-06, before any tier-2/3/4 result was inspected.** At the time
of writing, 8 of 40 runs had completed and no return, metric or contrast from
tiers 2–4 had been read by the analyst. The tier-1 gate (D-031) was inspected
before this document was written and is therefore **excluded from the
inferential family below** — it was a precondition check, is already reported,
and is not re-tested here.

Deviations from this plan must be recorded in §7 with a reason, not silently
applied.

---

## 1. Primary endpoint

Final greedy evaluation return, mean over the 32 evaluation episodes recorded
in `results/<arm>/seed_<s>/final_eval.json`, one number per seed, 5 seeds per
arm. Per-agent reward normalisation is on for every arm, so returns are
comparable within an N but **not across N** (they are not compared across N in
any contrast below).

No other quantity may be substituted as the primary endpoint. If
`final_eval.json` is missing for a run, the last `eval_return_mean` row of
`metrics.csv` is used and the substitution is logged.

## 2. Contrasts, fixed in advance

Six contrasts form the inferential family. Each is a two-sample comparison of
5 seeds against 5 seeds.

| ID | Tier | Question | A | B |
|----|------|----------|---|---|
| C1 | 2 | **Primary.** Does *which* peers attention selects carry information? | `red_po6_gather_k2` | `red_po6_random_k2` |
| C2 | 2 | What does sparsifying to k=2 cost against a working dense channel? | `red_po6_gather_k2` | `red_po6_dense` |
| C3 | 3 | Is return monotone in k? | `red_po6_gather_k4` | `red_po6_gather_k1` |
| C4 | 3 | Sparsification cost at the largest trained N | `red_po12_gather_k3` | `red_po12_dense` |
| C5 | 4 | Does a Lagrangian budget recover dense return? | `red_po6_lagrangian` | `red_po6_dense` |
| C6 | 4 | Does entmax recover dense return? | `red_po6_entmax` | `red_po6_dense` |

**C1 is the designated primary contrast.** It is the one the reduced design
exists to answer: `red_po6_gather_k2` and `red_po6_random_k2` are matched on k,
on executed FLOPs, and on every hyperparameter, and differ only in whether the
selected peers are chosen by attention score or uniformly at random.

## 3. Test, effect size, and interval

- **Test.** Exact two-sided permutation test on the difference of means. With
  5 vs 5 seeds there are C(10,5) = 252 distinct assignments, so the test is
  enumerated exactly, not sampled.
- **Effect size.** Hodges–Lehmann median of pairwise differences (primary,
  distribution-free) and Cohen's d (secondary, for comparability with the
  MARL literature).
- **Interval.** 95% percentile bootstrap over seeds, 10,000 resamples, fixed
  seed 0. Reported for every contrast and explicitly flagged as approximate at
  n=5.
- **Multiplicity.** Holm–Bonferroni across the six contrasts, family-wise
  error rate 0.05. Both uncorrected and Holm-adjusted p-values are reported.

## 4. Power, stated before seeing the data

With 5 vs 5 seeds the smallest attainable two-sided permutation p-value is
2/252 = **0.0079**. Holm's strictest threshold for a family of six is
0.05/6 = 0.0083. Therefore:

- **At most one contrast in this family can reach Holm-corrected significance,
  and only if its separation is perfect** (all five seeds of A above all five
  of B).
- Any non-significant result here is **under-powered, not evidence of
  absence**. The manuscript must say this in those words wherever a null is
  reported, and must not describe a null contrast as "no effect".
- This is a consequence of the 60 GPU-h budget (D-017/D-027), and it is a
  limitation of the design, not a finding.

## 5. Decision rules, fixed in advance

- **C1 (primary).** If A > B with Holm-adjusted p < 0.05, conclude the
  selection rule carries information. If not, report the point estimate and CI
  and conclude the design lacked power to detect a selection effect of this
  size — *not* that attention selects arbitrarily.
- **C2, C4, C5, C6.** These test whether a sparse or budgeted arm matches
  dense. A null here is the *expected* outcome under "sparsification is
  cheap in return", and because it is a null it cannot be claimed as
  equivalence. Any equivalence language requires a TOST against a
  pre-specified margin, which is **not** pre-registered and will therefore not
  be used.
- **C3.** Directional: k=4 is expected to be >= k=1 if k matters at all.
- No contrast may be dropped, added, or re-specified after unblinding.

## 6. Secondary, descriptive analyses (no correction, no inference)

Reported as trajectories and summary statistics only, never as hypothesis
tests: `mean_k`, `attn_entropy`, `lagrange_lambda` over the final decile of
training, and the measured visibility fraction of the partially observable
environment. These inform interpretation; they do not support claims.

## 7. Deviations log

_(append here; empty at time of writing)_

- 2026-08-06: none. The analysis was executed exactly as specified above:
  six contrasts, exact permutation test, Holm-Bonferroni, C1 primary. No
  contrast was added, dropped, re-specified or re-directed after unblinding.

- 2026-08-06, note (not a deviation): for C4 and C6 the 95% bootstrap CI
  excludes zero while the exact permutation test does not reach significance.
  The pre-registration named the permutation test as **the** test and the
  bootstrap interval as approximate at n=5, so the permutation result governs
  and both contrasts are reported as not separated. This tension is recorded
  rather than resolved in favour of whichever statistic reads better; it is a
  direct consequence of n=5 and is exactly what §4 anticipated.
