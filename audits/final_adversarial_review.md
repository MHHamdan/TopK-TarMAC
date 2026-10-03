# Final adversarial review (post-revision)

Posture: a TMLR reviewer looking for grounds to reject. Every objection is
resolved as FIX, WEAKEN CLAIM, ADD EVIDENCE, MOVE TO LIMITATION, or REJECT
OBJECTION WITH EVIDENCE.

| Axis | Strongest objection | Resolution |
|---|---|---|
| A. Mathematical correctness | "Prop. 1 says 'at most half of the arithmetic' but ignores softmax; with element-wise work counted the ratio exceeds 2." | FIX. Restated in multiply–accumulates; A7 gives (2d+α)/(d+σ), 2.09 at d=16. |
| A. | "The training bound 6N/(N+5k) is not what your autograd backward executes." | FIX. Bound attributed to a custom backward; measured backward executes 3N²d+3Nkd (ratio 2N/(N+k)); App. A.2. |
| A. | "A6 can raise the ceiling, contrary to your text." | FIX. |
| B. Asymptotic reasoning | "Corollary 1 classes do not predict your finite-N ordering: grid (case d) loses to all-pairs (case b) at k=16." | WEAKEN + EXPLAIN. §4.6 now says the asymptotic class does not fix the ordering in the measured range; constants and traffic decide. Grid is formally case (c) as implemented (comparison sort), case (d) with linear bucketing. |
| B. | "'Only full-score selection is bounded by two' is false." | FIX. Case (b) ≤ 2 when γp ≥ d. |
| C. Benchmark fairness | "Your sparse kernel is under-tuned (3 configs vs 28–106)." | MOVE TO LIMITATION + REJECT IN PART. Stated; the positive k=1 result is conservative under it; the negative k ≥ 4 result is explained by the fixed-set sweep (≥ 90% of device time is finding the set, not aggregation), which no tiling of the aggregation can remove. |
| C. | "The tiling-search efficiencies are sequential, from sessions with different peaks." | FIX. Stated in caption and limitations; ratios elsewhere are re-timed interleaved. |
| D. Timing methodology | "Single-call CUDA events include host launch latency; your 16× is a 10 µs kernel." | ADD EVIDENCE. CUDA-graph replay in the fixed-set sweep: 0.07 of dense at N=512, k=2 (≈14×) as device time; eager on this session's slower host 1.7–3.4× larger. The k=1 score-then-select crossover is identical under eager and graph timing (0.77 vs 0.77 at N=512). Stated as a deployment caveat. |
| D. | "Your 'paired median' is not paired." | FIX. Renamed "ratio of medians"; the new sweep stores raw samples and reports genuinely paired medians with bootstrap CIs (±0.8% for the score-then-select kernel). |
| D. | "No L2 flush, fixed round-robin order." | MOVE TO LIMITATION (implicit): the minimum is an uncontended, warm-cache envelope; stated as such in §4.1 and App. B.1. Not changed, since all implementations share the regime. |
| E. Hardware dependence | "One GPU." | MOVE TO LIMITATION. Abstract scopes to one architecture; the fixed-set sweep on the second card of the host reproduces the fused ratios (0.77/1.27/1.93 vs 0.81/1.27/1.96). |
| F. Implementation equivalence | "The sort kernel cannot even run at HEAD; how was it verified?" | FIX. Wrapper restored to the code that produced the artifacts; verifier fails closed on exceptions; full equivalence sweep re-run (`results/equivalence.json`), adding the grid selector's fused path, which had never been checked. |
| F. | "References are fp32, not float64." | MOVE TO LIMITATION (App. C); float64 used in the precision study where rounding changes the retained set. |
| G. Baseline strength | "Did you use FlashAttention?" | REJECT WITH EVIDENCE. Backend probe (App. C): FlashAttention rejects masks and fp32 on this build; FlexAttention added as a dense baseline and wins 9/12 reduced-precision shapes. |
| H. Selector comparison | "Selectors attend to different peers; cost and quality are confounded." | ADD EVIDENCE. Fixed-set sweep holds the exact top-k set fixed and varies only how it is found. |
| I. k=1 interpretation | "The only positive score-then-select result is k=1, a degenerate softmax." | WEAKEN. Degeneracy now stated in abstract, intro and §4.4 adjacent to every statement of the crossover. |
| J. Statistical reporting | "No intervals." | ADD EVIDENCE where meaningful: bootstrap CIs in the new sweep; explanation why a bootstrap of a minimum on a non-stationary shared device is not reported for the older sweeps (App. B.2). |
| K. Related work | "The 2× ceiling for top-k is known (DFSS 2023)." | FIX. Credited in §1, §2, §3.2; contribution reframed. Also engaged: Block-Sparse FlashAttention, DeepSeek Sparse Attention (case b at scale), ZETA (Z-order top-k). |
| L. Citation accuracy | "Agent counts misreported; FlashAttention does not keep the operation count; CPO is not a Lagrangian method." | FIX. All three corrected; metadata fixed for NSA, Hoefler, Triton; 25 uncited entries removed. |
| M. Reproducibility | "README and Makefile describe another paper; numbers checker only checks first occurrences." | FIX. New README/Makefile; file-scoped manifest of 45 main-text numbers; anonymised supplement builder with a grep gate. |
| N. Writing clarity | "Dense paragraphs mixing definition, result and caveat." | FIX in part: §4 reorganised into Q1–Q5, a 'What a designer can take from this' paragraph separates lessons from limitations, conclusion rewritten around the methodological lesson. |
| O. Overclaiming | "Abstract attaches 16× to a sub-quadratic selector." | FIX. Attention stage vs end-to-end separated in abstract, §4.3, §4.6, §5. |
| P. Originality | "Is the measurement study new?" | REJECT WITH EVIDENCE. No audited paper gives the selector-cost classes, the kd ≤ N criterion, or measurements at tens–hundreds of agents with narrow heads (recent_related_work_audit.md). |
| Q. TMLR formatting | "Main body over 12 pages; table scaled to 12.5 pt." | FIX. 12 pages exactly; natural-size table; official style files byte-identical. |
| R. LLM disclosure | "Missing." | FIX (authors must confirm wording). |
| S. Broader impact | "Required?" | REJECT: optional for work without significant risk of harm (TMLR editorial policy); retained, concrete. |

## Status

Scientific status after this revision: the claims in the main text are each
traceable to an artifact or a derivation, the known defects are fixed, and the
residual weaknesses are stated where the claims they limit are made. Remaining
author actions: confirm the LLM-use footnote; decide whether to release the
code publicly at camera-ready (the statement no longer promises it).
