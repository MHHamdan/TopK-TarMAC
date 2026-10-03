# TMLR reviewer report (pre-revision manuscript, commit 53ef0f4)

Read as a skeptical TMLR reviewer applying the two acceptance criteria: are the
claims supported by accurate, convincing and clear evidence, and would some of
the TMLR audience learn something. Sources: the six specialist audits in this
directory plus a full read of the main text and appendices. Each issue carries
the disposition taken in this revision (see `final_adversarial_review.md` for
the resolution detail).

## Summary judgement

The paper's central message, that an operation count does not predict when
score-then-select sparse attention saves time, is supported by unusually careful
measurement: equivalence checks at every timed configuration, interleaved
timing, a searched dense baseline, three formats, replicated sweeps and
disclosed estimator reversals. The weaknesses were (i) a novelty framing that
did not acknowledge that the "at most 2x" limit on top-k attention had been
stated before (DFSS, PPoPP 2023), (ii) several mathematical statements that were
imprecise or wrong in their side conditions, (iii) a handful of numbers that did
not trace to the artifact they were attributed to, (iv) a broken kernel wrapper
at HEAD that would have made the release fail to regenerate one table, and
(v) a main body two lines over the 12-page review threshold with no LLM-use
disclosure. None of these undermines the central result; all were fixable.

## Issues

| # | Severity | Issue | Evidence | Disposition |
|---|---|---|---|---|
| 1 | CRITICAL | `fused_topk_attention_sort` raised `NameError` at HEAD (undefined `block_dc`, `precision`, `num_warps`, `num_stages`, introduced by a search-and-replace in 66696b7); the verifier counted an exception as a skipped check, so a re-run would have reported zero failures with the sort kernel unchecked | benchmark_redteam.md | FIX: wrapper restored to the pre-66696b7 code that produced the artifacts; verifier now counts exceptions as failures; equivalence re-run |
| 2 | CRITICAL | Abstract and conclusion: fused kernels "1.3 to 28 times slower" from k=4. 27.9 is the composed gathered path (N=192, k=128), not a fused kernel | claim_evidence_matrix.csv CL07; recomputed from attention_core_d16.json | FIX: "1.3 to 2.3 times slower at four retained peers, even in fp32" (range now spans both sweeps that measure it) |
| 3 | MAJOR | Novelty: DFSS Prop. 4.3 bounds the speedup of top-k attention by a constant even with a free oracle mask; Block-Sparse FlashAttention computes all scores and skips ~half of a pruned block's work | recent_related_work_audit.md; verified against the DFSS PDF and the BSFA abstract | FIX: credited in Sec. 2, Sec. 3.2 and the intro; Prop. 1 now framed as a closed-form MAC statement extending to training and serving as the limiting case of the selector accounting |
| 4 | MAJOR | Training bound 6N/(N+5k) was said to be "visible in the measurements"; the measured autograd backward forms dense dQ and dK, so its executed ratio is 2N/(N+k) | mathematical_redteam.md item 7 | FIX: bound stated as a property of a custom backward; App. A.2 explains what the measured backward executes |
| 5 | MAJOR | A6 sentence: "every other assumption leaves the ratio unchanged or moves it closer to one" is false for d_v > d_k (ceiling 1 + d_v/d_k) | math item 3 | FIX |
| 6 | MAJOR | "At most half of the arithmetic": softmax and exponentials are not counted; under the paper's own stage model the ratio can exceed 2 (2.09 at d=16) | math item 4 | FIX: "multiply-accumulates"; new assumption A7 with the general ceiling (2d+alpha)/(d+sigma) |
| 7 | MAJOR | Selector classes: "only full-score selection is bounded by two" is false (case (b) is <= 2 when gamma*p >= d); grid selector is formally case (c) as implemented; Cor. 1(c) proof remark wrong | math items 12, 15, 20 | FIX |
| 8 | MAJOR | Byte criterion presented as exact; whole-path exact form is k(d+4) <= N (k <= 25 at d=16, not 32) | math item 21 | FIX: exact and leading-order forms; called a screen, not a GPU model |
| 9 | MAJOR | "Both paths are below every ridge point" is false for the fused dense baseline (intensity 138 at N=512) | math item 24 | FIX: scoped to the two composed paths at d=16 |
| 10 | MAJOR | Pre-score selectors attend to different peers than top-k; selection cost and selection quality were not separated anywhere | prompt Sec. 19 | ADD EVIDENCE: fixed-retained-set sweep (`scripts/bench_fixed_set.py`, `results/fixed_set.json`, Table 2) |
| 11 | MAJOR | Host launch latency inside single-call event timing: the 16x attention stage is a ~10 us kernel | benchmark_redteam.md | ADD EVIDENCE + LIMITATION: CUDA-graph replay in the fixed-set sweep gives 0.07 (14x) device time; eager figures on a slow host are 1.7-3.4x larger; stated in Sec. 4.6 and Sec. 5 |
| 12 | MAJOR | "Paired median" is a ratio of separately computed medians | benchmark_redteam.md | FIX: renamed "ratio of medians" throughout; the new sweep stores raw samples and reports a genuinely paired median with bootstrap CI |
| 13 | MAJOR | Estimator sensitivity undisclosed in two places: one bf16 cell at parity (0.99) under the median; attention stage 0.18 under the median in the second pre-score sweep | benchmark_redteam.md; recomputed | FIX: both disclosed in Sec. 4 |
| 14 | MAJOR | Main body 12 pages + 2 lines (over the 12-page threshold) | latex_format_audit.md | FIX: 12 pages exactly; first-sweep pre-score figure and tiling-search table moved to App. B (both still referenced) |
| 15 | MAJOR | No LLM-use disclosure, which TMLR requires as a first-page footnote | tmlr_official_requirements.md | FIX: footnote added; **authors must confirm its wording** |
| 16 | MAJOR | Literature claim "two to roughly thirty agents, DGN low hundreds" contradicted by ATOC (N=100) and mean-field MARL (N up to 1000) | citation_audit.csv | FIX |
| 17 | MAJOR | Reproducibility statement: "a single script" (two) and "every number quoted in the text" (first-occurrence regex, spelled-out numbers unchecked) | reproducibility_notes.md | FIX: statement corrected; main-text manifest `scripts/paper_numbers.py` (45 numbers, file-scoped) added behind `--verify-paper-numbers` |
| 18 | MAJOR | Stale tooling: Makefile, `check_bib.py`, README targeted the deleted `paper/` tree and an earlier paper; checker rewrote stale tables in place so a stale table passed on the second run | reproducibility_notes.md | FIX |
| 19 | MINOR | "Environment takes 78 to 96%" is environment + per-step action selection | reproducibility audit | FIX |
| 20 | MINOR | Tiling-search table timed sequentially, contradicting "every claim is a ratio from one interleaved sweep"; its two sessions had 58 vs 71 TFLOP/s peaks | benchmark_redteam.md; autotune*.json | FIX: caption and limitation state it |
| 21 | MINOR | Z-order median quoted as 2.06 in a k=4 context (2.06 is k=2; k=4 is 2.10); "between 7 and 108" should be 107 | benchmark/math audits | FIX |
| 22 | MINOR | Intro "composed ... up to 37x": the masked composed path reaches 40.5x | benchmark_redteam.md | FIX: "composed gathered" / "composed from library operators" with the gathered path as the stated source |
| 23 | MINOR | FlashAttention family "without changing the operation count" (FA1 increases FLOPs by recomputation); Hoefler and Belli cited for "shared hardware" practice it does not specifically address; CPO cited for a Lagrangian it contrasts itself with | citation_audit.csv | FIX |
| 24 | MINOR | Metadata: NSA author name and DOI; Hoefler and Tillet pages/DOI | citation_audit.csv | FIX |
| 25 | MINOR | 25 uncited bib entries including a self-citation (anonymity risk in the supplement) | citation_audit.csv | FIX: removed; `check_bib.py` now gates the submission bib |
| 26 | MINOR | Tuned-dense harness crashed (KeyError) at four d>=64 configurations where the search winner was the library kernel; batch sweep used the default tiling off batch 256 | benchmark_redteam.md | FIX (crash) + disclosed in the grid caption; baseline unaffected because the library winner is timed |
| 27 | MINOR | `\resizebox` scaled the precision table up to ~12.5 pt | latex_format_audit.md | FIX: natural size |
| 28 | STYLE | "Learned inter-agent communication has converged on attention" while citing CommNet's mean pooling; "much of this literature ... usually motivated by cost" | claim matrix CL02 | FIX: narrowed |
| 29 | STYLE | Resubmission cue "in this revision" in an appendix | latex_format_audit.md | FIX |

## Not fixed (stated as limitations)

- One GPU architecture; the fixed-set sweep is a second card of the same model.
- No streaming exact top-k kernel; sparse tiling search admits 3 vs 28-106 configurations above N=192.
- Pre-score selectors composed from library operators (selector costs are upper bounds); one head width, one batch, fp32, uniformly placed agents.
- Equivalence references are fp32 composed paths (float64 only in the precision study); top-k set agreement is recorded only in the new fixed-set sweep and the precision study.
- No hardware counters; byte counts are analytic.
- Learning study: five seeds; not powered to rank sparsification variants.
