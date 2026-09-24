# TMLR submission package

Manuscript: `paper_submission_revision/main.pdf` (built from `db93c68` onward).
Main text + references: pp. 1–15. Appendix: pp. 16–34.

## 1. Manuscript audit

**Contribution.** A cost model for peer selection in attention-based multi-agent
communication, tested against measured implementations of one function:

- a forward-pass ceiling of 2 for score-then-select (6 for forward + backward);
- a selector-cost accounting that sorts mechanisms into four classes before any
  measurement (full-score < 2; all-pairs in a p-dim space → constant;
  sub-quadratic → unbounded; linear/free → Θ(N/k));
- a byte criterion, kd ≤ N, for when compaction is traffic-neutral;
- measurements answering three questions: is the reduction executed? does it
  become time? when is sparse attention faster?

**Strengths.**
- Every implementation is verified equivalent at every reported configuration.
- The dense baseline is searched over the same space and filter as the sparse
  kernels, with admitted-configuration counts reported.
- Every prose number is re-derived by `scripts/check_revision_numbers.py`
  (383 conditions).
- Estimator sensitivity is reported where it changes a conclusion.

**Weaknesses (remaining).**
- One GPU architecture, shared with other tenants, with unlocked clocks.
- No streaming exact top-k kernel.
- Pre-score selectors are composed from library operators.
- The learning study is under-powered.

**Reviewer attack points and status.**

| Attack | Status |
|---|---|
| "FLOPs ≠ latency is obvious" | Cited Dehghani et al. 2022 and Ivanov et al. 2021. Contribution framed as selector accounting + byte criterion + regime-specific measurements. |
| Ceiling ignores training | Fixed: forward-only scope; 6N/(N+5k) bound in §3.2 / App. A.2. |
| Crossover is an fp32 artefact | Tested (§4.5): with TF32/bf16 dense gains 2.7–8.1×; the advantage is gone from N = 384. |
| No production sparse baseline | FlexAttention added (dense and top-k block mask; Z-order band mask). |
| kNN selector is a strawman | Grid kNN, Θ(Nk), recall ≥ 0.9993, added; still 9.85× dense at N = 512, k = 4. |
| Only one GPU | Not fixed. Scoped in abstract and §5. |
| Your kernels, not the algorithm | Residency marked as ours. Streaming top-k named as untested. Search counts reported. |

## 2. Final title

**Score-Then-Select Sparse Attention in Multi-Agent Communication:
Operation-Count Limits and Measured Latency**

## 3. Final abstract (217 words)

See `paper_submission_revision/sections/00_abstract.tex`. It follows the order:
problem, the FLOPs argument, the cost model, the measurements, the practical
implication. It contains no notation.

## 4. Contribution paragraph

As in §1 of the paper (bullets):

1. **Cost model.** Under a MAC count, score-then-select removes at most half of
   the forward-pass attention arithmetic (Prop. 1). Charging selectors for their
   cost gives four classes (Prop. 2, Cor. 1). A byte criterion predicts when
   compaction is traffic-neutral.
2. **Measurement of what executes.** Masked, gathered and fused implementations
   of one function. A dense baseline tuned over the same space, with efficiency
   reported. Three numerical formats.
3. **Characterisation of when sparse attention is faster.**
   - Score-then-select: narrowly (k ≤ 2, d ≤ 32, no tensor cores).
   - Selection before scoring: more broadly, when the selector is cheap in
     absolute terms and the remaining attention is fused.

## 5. Section outline and page budget

| § | Section | Starts on page | Target length |
|---|---|---|---|
| 1 | Introduction | 1 | 1.5 |
| 2 | Related Work | 2 | 1 |
| 3 | A Cost Model for Peer Selection (Fig. 1, Props 1–2, Cor. 1, byte criterion) | 3 | 2.5–3 |
| 4 | Experiments: 4.1 Setup, 4.2 Q1, 4.3 Q2 (Tab. 1), 4.4 fused (Fig. 2, Tab. 2), 4.5 precision (Tab. 3), 4.6 pre-score (Tab. 4, Fig. 3) | 6 (4.5 on 9, 4.6 on 10) | 5 |
| 5 | Discussion and Limitations | 11 | 0.5–1 |
| 6 | Conclusion, impact and reproducibility statements | 12 | 0.5 |
| — | References | 13–15 | remaining |

## 6. Experiments added this revision

| Experiment | Artifact | Headline result |
|---|---|---|
| Precision: fp32 / TF32 / bf16, both kernels re-searched per format | `results/precision.json` | Fused k=1 ÷ best dense: fp32 0.67–0.91; TF32 0.83–1.29; bf16 0.95–1.38. Dense gains 2.7–4.0× (TF32) and 3.9–8.1× (bf16). |
| FlexAttention as dense baseline | same | Best dense in 9 of 12 reduced-precision shapes. |
| FlexAttention as top-k (block mask from retained set) | same | 7.7–218× dense: every 128-block active, and the mask is rebuilt per call. |
| FlexAttention over the Z-order band | `results/prescore_selectors.json` | 1.61× dense at N = 512, k = 4. Faster than composed (2.11), slower than fused (0.86). |
| Grid kNN selector, Θ(Nk) | same | Selector 0.51 → 1.67 ms (N 192 → 512) vs all-pairs 0.32 → 5.69. Path 9.85× dense. Recall ≥ 0.9993. |
| Second pre-score sweep (replication) | same | Z-order 0.86 (first sweep 0.85). All-pairs 37.5 (32.3). |
| Paired run-to-run bands on Fig. 2 | figure | Shows estimator and run sensitivity per point. |

Not done:
- Second GPU architecture (none available).
- Streaming exact top-k kernel.
- Fused pre-score selectors.
- Additional MARL experiments (deliberately out of scope).

## 7. Material moved to the appendix (nothing deleted)

| Appendix | Contents |
|---|---|
| A | Proofs of Props 1–2 and Cor. 1; the diagonal convention; assumptions A1–A6; the training bound; closed forms for three selectors; machine balance and intensity. |
| B | Full timing protocol; crossover and selection-strategy tables; head-width/batch sensitivity; per-stage accounting (table + figure); backward and training-iteration tables; robustness; first pre-score sweep (table + paragraphs). |
| C | Eight sources of measurement error (table). |
| D | Learning study (pre-registration, controls, primary comparison). |
| E–I | Unchanged: grid, dispersion, drift, validation, training detail, literature audit, reproduction. |

## 8. Claim audit (main text)

| Claim | Evidence | Risk | Final wording |
|---|---|---|---|
| At most half of attention arithmetic removable | Prop. 1 | High (training) | "…of the arithmetic of an attention forward pass"; 6N/(N+5k) for training |
| Composed sparse slower everywhere | Tab. 1, headline sweep | Low | "at every agent count and sparsity we measure" |
| Fused faster at k = 1 | Tab. 2, Tab. 3 | High (precision) | "only when one peer is retained … and only without tensor cores" |
| From k = 4 slower "in every format" | TF32 k ≥ 4 excluded | Medium | "in every configuration we could compare" |
| 1.3–28× slower from k = 4 | autotune (1.35), headline (27.9) | Low | kept; "in fp32" in conclusion |
| Crossover depends on dense efficiency, not d | Tab. 3 (same d, TF32) | Medium | "supports the first reading" (App. B); main text: "depends on how close dense attention runs to the machine" |
| Attention stage up to 16× faster | 0.060 / 0.06 | Low | "more than sixteen times" / "up to sixteen" |
| Pre-score path reaches parity at N = 512 | 0.85 / 0.86 on minima; median 2.06 vs 0.92 | High | "best-case result … holds on minima in both sweeps, under the median in one" |
| Grid kNN escapes quadratic growth | selector ms scaling | Medium | "removes the quadratic growth but not a large constant" |
| All-pairs selector "cannot exceed a constant" | Cor. 1(b), asymptotic | Low | kept (asymptotic statement) |
| FlexAttention top-k slow | Tab. 3 | Medium | mechanism stated (full scores + per-call mask + all blocks active) |
| MARL relevance | App. D | Medium | "not powered to compare sparsification variants; no main-text claim rests on it" |

No sentence uses "proves", "always" or "never". The "cannot" sentences are
mathematical (§3) or scoped (§5).

## 9. Reviewer response preparation

| Criticism | Response | Manuscript change |
|---|---|---|
| "Of course FLOPs are not latency." | Agreed and cited. The paper's claims are the selector-cost classes, which separate mechanisms before measurement, the byte criterion, and measurements in a regime (tens–hundreds of agents, d ≤ 32) that the long-context literature does not cover. | §2 last paragraph; §1 approach |
| The ceiling is trivial. | It is short by design. Its use is as the limiting case of Prop. 2, which is what organises the mechanisms. | §3.3 |
| The ceiling ignores backward. | Correct, and now stated. The training bound is 6N/(N+5k). The measured backward narrows the gap (18.4× → 6.2×) as it predicts. | §3.2, App. A.2, §4.3 |
| The crossover is an fp32 artefact. | Tested. With tensor cores, dense gains 2.7–8.1× and the advantage is gone from N = 384. The positive result is reported as conditional on a dense kernel without tensor cores. | §4.5, Tab. 3 |
| You compared against your own dense kernel. | The dense baseline is the best of vendor SDPA, our tuned kernel and (in §4.5) FlexAttention. FlexAttention wins 9 of 12 reduced-precision shapes and is used there. | §4.1, §4.5 |
| No production sparse attention. | FlexAttention top-k is 7.7–218× dense. FlexAttention on the Z-order band is 1.6× dense. Block-sparse execution needs block-structured sparsity. | §4.5, §4.6 |
| The kNN selector is naive. | Added an O(Nk) grid selector with recall ≥ 0.9993. It removes the quadratic growth, but at N ≤ 512 its path is still 9.85× dense. | §4.6, Tab. 4 |
| Only one GPU. | Acknowledged; no second architecture was available. All claims are paired ratios, and the abstract states the scope. | abstract, §5 |
| Your sparse kernels are under-tuned. | Search counts are reported (3 vs 28–106 above N = 192). Sparse efficiency figures are lower bounds, and residency is marked as a property of our kernels. | §4.4, §5 |
| The shared device makes timings unreliable. | Interleaved timing and paired ratios are used. The headline sweep and the pre-score sweep were each repeated. Fig. 2 bands show run-to-run ratios. Estimator reversals are reported. | §4.1, App. B |
| The MARL study adds nothing. | It establishes that the architecture is not inert. No main-text claim rests on it. | §1, App. D |

## Before uploading

- [ ] Produce the anonymised supplementary archive (code + `results/`), with git history removed.
- [ ] Scrub `configs/environment.lock` (`-e file:///home/mhamdan/agentic_marl`).
- [ ] The reproducibility statement says the code "will be released publicly
      with the camera-ready version". Keep it only if you intend to release it.
- [ ] Pre-registration timestamps: include them in the anonymised archive.
