# Mathematical red-team audit: `paper_submission_revision/`

Scope: every equation, closed form and number derived from an equation in
`sections/00,01,03,04,09,11,12*.tex` and `tables/prescore.tex`, re-derived from scratch
(sympy 1.14, `.venv/bin/python`), plus the scripts that compute them
(`src/kernels/prescore_selection.py::analytic_flops`, `scripts/stage_accounting.py`,
`scripts/roofline.py`, `scripts/check_revision_numbers.py`) and an executed-FLOP check of the
gathered path's backward pass with `torch.utils.flop_counter`.

Line numbers are the line numbers in each file.

## Verdict summary

| # | Item | Verdict |
|---|------|---------|
| 1 | Eq. (1) counts, Prop. 1 `2N/(N+k) < 2`, supremum, monotonicity | CORRECT |
| 2 | Diagonal convention `2(N-1)/(N-1+k)` strictly smaller | CORRECT |
| 3 | "each assumption other than full-score selection either leaves the ratio unchanged or moves it closer to one" (A6) | **WRONG** |
| 4 | "at most half of the arithmetic of an attention forward pass" (abstract, intro, conclusion): no assumption excludes softmax | **WRONG as worded** (true for MACs only) |
| 5 | `1 - k/(N-1)` "overstates ... by a factor approaching two" | CORRECT BUT IMPRECISE |
| 6 | A1, A3, A4, A5 | CORRECT |
| 7 | Training bound `6N/(N+5k)` and "a gathered top-k backward executes each over the Nk retained pairs" | bound CORRECT; **implementation claim WRONG** for the measured gathered path |
| 8 | FlashAttention recompute, "raises the bound to 7" | CORRECT BUT IMPRECISE |
| 9 | Prop. 2 (i), (ii) | CORRECT BUT IMPRECISE (limit regime not stated) |
| 10 | Score-then-select as the limiting case | CORRECT |
| 11 | Cor. 1 (a), (b) | CORRECT |
| 12 | Cor. 1 (c) `Θ(Nd/log N)`; proof: "with d = Θ(log N) the penalty disappears" | statement IMPRECISE; **proof remark WRONG** |
| 13 | Cor. 1 (d) `S → 2N/((2+c)k)` | CORRECT BUT IMPRECISE (it is an equality) |
| 14 | Cor. 1 unit convention vs the closed forms | **INCONSISTENT** (numbers right, convention sentence wrong) |
| 15 | "only full-score selection is bounded by two"; all-pairs selector "escapes the twofold ceiling" | **WRONG** (case (b) is bounded by 2 whenever γp ≥ d) |
| 16 | All-pairs NN closed form `Nd/(kd+N)`, 14.2, limit d, "rises towards d and no further" | CORRECT |
| 17 | Z-order closed form, 107, 84 % | CORRECT |
| 18 | "between 7 and 108 times in operations at four retained peers" | **WRONG** (max is 107.08) |
| 19 | `tables/prescore.tex` "ops saved" column (k = 2) | CORRECT (all 10 entries) |
| 20 | Grid selector "C_sel = Θ(Nk) ... case (d) with c = Θ(1/d)" | **WRONG asymptotically** as charged (the charged sort makes it Θ(N log N), case (c)); correct in the measured range |
| 21 | Byte criterion `kd ≤ N`, "admits k ≤ 32", "k ≤ 4" | CORRECT BUT IMPRECISE (necessary, not sufficient; leading-order; exact thresholds 25 and 3) |
| 22 | "four times past that bound", "about four times the bytes" | CORRECT |
| 23 | `(N+k)/2N` dotted line; 75 % at N=512, k=127 | CORRECT |
| 24 | Intensity: "both paths ... below every ridge (23, 53, 39; host 8, 28)" | CORRECT for the composed paths at d=16; IMPRECISE (false for the fused dense baseline and for composed paths at d ≥ 64) |
| 25 | Ridge 39 = 57.8 TFLOP/s / 1497 GB/s; 23, 53, 8, 28 | CORRECT |

---

## 1. Eq. (1) and Proposition 1

**STATEMENT** (03_analysis.tex:59-77; 11_app_theory.tex:5-11). `C_dense = 2N²d`,
`C_top-k = N²d + Nkd`, ratio `2N/(N+k) < 2` for all k ≥ 1, supremum 2 approached only as k/N → 0;
decreasing in k.

**INDEPENDENT DERIVATION.** Ratio = `2N/(N+k)`; `2 − ratio = 2k/(N+k) > 0` for k ≥ 1;
`∂/∂k = −2N/(N+k)² < 0`. Over integers k ≥ 1, N ≥ 2 the supremum is 2 and a sequence approaches
it iff k/N → 0 (ratio = 2/(1+k/N)). Removable fraction `1 − C_top-k/C_dense = (N−k)/(2N) < 1/2`.
Units: both terms are MACs of the two products; consistent. The counted model matches the
gathered implementation's forward (FlopCounterMode: fwd = 1.0156 × N²d·2 FLOPs at N=256, k=4,
i.e. N²d + Nkd MACs; `scripts`/scratch check below).

**VERDICT.** CORRECT. ("at most half" is true; the removable fraction is strictly below one half.)

**REQUIRED FIX.** None for the proposition itself (see items 3 and 4 for its prose).

## 2. Diagonal convention

**STATEMENT** (11_app_theory.tex:13-18). Exact diagonal gives `2(N−1)/((N−1)+k)`, strictly smaller
for every 1 ≤ k ≤ N−1.

**DERIVATION.** `2N/(N+k) − 2(N−1)/(N−1+k) = 2k/((N+k)(N+k−1)) > 0` for k ≥ 1. Also the
implementations do execute the N×N product with −∞ on the diagonal (`fused_topk_attention.py:394-397`,
`mappo.py:191-192`), so N²d is the executed count.

**VERDICT.** CORRECT (strict for every k ≥ 1, not only k ≤ N−1).

## 3. A6 and "moves it closer to one"

**STATEMENT** (03_analysis.tex:89-91): "each assumption other than full-score selection either leaves
the ratio unchanged or moves it closer to one." (11_app_theory.tex:20 "of which only the second is
essential"; 11:30-32 A6 ceiling `1 + d_v/d_k`.)

**DERIVATION.** With key width d_k, value width d_v:
ratio = `N(d_k+d_v)/(N d_k + k d_v)`, supremum (N → ∞, k fixed) `1 + d_v/d_k`, not attained.
For d_v > d_k it exceeds 2 (e.g. d_v = 2d_k, N=512, k=1: 2.988). Relaxing A6 therefore moves the
ceiling *away* from one whenever d_v > d_k. The A6 formula itself is right.
A1: both terms × R, unchanged. A3: H heads of width d/H: `H·2N²(d/H) / H(N²d/H + Nkd/H) = 2N/(N+k)`,
unchanged. A4: `r − (2N²d+P)/(N²d+Nkd+P) = P(N−k)/((N+k)(N²d+Nkd+P)) > 0`, strictly closer to 1.
A5: adds to the denominator only, decreases.

**VERDICT.** WRONG (the A6 exception). The formulas are correct; the summarising sentence is not.
"only the second is essential" (11:20) is also wrong for A6: with d_v > d_k the bound 2 fails.

**DOWNSTREAM.** 03_analysis.tex:89-91; 11_app_theory.tex:20; and every unqualified "at most half"
(00_abstract.tex:5; 01_intro.tex:26, 58; 09_conclusion.tex:4, 6) is conditional on d_v ≤ d_k.

**REQUIRED FIX.** 03:89-91 → "each of A1 and A3--A5 either leaves the ratio unchanged or moves it
closer to one; relaxing A6 changes the ceiling to $1 + d_v/d_k$, which exceeds two when values are
wider than keys." 11:20 → "Six assumptions support the statement; the second is essential, and the
sixth fixes the constant."

## 4. "At most half of the arithmetic" (softmax excluded but not listed as an assumption)

**STATEMENT** (00_abstract.tex:4-6 "remove at most half of the arithmetic of an attention forward
pass"; 01_intro.tex:26 "removable forward-pass arithmetic is at most half"; 09_conclusion.tex:4
"at most half of the forward-pass arithmetic"; 01_intro.tex:57-58 is correctly qualified "Under a
multiply–accumulate count").

**DERIVATION.** Eq. (1) counts only the two products. None of A1–A6 excludes softmax/exp/scale.
Charging α per softmaxed element and σ per score for selection:
sup ratio = `(2d + α)/(d + σ)`, which exceeds 2 iff α > 2σ. The paper's own per-stage FLOP model
(`scripts/stage_accounting.py:118-140`, `scripts/roofline.py`) charges softmax 5 FLOP/element and
selection 0: FLOP ratio `(4N²d + 5N²)/(2N²d + 2Nkd + 5Nk)` = 2.151 (k=1), 2.137 (k=4), 2.011 (k=32)
at N=512, d=16, and exceeds 2 for every k < 1280/37 ≈ 34.6. Even charging one comparison per score
(the information-theoretic minimum), sup = 69/33 = 2.09 at d=16.

**VERDICT.** WRONG as worded in abstract/intro:26/conclusion (true only for MACs of the two products);
Proposition 1 itself is correctly stated in MACs.

**DOWNSTREAM.** 00_abstract.tex:5; 01_intro.tex:26; 09_conclusion.tex:4, 6 ("bounded by two");
12_app_measurement.tex:107 and 04_measurement.tex:133 ("about half the arithmetic" — fine for MACs).

**REQUIRED FIX.** Add to 11_app_theory.tex after A6: "\textbf{(A7) Element-wise work excluded}: the
scale, softmax and exponentials are not counted. Charging $\alpha$ operations per softmaxed element
and $\sigma$ per ranked score gives a ceiling $(2d+\alpha)/(d+\sigma)$, which exceeds two when
$\alpha > 2\sigma$; at $d{=}16$ with five operations per softmax element and one comparison per score
it is $2.09$." Abstract:4-6 → "...so they can remove at most half of the multiply--accumulates of an
attention forward pass at any sparsity." 01_intro:26 → "the removable forward-pass multiply--accumulates
are at most half". 09:4 → "at most half of the forward-pass multiply--accumulates is removable".
(Abstract may not contain math; the wording above has none.)

## 5. `1 − k/(N−1)` "overstates ... by a factor approaching two"

**STATEMENT** (03_analysis.tex:80-84).

**DERIVATION.** Under one consistent (masked-diagonal) count the actual removable fraction is
`1 − (N−1+k)/(2(N−1)) = ½(1 − k/(N−1))`: the quoted figure is **exactly** twice it, for every N, k.
"Approaching two" only arises from mixing conventions: `(1 − k/(N−1))/((N−k)/(2N))`
`= 2 − 2k/((N−k)(N−1))`, which is below 2 and tends to 2.

**VERDICT.** CORRECT BUT IMPRECISE.

**REQUIRED FIX.** 03:82-83 → "it overstates the achievable reduction by exactly a factor of two
(under the masked-diagonal count the removable fraction is $\tfrac12(1 - k/(N-1))$)".

## 6. Intro "aggregation shrinks by a factor of about N/k"

01_intro.tex:6-7. Aggregation `N(N−1)d → Nkd`, factor `(N−1)/k ≈ N/k`. CORRECT.

## 7. Training bound and the gathered backward

**STATEMENT** (11_app_theory.tex:44-61; 03_analysis.tex:91-94): dense f+b `6N²d`; "a gathered top-k
backward executes each [of dV, dP, dQ, dK] over the Nk retained pairs alone"; ratio
`6N/(N+5k) < 6`; "at most five sixths ... removable"; "The asymmetry is visible in the measurements
of Appendix backward". 12_app_measurement.tex:287-291: "the backward is a scatter of gradients into them".

**DERIVATION.** Dense: fwd QKᵀ, PV (2N²d); bwd dV = PᵀdO, dP = dO Vᵀ, dQ = dS K, dK = dSᵀQ (4N²d);
total 6N²d. For exact top-k with renormalised softmax, dS_ij = 0 for j ∉ R_i, so the *minimum*
backward is 4Nkd; f+b = N²d + 5Nkd; ratio `6N/(N+5k)`, `6 − ratio = 30k/(N+5k) > 0`; removable
fraction `5(N−k)/(6N) < 5/6`. The bound is a correct supremum for score-then-select.

The *measured* gathered path (`reference_topk_attention`, `mappo.py:_forward_topk_gather`) does
**not** execute that. `topk`'s autograd backward scatters into a dense zero (B,N,N) tensor, and the
backward of `q @ kᵀ` then forms dQ and dK as dense N²d products. FlopCounterMode, N=256, d=16, k=4:

| path | fwd / (N²d MACs) | bwd / (N²d MACs) |
|---|---|---|
| dense | 2.000 | 4.000 |
| gathered | 1.0156 (= 1 + k/N) | 2.031 (= 2 + 2k/N) |

So the executed f+b count is `3N²d + 3Nkd` and the executed ratio is `2N/(N+k) < 2`, the same
ceiling as inference, not `6N/(N+5k)`. Reaching `6N/(N+5k)` requires a custom backward that gathers
the retained K rows and forms dQ, dK over the Nk pairs.

**VERDICT.** Bound CORRECT; the sentence that a gathered backward executes over retained pairs only,
and its link to the measured backward, WRONG for the implementation measured.

**DOWNSTREAM.** 03_analysis.tex:92-94; 11_app_theory.tex:47-50, 59-61; 12_app_measurement.tex:289-291
("the backward is a scatter of gradients into them" — it is a scatter followed by two dense N²d products).

**REQUIRED FIX.** 11:47-50 → "...a top-$k$ backward needs each only over the $Nk$ retained pairs,
given the indices recorded in the forward pass. Counting both passes for such a backward, without
recomputation," and after the equation add: "The composed gathered path we time does not reach this
bound: autograd scatters the retained-score gradient into a dense $N \times N$ tensor and forms the
query and key gradients as dense products, so its executed count is $3N^2d + 3Nkd$ and its ratio is
$2N/(N+k)$, as in the forward pass." 11:59-61 → "The measurements of Appendix~\ref{app:backward}
show the gathered path's backward is cheap relative to its forward because selection has no adjoint,
not because its backward arithmetic is sparse." 03:92-94 → "...so a backward written for it needs
the gradient products only over the retained pairs, and the analogous bound is $6N/(N+5k) < 6$; the
autograd backward we time executes dense query and key gradients and stays below $2$."
12:289-291 → "the backward is a scatter of gradients into the retained positions followed by two
dense products, cheap next to a selection that has almost no adjoint."

## 8. FlashAttention recompute, "raises the bound to 7"

**DERIVATION.** Dense with recompute: 2N²d + N²d (recompute S) + 4N²d = 7N²d. Against a top-k that
stores indices and does not recompute: `7N/(N+5k) < 7`. If the top-k path also recomputes its
retained scores (stores only indices): `7N/(N+6k)`, supremum still 7. If it must recompute all
scores to re-select: `7N/(2N+5k) < 3.5`. Against the measured gathered autograd path: `7N/(3(N+k)) < 7/3`.

**VERDICT.** CORRECT BUT IMPRECISE: 7 is the supremum only when the dense side recomputes and the
top-k side does not recompute the full score matrix; the ratio grows because the dense baseline does
more work, not because sparsity removes more.

**REQUIRED FIX.** 11:57-59 → "A dense implementation that recomputes the scores in its backward pass,
as FlashAttention does, adds $N^2 d$ to the dense count; against a top-$k$ backward that stores its
indices and recomputes at most the retained scores the supremum becomes $7$, a difference in what the
dense baseline recomputes rather than in what sparsity removes."

## 9. Proposition 2

**STATEMENT** (03_analysis.tex:106-118; 11:64-73).

**DERIVATION.** `S/(N/k) = 2Nkd/(2Nkd + C_sel) = 1/(1 + C_sel/(2Nkd))`. (i) → 1 iff
C_sel/(Nkd) → 0, along any sequence (N, k(N), d(N)); k may vary. Equality iff C_sel = 0. (ii)
denominator/(N²d) = 2k/N + C_sel/(N²d) → 0 ⇒ S → ∞; holds for k fixed or growing with k = o(N).

**VERDICT.** CORRECT BUT IMPRECISE: the asymptotic variable is never named in the proposition (only
in the corollary).

**REQUIRED FIX.** 03:115 → "(i) Along any sequence with $N \to \infty$ (with $k$ and $d$ allowed to
depend on $N$), if $C_{\mathrm{sel}} = o(Nkd)$ then $S/(N/k) \to 1$; ..."

## 10. Score-then-select as the limiting case

03:120-123; 11:75-81. `S(C_sel = N²d − Nkd) = 2N²d/(N²d + Nkd) = 2N/(N+k)`; since S is decreasing in
C_sel, the lower bound on C_sel gives the upper bound of Prop. 1. Direction correct. CORRECT.

## 11. Corollary 1 (a), (b)

(a) S ≤ 2N/(N+k) < 2. (b) `S = 2Nd/(2kd + γNp)`, `∂S/∂N = 4d²k/(γNp + 2dk)² > 0`, limit `2d/(γp)`
approached from below. CORRECT.

## 12. Corollary 1 (c) and its proof remark

**STATEMENT** (03:134-135) `S = Θ(Nd/log N)`; (11:88-90) "S/(N/k) = Θ(kd/log N) → 0; the statement
needs d = O(1), since with d = Θ(log N) the penalty disappears."

**DERIVATION.** With C_sel ~ c N log N: `S = 2Nd/(2kd + c log N)`. With k, d fixed,
S = Θ(N/log N); writing d inside Θ is vacuous at fixed d and is only meaningful if c is independent
of d and kd = O(log N). `S/(N/k) = 2kd/(2kd + c log N)`. With d = β log N:
`S/(N/k) → 2βk/(2βk + c)`, a constant **strictly below 1**: the penalty becomes a constant factor,
it does not disappear. It disappears only for kd = ω(log N).

**VERDICT.** Statement CORRECT BUT IMPRECISE; proof remark WRONG.

**REQUIRED FIX.** 03:135 → "$S = 2Nd/(2kd + \Theta(\log N)) = \Theta(N/\log N)$, unbounded but a
vanishing fraction of $N/k$." 11:89-90 → "the statement needs $kd = o(\log N)$; with
$d = \Theta(\log N)$ the penalty is a constant factor, and it disappears only when $kd$ grows faster
than $\log N$."

## 13. Corollary 1 (d)

`S = 2N²d/((2+c)Nkd) = 2N/((2+c)k)` exactly (sympy). Writing "S → 2N/((2+c)k)" (03:137) with an
N-dependent right-hand side is a misuse of →; the proof (11:90) correctly says "exactly".
CORRECT BUT IMPRECISE. Fix 03:137: "$S = 2N/((2+c)k)$ exactly".

## 14. Unit convention

**STATEMENT** (03:129-130): "Count C_sel in MAC-equivalents, one comparison, integer operation or
distance term being one unit." (11:114-121): NN charges "three operations for each two-dimensional
distance and one comparison per candidate, 4N² operations or 2N² MAC-equivalents"; Z-order
"N(15 + ⌊log₂N⌋ + 1) MAC-equivalents".

**DERIVATION.** The code (`prescore_selection.py:216-254`) counts the dense side in FLOPs (2 per MAC:
`dense_total = 4BN²d`) and charges selector *operations* one FLOP each, i.e. **one operation = half a
MAC-equivalent**: NN `4N²` ops = 2N² MAC-eq; Z-order `N(30 + 2·bit_length(N))` ops =
N(15 + ⌊log₂N⌋ + 1) MAC-eq; grid `N(36M + 2·bit_length)` ops. Under the convention as worded in
03:129-130 (comparison = 1 unit), the 3-op distance + 1 comparison would be 4N² units and the NN
ratio would be `Nd/(kd + 2N)` (limit d/2, 7.53 at N=512, k=4), and Z-order's 30 integer operations
would be 30 units. The published numbers follow the code (two operations per MAC-equivalent); the
sentence describing the convention does not. (NN's total coincidentally also matches "distance term
= 1 unit, comparison = 1 unit", but Z-order's does not.)

**VERDICT.** INCONSISTENT (IMPRECISE): closed forms and tables are right, the convention sentence is
wrong.

**REQUIRED FIX.** 03:129-130 → "Count $C_{\mathrm{sel}}$ in MAC-equivalents of two scalar operations
each, so that a comparison or integer operation is half a unit and a three-operation two-dimensional
distance is one and a half."

## 15. "Only full-score selection is bounded by two"; "escapes the twofold ceiling"

**STATEMENT** (03:141-146; 09_conclusion.tex:5-7).

**DERIVATION.** Case (b) has `S < 2d/(γp)` for every N; this is ≤ 2 whenever γp ≥ d (e.g. a learned
pairwise gate over d-dimensional features with γ ≥ 1 — exactly the G2ANet/MAGIC family cited at
11:105-109). Such a selector does not escape the twofold ceiling and is also bounded by two. The NN
selector (γ = 1, p = 2, d = 16) does escape (cap 16).

**VERDICT.** WRONG as a universal statement.

**REQUIRED FIX.** 03:142-146 → "...and full-score selection is bounded by two at every $N$. The
intermediate cases are where multi-agent methods sit, and they differ: a selector that ranks all pairs
in a feature space of per-pair cost $\gamma p$ is capped at $2d/(\gamma p)$ --- above two only when
$\gamma p < d$ --- while a sub-quadratic one is unbounded."

## 16. All-pairs NN closed form

**DERIVATION.** `S = 2N²d/(2Nkd + 2N²) = Nd/(N + kd)`; `∂S/∂N = d²k/(N+kd)² > 0`; limit d, never
reached. N=512, k=4, d=16: 14.222. Cor (b) with γ=1, p=2 gives 2d/(γp) = d, consistent. "Charged one
comparison per candidate, a lower bound for any selection algorithm" — selecting the k smallest of
N−1 needs ≥ N−2 comparisons, so this is a lower bound up to O(1); and 3 operations per 2-D squared
distance is itself an undercount (2 sub, 2 mul, 1 add = 5, or 4 with FMA). Both undercounts push S up,
so "the formula overstates that selector's ratio" (11:118-119) is the right direction.

**VERDICT.** CORRECT (11:114-119, 11:122-123 "14.2 against its ceiling of d = 16",
04_measurement.tex:207-208 "rises towards d and no further").

## 17. Z-order closed form

`S = 2N²d/(2Nkd + N(16 + L)) = 2Nd/(2kd + 16 + L) = 4Nd/(4kd + 32 + 2L)`, L = ⌊log₂N⌋ (sympy
identity: True). N=512, k=4, d=16: 32768/306 = 107.085; /128 = 0.8366 → 84 %.
`check_revision_numbers.py:654` uses `30 + 2·bit_length = 32 + 2L`, consistent. Case (c): S/(N/k) → 0.
CORRECT.

## 18. "between 7 and 108 times in operations at four retained peers"

**STATEMENT** (12_app_measurement.tex:425-427).

**DERIVATION.** k = 4, d = 16, N ∈ {48, 96, 192, 384, 512}:
NN = 6.86, 9.60, 12.00, 13.71, 14.22; Z = 10.31, 20.48, 40.69, 80.84, **107.08**
(identical in `results/prescore.json` and `prescore_repeat.json`). Range 6.86–107.08, i.e. 7 to 107.
11_app_theory.tex:123 correctly says 107.

**VERDICT.** WRONG (108 → 107).

**REQUIRED FIX.** 12:426 → "by between $7$ and $107$ times in operations at".

## 19. `tables/prescore.tex` "ops saved" (k = 2)

Formula vs table: NN 9.6/12.0/13.71/14.77/15.06 → 10/12/14/15/15 ✓; Z 18.07/35.72/70.62/139.64/184.09
→ 18/36/71/140/184 ✓. CORRECT.

## 20. Grid selector class

**STATEMENT** (11_app_theory.tex:124-129): charges "one sort to bucket agents" plus 9M candidates per
agent; "C_sel = Θ(Nk) and the selector is in case (d) with c = Θ(1/d)". 04_measurement.tex:209-211:
"its cost is Θ(Nk) at bounded density, case (d)".

**DERIVATION.** As charged (`analytic_flops`, 'grid_knn'): C_sel = N(18M + ⌊log₂N⌋ + 1) MAC-eq with
M = mean + 3√mean + 2, mean ≈ max(2,k) (cells are sized for ~k agents, `grid_knn_shape`). So
C_sel = Θ(Nk + N log N). With k, d fixed and N → ∞ the sort term dominates: the selector is in
case (c), S = Θ(N/log N), not case (d). In the measured range the sort term is 3–4.4 % of C_sel and
remains subordinate until ⌊log₂N⌋ ≈ 18M ≈ 216; there c = C_sel/(Nkd) = 4.06, 3.80, 3.53 at N = 192,
384, 512 (k=4, d=16), which is Θ(1/d) at fixed k (also Θ(1) in k, not constant in k for small k). The
implementation uses `argsort` (`prescore_selection.py:152`). Also, "bounded density" is not what makes
M = Θ(k): the grid side is set from N/k, so occupancy ≈ k for any spatially uniform placement; what is
needed is bounded *clustering* (Poisson-like occupancy). At N = 48 the grid falls back to the all-pairs
selector (9M = 126 ≥ 48), i.e. case (b).

**VERDICT.** WRONG asymptotically (internally inconsistent with the paper's own charged sort);
case (d) is a correct description only with a linear-time bucketing (counting sort) or for
N < 2^(18M).

**DOWNSTREAM.** 11_app_theory.tex:127-129; 04_measurement.tex:209-211 ("case (d)");
03_analysis.tex:144-146 does not cite grid, unaffected.

**REQUIRED FIX.** 11:127-129 → "for spatially uniform positions the cells hold $\Theta(k)$ agents, so
the candidate term is $\Theta(Nk)$ and the bucketing sort adds $\Theta(N \log N)$. With a linear-time
bucketing the selector is in case~(d) with $c = \Theta(1/d)$ (about $3.5$ at $d{=}16$, $k{=}4$); as
implemented with a comparison sort it is formally case~(c), though the sort is under $5\%$ of the
charged cost at every agent count we measure." 04:209-211 → "its cost is $\Theta(Nk)$ plus one sort
for spatially uniform positions --- case~(d) up to a sort that is under $5\%$ of its count here ---
and it retains ..."

## 21. Byte criterion `kd ≤ N`

**STATEMENT** (03_analysis.tex:158-167; 04:70-80; 12:201-205, 249-254; 01_intro:30; 09:10, 23):
"Counting each tensor read or written once, the compacted operand ... holds kd elements per query and
the score row it replaces holds N, so gathering is byte-neutral only when kd ≤ N. At d=16, N=512 the
criterion admits k ≤ 32; at d=128 ... k ≤ 4."

**DERIVATION.** Per batch element, fp32 elements, int64 index = 2 elements, V/Q/K/output counted once,
exactly as `scripts/stage_accounting.py:103-140` counts them. The score product is shared and cancels.

- Dense after scores: softmax (read N², write N²) + aggregate (read N², read V Nd, write O Nd)
  = `3N² + 2Nd`.
- Gathered after scores: top-k (read N², write values Nk, indices 2Nk) + softmax_k (2Nk) + gather
  (read V Nd, read indices 2Nk, write Nkd) + contraction (read Nk, read Nkd, write Nd)
  = `N² + 8Nk + 2Nkd + 2Nd`.
- Difference = `2Nkd + 8Nk − 2N²`: **byte-neutral iff k(d + 4) ≤ N.**
- Contraction stage alone vs the dense multiply it replaces: `Nk + Nkd + Nd ≤ N² + 2Nd` ⇔
  `k(d + 1) ≤ N + d`.

`kd ≤ N` is the leading-order form of both (drops the O(k) weight/index terms and the O(d) term);
it is a **necessary** condition for neutrality (k(d+4) ≤ N ⇒ kd ≤ N) but not sufficient. The paper's
"writes and re-reads a compacted tensor" (03:159-160; 04:71) is accounted for: the 2Nkd write+read is
offset by the dense path's softmax write+read 2N², which is why the criterion is kd ≤ N and not 2kd ≤ N
— but the text never says that the softmax traffic is what is credited; as written ("the score row it
replaces") a reader would derive 2kd ≤ N.

Thresholds at N = 512:

| d | leading `kd ≤ N` | whole path `k(d+4) ≤ N` | contraction `k(d+1) ≤ N+d` |
|---|---|---|---|
| 16 | k ≤ 32 | k ≤ 25 | k ≤ 31 |
| 128 | k ≤ 4 | **k ≤ 3** | k ≤ 4 |

Note: `scripts/roofline.py:66-80` uses a different byte model from `stage_accounting.py` for the
gather (indices 4 bytes, gathered V rows read as Nkd instead of V once), so the two scripts do not
implement the same "each tensor once" rule.

**VERDICT.** CORRECT BUT IMPRECISE. "Only when kd ≤ N" (necessary) is true; "admits k ≤ 32 / k ≤ 4"
reads as sufficiency and is off (25 / 3 under the whole-path model). 09:10 "adds more traffic than it
removes whenever kd > N" is TRUE (sufficient direction). 09:23 "check kd ≤ N before compacting" is a
necessary check only.

**DOWNSTREAM.** 03:162-167; 04:71-73, 79-80; 12:201-205; 01_intro:30-31; 09:23.

**REQUIRED FIX.** 03:162-167 → "\paragraph{Byte criterion.} Counting each tensor read or written once
(fp32 values, int64 indices), replacing the dense softmax and aggregation by selection, a softmax over
$k$, the gather and the compacted contraction changes traffic by $2Nkd + 8Nk - 2N^2$ elements per
batch element: the compacted tensor written and re-read costs $2kd$ per query where the dense weight
row written and re-read cost $2N$. Gathering is therefore byte-neutral only if $k(d+4) \le N$, to
leading order $kd \le N$. At $d{=}16$ and $N{=}512$ the leading-order criterion admits $k \le 32$
($k \le 25$ exactly); at $d{=}128$, $k \le 4$ ($k \le 3$ exactly)." 12:204 → "permits at most
$k \le 4$ (exactly $k \le 3$)". 09:23 → "to check $kd \le N$, a necessary condition, before compacting".
12:253-254: the criterion is defined in Section~\ref{sec:roofline}, not \ref{sec:composed}.

## 22. "Four times past that bound"; "about four times the bytes"

04:72-73: k = 127, d = 16: kd/N = 3.97 (k/32 = 3.97); whole-path k(d+4)/N = 4.96. ✓.
04:76-77, 12:249-251: contraction bytes `(Nk + Nkd + Nd)/(N² + 2Nd)` at k=127 = 3.998; the stage table
(1.14 GB / 0.29 GB, B=256) reproduces 3.998. The stage-table FLOPs (2.15 and 0.53 GFLOP) confirm
k = 127. CORRECT.

## 23. Dotted line and the 75 % credit

04:96-98: predicted gathered/dense = C_top-k/C_dense = `(N+k)/(2N)` < 1 for all k < N. ✓.
04:46-47: retained fraction 127/511 = 0.2485 → 75.1 % (127/512 → 75.2 %). ✓. CORRECT.

## 24. Arithmetic intensity

**STATEMENT** (03:153-157; 11_app_theory.tex:131-152): "both paths in this study have an arithmetic
intensity below every machine balance we measure"; GPU ridges 23 / 53 (and 39), host 8 / 28.

**DERIVATION.** Composed dense path (scores + softmax(5 FLOP/elt) + aggregate, each tensor once):
`AI = N(4d+5)/(16(N+d))`, supremum over N = d/4 + 5/16 = 4.31 at d=16; 4.18 at N=512 (matches
`results/roofline.json` 4.1818). Score product alone 7.53 (matches stage table). Gathered path lower
(0.66). All below 8 (host cache ridge), so the statement holds at d=16 for the composed paths.
But (a) at d=32 the composed dense path reaches 7.82 at N=512, within 3 % of the host ridge 8.06;
at d=128 it is 25.9 (> 23 GPU cache ridge, > 8 host) — "in this study" includes d up to 512;
(b) the best dense baseline the paper compares against is fused; under the same accounting (read Q, K, V
and write O once) its intensity is `N(4d+5)/(16d)` = 51.8, 103.5, 138 at N = 192, 384, 512 (d=16),
above every ridge (23, 39, 53). "Both paths" must therefore mean the two composed paths.

**VERDICT.** CORRECT BUT IMPRECISE.

**REQUIRED FIX.** 03:154-157 → "At head width $16$, the two composed paths in this study have an
arithmetic intensity below every machine balance we measure on the same device
(Appendix~\ref{app:roofline}), so their accounting is of bytes; fused kernels, which never write the
score matrix, are not covered by this argument." 11:145-147 → "Both composed paths sit below every one
of those figures at every agent count at $d{=}16$, on both devices..."

## 25. Ridge points

57.8e12/1497e9 = 38.6 → 39 ✓; `roofline.json`: 22.75, 52.49 (GPU), 8.06, 28.17 (host) ✓. CORRECT.

---

## Sympy script (run with `.venv/bin/python`)

```python
import sympy as sp
from math import floor, log2
N,k,d,dk,dv,P,R,c,g,p,a,s,L = sp.symbols('N k d d_k d_v P R c gamma p alpha sigma L', positive=True)
def show(label, expr): print(f"{label}: {expr}")

# Prop 1
Cd, Ct = 2*N**2*d, N**2*d + N*k*d
r1 = sp.simplify(Cd/Ct);                         show("P1 ratio", r1)
show("P1 2 - ratio", sp.factor(2 - r1))
show("P1 d/dk", sp.factor(sp.diff(r1, k)))
show("P1 lim N->oo", sp.limit(r1, N, sp.oo))
show("P1 removable fraction", sp.factor(1 - Ct/Cd))
rdiag = 2*(N-1)/((N-1)+k)
show("diag: ratio_N2 - ratio_diag", sp.factor(r1 - rdiag))
quoted = 1 - k/(N-1); actual_diag = 1 - ((N-1)+k)/(2*(N-1)); actual_N2 = (N-k)/(2*N)
show("quoted/actual (consistent diag)", sp.simplify(quoted/actual_diag))
show("2 - quoted/actual (mixed)", sp.factor(2 - quoted/actual_N2))
r6 = (N**2*dk + N**2*dv)/(N**2*dk + N*k*dv)
show("A6 ratio", sp.simplify(r6)); show("A6 sup", sp.limit(r6, N, sp.oo))
show("A6 d_v=2d_k,N=512,k=1", r6.subs({dk:1,dv:2,N:512,k:1}).evalf(5))
show("A4 r1 - r4", sp.factor(r1 - (Cd+P)/(Ct+P)))
H = sp.symbols('H', positive=True)
show("A3 heads", sp.simplify((H*2*N**2*(d/H))/(H*(N**2*(d/H)+N*k*(d/H)))))
rs = (2*N**2*d + a*N**2)/(N**2*d + N*k*d + a*N*k + s*N**2)
show("softmax-inclusive sup", sp.limit(rs, N, sp.oo))
rf = (4*N**2*d + 5*N**2)/(2*N**2*d + 2*N*k*d + 5*N*k)
for kk in (1,4,32,127): show(f"stage-model FLOP ratio k={kk}", rf.subs({N:512,d:16,k:kk}).evalf(4))
show("ratio>2 iff k <", sp.solve(sp.Eq(rf.subs({N:512,d:16}),2), k))

# Training
tr = 6*N**2*d/(N**2*d + 5*N*k*d); show("train", sp.simplify(tr)); show("6-train", sp.factor(6-tr))
show("FA", sp.simplify(7*N**2*d/(N**2*d + 5*N*k*d)))
show("FA vs top-k recomputing retained", sp.simplify(7*N**2*d/(N**2*d + 6*N*k*d)))
show("executed autograd gathered", sp.simplify(6*N**2*d/(3*N**2*d + 3*N*k*d)))

# Prop 2 / Cor 1
Csel = sp.symbols('C_sel', nonnegative=True)
S = 2*N**2*d/(2*N*k*d + Csel)
show("S/(N/k)", sp.simplify(S/(N/k)))
show("score-then-select", sp.simplify(S.subs(Csel, N**2*d - N*k*d)))
show("Cor b", sp.limit(S.subs(Csel, g*N**2*p), N, sp.oo))
show("Cor b dS/dN", sp.factor(sp.diff(S.subs(Csel, g*N**2*p), N)))
cc = sp.symbols('cc', positive=True); Sc = S.subs(Csel, cc*N*sp.log(N))
show("Cor c", sp.simplify(Sc)); show("Cor c S/(N/logN)", sp.limit(Sc/(N/sp.log(N)), N, sp.oo))
show("Cor c, d=beta log N", sp.limit((Sc/(N/k)).subs(d, sp.Symbol('beta',positive=True)*sp.log(N)), N, sp.oo))
show("Cor d", sp.simplify(S.subs(Csel, c*N*k*d)))

# Closed forms
Snn = sp.simplify(S.subs(Csel, 2*N**2)); show("NN", Snn); show("NN lim", sp.limit(Snn, N, sp.oo))
show("NN if 4N^2 units", sp.simplify(S.subs(Csel, 4*N**2)))
Sz = sp.simplify(S.subs(Csel, N*(16 + L)))
show("Z == 4Nd/(4kd+32+2L)", sp.simplify(Sz - 4*N*d/(4*k*d+32+2*L)) == 0)
nn = lambda n,kk,dd=16: n*dd/(kk*dd+n)
z  = lambda n,kk,dd=16: 4*n*dd/(4*kk*dd+32+2*floor(log2(n)))
show("NN(512,4)", nn(512,4)); show("Z(512,4)", z(512,4)); show("Z/(N/k)", z(512,4)/128)
for kk in (2,4):
    show(f"k={kk} NN", [round(nn(n,kk),2) for n in (48,96,192,384,512)])
    show(f"k={kk} Z",  [round(z(n,kk),2)  for n in (48,96,192,384,512)])
def grid(n,kk):   # mirrors prescore_selection.grid_knn_shape / analytic_flops
    occ=max(2,kk); G=max(1,int((n/occ)**0.5)); mean=n/(G*G); M=int(mean+3*mean**0.5+2)
    return M, n*(4*9*M+2*n.bit_length())/2
for n in (192,384,512):
    M,cs = grid(n,4); show(f"grid N={n}: M={M}, c", cs/(n*4*16))
    show("  sort share", n.bit_length()/(18*M+n.bit_length()))

# Bytes (fp32 elements, int64 index = 2)
dense_b  = 2*N**2 + (N**2 + 2*N*d)
sparse_b = (N**2 + 3*N*k) + 2*N*k + (N*d + 2*N*k + N*k*d) + (N*k + N*k*d + N*d)
show("sparse - dense", sp.expand(sparse_b - dense_b)); show("neutral k", sp.solve(sp.Eq(sparse_b, dense_b), k))
show("contraction neutral k", sp.solve(sp.Eq(N*k + N*k*d + N*d, N**2 + 2*N*d), k))
show("contraction byte ratio k=127", sp.Rational(127+127*16+16, 512+32).evalf(4))

# Intensity
ai_dense = (4*N**2*d + 5*N**2)/(4*(4*N**2 + 4*N*d))
show("composed dense AI", sp.simplify(ai_dense)); show("sup", sp.limit(ai_dense, N, sp.oo))
for dd in (16,32,128): show(f"AI N=512 d={dd}", ai_dense.subs({N:512,d:dd}).evalf(4))
ai_fused = (4*N**2*d + 5*N**2)/(16*N*d)
for n in (192,384,512): show(f"fused AI N={n}", ai_fused.subs({N:n,d:16}).evalf(4))
```

Output (abridged): `P1 2 - ratio: 2k/(N+k)`; `diag diff: 2k/((N+k)(N+k-1))`;
`quoted/actual (consistent): 2`; `A6 sup: (d_k+d_v)/d_k`; `A6 at d_v=2d_k: 2.988`;
`softmax-inclusive sup: (alpha+2d)/(d+sigma)`; stage-model FLOP ratio 2.151/2.137/2.011/1.676,
`>2 iff k < 1280/37`; `executed autograd gathered: 2N/(N+k)`; `Cor c, d=β log N: 2βk/(2βk+cc)`;
`Cor d: 2N/(k(c+2))`; `NN: Nd/(N+dk)`, lim d; `Z identity: True`; NN(512,4)=14.222, Z=107.085,
0.8366; k=4 Z max 107.08; grid c = 4.06/3.80/3.53, sort share 3.1–4.4 %; `sparse − dense =
2Nkd + 8Nk − 2N²`, neutral k = N/(d+4); contraction neutral k = (N+d)/(d+1); contraction ratio 3.998;
composed dense AI `N(4d+5)/(16(N+d))`, 4.18/7.82/25.85 at d=16/32/128; fused AI 51.75/103.5/138.

## Executed-FLOP check of the backward pass

```python
import torch, sys
sys.path.insert(0, "/home/mhamdan/TopK-TarMAC")
from torch.utils.flop_counter import FlopCounterMode
from src.kernels.fused_topk_attention import reference_topk_attention
B,N,d,k = 2,256,16,4
def dense(q,kk,v):
    s = (q@kk.transpose(-1,-2))/d**0.5
    s = s.masked_fill(torch.eye(N,dtype=torch.bool), float('-inf'))
    return torch.softmax(s,-1)@v
for name,f in (("dense",dense),("gathered",lambda q,kk,v: reference_topk_attention(q,kk,v,k))):
    q,kk,v = (torch.randn(B,N,d,requires_grad=True) for _ in range(3))
    with FlopCounterMode(display=False) as fc: o=f(q,kk,v)
    fwd=fc.get_total_flops()
    with FlopCounterMode(display=False) as fc2: o.sum().backward()
    unit=2*B*N*N*d
    print(name, fwd/unit, fc2.get_total_flops()/unit)
# dense 2.0 4.0 ; gathered 1.015625 2.03125   (k/N = 0.015625)
```
