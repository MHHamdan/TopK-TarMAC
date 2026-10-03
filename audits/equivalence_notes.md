# Equivalence checking: how it is done, and the gaps

Companion to `audits/equivalence_matrix.csv`. All figures were recomputed from the released artifacts. No GPU was used.

## Three independent checking paths

### 1. `scripts/verify_all_configurations.py` → `results/equivalence.json`
This is the paper's evidence: "158 configurations, 136 with k < N-1, 4 tie rows in 1.25 M".
- **Configuration set**: every (N, d, k) whose latency appears in any of the five `attention_core_*.json` artifacts (`:32-53`). Recomputed: 158 configurations, 136 exercise selection, 1,251,200 query rows, 4 rows with an exact tie at the k-th score, 3 tie-divergent configurations, 0 failures. All of these match the paper.
- **Inputs**: iid N(0,1) q, k, v at **batch 32** (`:113`), seed `N*131 + d*17 + k` (`:134-140`). These are not the batch-256 tensors that were timed.
- **References**: **fp32, not float64.** The dense reference is `dense_unfused` and the top-k reference is `topk_gather` (`:142-143`). Both are composed PyTorch paths and serve as their own references, so their recorded deviation is 0 by construction.
- **Metric**: `max |got - ref|` over the whole output (`:176`). Diverging rows are counted as rows whose max over d exceeds `tol` = 1e-4 absolute (`:185-188`). **No relative error, no mean error, and no retained-set comparison.** A reference retained set is computed only as a support size (`:157`).
- **Tie handling** (`:189-201`): for `topk_fused_masked` and `topk_fused_sort` only, a deviation above tol is excused when `0 < diverging_rows <= rows_with_exact_tie`. The test compares **counts**, not row identity. It does not verify that the diverging rows are the tied rows.
- **Fail-open** (`:177-180`): an exception sets `deviation = None` and `continue`s. It is never counted in `failures`. At HEAD, `fused_topk_attention_sort` raises `NameError` (`src/kernels/fused_topk_attention.py:523, 537-539`; introduced in 66696b7). A rerun would therefore report 0 failures with the sort kernel unchecked. The released equivalence.json (2026-08-23) predates the break.
- **`dense_tuned` is missing from `DENSE`** (`:35`), so it is compared against the top-k reference at k = N-1. That is mathematically the same as dense, so the result is valid, but only by accident. At d = 64/128 it was checked only at N = 96, because the cells at N = 192/512 crashed in the sweep (`KeyError: 'BLOCK_M'`) and so do not appear in `configurations()`.
- **Pre-score** (`:56-107`): the fused pre-score kernel is compared with the composed `prescore_attention` on the same index set, for `radius_knn` and `morton_window`, at every (N, k) of `prescore.json` (40 configurations, max 1.07e-6, batch 32).

### 2. In-harness checks embedded in each `attention_core_*.json` (`bench_attention_core.py:359-413`)
These run at every k of every sweep, at batch `min(B, 32)` and seed 99, against the same fp32 references. `bench_attention_core.py:401-404` writes `out[name] = 0.0` for tie-excused rows, so the embedded `max_deviation_by_impl` hides tie divergences. They are listed separately in `tie_divergences` (4 in dsweep).

### 3. `scripts/bench_precision.py` → `results/precision.json` (the only float64 check)
- **References**: float64 dense and float64 `torch.topk` from the **format-rounded** inputs (`:239-252`).
- **Metric**: max abs error and the fraction of query rows whose row-max deviation exceeds `tol` = {fp32 1e-4, tf32 2e-2, bf16 5e-2} (`:162-165, 332`).
- **Admission**: a configuration is admitted when the row fraction is at most `ROW_TOL` = 1e-3 (`:159, 174`). TF32 top-k at k ≥ 4 (and k = 2 at several shapes) admitted 0 configurations. In fp32-IEEE, all three top-k paths differ from float64 in 0.0008% of rows (d=32, N=512, k=4), so **the fp32 "reference" of path 1 is not ground truth**. It agrees with the fused kernels because they share the fp32 near-tie behaviour.
- **Coverage**: N ∈ {192, 384, 512}, d ∈ {16, 32}, k ∈ {1, 2, 4, 8, 16}, at batch 256.

### Other
- **Z-order FlexAttention band**: `flex_deviation` = max |flex_morton − composed Morton window| (`bench_prescore.py:240-242`), max 1.19e-6 over 20 configurations. It is recorded, but no tolerance is applied.
- **Grid kNN**: recall against exact all-pairs kNN (`prescore_selection.py:180-185`), minimum 0.99927 (N=512, k=2). The paper's "at least 99.93%" matches. Recall counts retained peers that are also in the exact kNN set; it does not count duplicated or invalid candidates.

## Gaps

| Gap | Affects | Severity |
|---|---|---|
| fp32 references, not float64, for the 158-configuration check. The composed paths are their own references | dense reference, gathered top-k | MINOR (precision.json shows fp32 itself mis-ranks ~1e-5 of rows) |
| Retained-set agreement **NOT RECORDED** anywhere. Only output-deviation row counts serve as a proxy | all top-k rows | MINOR |
| Mean abs error and relative error **NOT RECORDED** | all | MINOR |
| Tie excusal is count-based, not row-identity-based | fused masked, fused sort | MINOR |
| Verifier fails open on exceptions. The sort launcher is broken at HEAD | fused sort | **CRITICAL for reproducibility** |
| Checked at batch 32 on iid Gaussian data, not on the timed tensors and not on tie-heavy data (duplicate observations, so duplicate keys, are common in MARL) | threshold kernel especially | MINOR |
| equivalence.json (08-23) predates precision.json and prescore_selectors.json (09-24). The **grid-kNN fused path is never checked**. The FlexAttention dense/top-k and TF32/bf16 paths are checked only in precision.json | grid pre-score, Flex | MINOR |
| `dense_tuned` at d = 64/128 checked only at N = 96 (crashed cells elsewhere) | dense tuned | MINOR |
| Composed pre-score attention has no independent reference | pre-score | MINOR |
| No equivalence record carries a git SHA | all | MINOR |
