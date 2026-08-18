# Reviewer response matrix

One row per reviewer point. "Remaining limitation" is what a reader should still
discount, stated because the alternative is for the reviewer to find it.

---

### Forward-pass-only benchmarking; no backward-pass timing

- **Action.** Report forward, backward and their sum for every differentiable
  implementation. The data existed in the harness and had never been reported.
- **New experiment.** The main sweep now records both; the backward column is
  new to the manuscript.
- **Location.** Section 4.6, Table 4.
- **Result.** The sparse path's backward is inexpensive relative to its own
  forward, because selection has almost no adjoint — the retained indices are
  recorded in the forward pass and the backward is a scatter of gradients into
  them. Dense attention's backward costs several times its forward. The sparse
  path's disadvantage on forward-plus-backward is therefore materially smaller
  than the forward-only measurement suggests, and is still a disadvantage.
- **Remaining limitation.** The fused kernels are hand-written forward passes
  and are excluded from this comparison; timing one against an autograd graph
  would compare two different things.

### No end-to-end training throughput evidence

- **Action.** Measure it rather than assert that it is environment-bound.
- **New experiment.** `scripts/bench_training_step.py` times one complete MAPPO
  iteration by phase, with the communication implementation swapped and
  everything else fixed.
- **Location.** Section 4.6, Table 5.
- **Result.** The environment and per-step action selection dominate; the
  implementations are separated by less than the variation between iterations of
  the same implementation. Reported as the null it is.
- **Remaining limitation.** This bounds the practical stakes only in the regime
  the training study occupies — few agents, a CPU-bound environment. It says
  nothing about a pipeline whose environment is cheap or whose agent count is
  large.

### No purpose-built fused sparse kernel

- **Action.** Write one, and let the result change the paper.
- **New experiment.** `src/kernels/fused_topk_attention.py`: two fused kernels
  that write no score matrix, index tensor or gathered value tensor to memory.
  One selects by successive argmax rounds on the matrix-multiply units and keeps
  the aggregation sparse; one selects by an exact bisection on the integer key
  of the score, so selection cost is independent of the number retained, at the
  cost of aggregating densely.
- **Location.** Section 4.3, Figure 2, Table 2.
- **Result.** **This overturns one of the manuscript's claims, and the
  correction is stated as such.** Fusion removes most of the measured penalty,
  and at the sparsest settings the sparse path becomes genuinely faster than the
  best dense implementation available to us. The advantage is smaller than the
  analytic ceiling permits and disappears well before the sparsity levels this
  literature uses. Two structural obstacles survive: ranking candidates costs
  more on the units that do the ranking than aggregating over the survivors
  saves, and the only selection strategy whose cost is independent of the number
  retained achieves that by abandoning the sparse aggregation.
- **Remaining limitation.** The kernels are Triton rather than tuned assembly;
  their selection strategies are argmax rounds and bisection rather than a radix
  or bitonic select; and they hold the whole score row in registers, which caps
  them at 1024 agents and limits occupancy at the largest sizes. A better kernel
  would narrow the remaining gap further without removing the requirement to
  form and rank the score matrix.

### The dense baseline was not of matched engineering effort

Not raised by the reviewer; found while addressing the previous point, and more
dangerous than it.

- **Action.** Write a dense attention kernel in the same framework, with the
  same tiling policy, in the same session. Report the better of it and the
  library path as the dense baseline.
- **Location.** Section 4.1; Section 7, as an eighth pitfall.
- **Result.** At the head widths multi-agent communication uses, the library
  fused path is short of what a straightforward hand-written kernel achieves.
  Against the library path alone, our fused sparse kernel appeared faster than
  Proposition 1 permits from sparsity — which is the signal that the baseline,
  not the algorithm, was producing the result. We would have published the
  unmatched comparison, and the checker now refuses a sparse speedup larger than
  the ceiling permits.
- **Remaining limitation.** "Matched effort" is a judgement, not a measurement.
  Both kernels were written by the same person in the same session, which is the
  closest control available to us.

### Incomplete exploration of very small k

- **Action.** Sweep absolute k rather than a fraction of peers.
- **New experiment.** k in {1, 2, 4, 8, 16, 32, 64, 128} across agent count,
  head width and batch.
- **Location.** Section 4.3, Figure 2, Table 2.
- **Result.** The crossover between sparse and dense lives entirely in this
  range, so the earlier fractional sweep — whose smallest k was 3 at 384 agents
  and 5 at 512 — could not have found it. The measurement is reported as a line
  plot of latency ratio against retained peers, with the analytic count model
  overlaid, rather than as another heatmap.
- **Remaining limitation.** k = 1 retains a single peer, which is a
  communication design few would deploy; the crossover is reported as a fact
  about the cost model rather than as a recommendation.

### Limited head-width range

- **Action.** Extend to head width 512.
- **Location.** Section 4.4, Appendix A.
- **Remaining limitation.** The sweep now stops at 512 rather than 256; we make
  no claim beyond the measured endpoint.

### One GPU generation and one software stack

- **Action.** None available. The host has two cards of the same model and the
  older hardware that produced this project's earlier runs is gone.
- **Location.** Section 8.
- **Remaining limitation.** Generality across GPU generations is untested. The
  arithmetic-intensity argument predicts the same qualitative behaviour wherever
  the machine balance is similar, but an argument that is device-independent by
  construction cannot establish device-independence.

### CPU allocator dependence

- **Action.** No new experiment. The effect is isolated by pinning the heap and
  counting minor faults inside the timed region, and both the pinned and
  un-pinned runs are released.
- **Location.** Section 4.7, Appendix B.
- **Remaining limitation.** Swapping to a different allocator would demonstrate
  that a different allocator has a different page-return policy, which is not a
  fact about attention. We declined it.

### No energy or power measurement

- **Action.** Declined, with the reason stated in the manuscript.
- **Remaining limitation.** Both GPUs carry other tenants' work throughout, so
  board-level power would attribute their computation to our kernels; per-process
  energy accounting is unavailable to this account. **No energy claim is made
  anywhere in the paper.** A software proxy would have satisfied the request
  while being wrong.

### Insufficient quantitative bridge to block-sparse and fused methods

- **Action.** Rewritten with primary sources verified against the published
  papers rather than against summaries.
- **Location.** Section 2.3.
- **Result.** The bridge names the three properties that separate that regime
  from ours: selection amortised over blocks rather than performed per pair;
  block structure preserving the dense matrix-multiply shape inside each
  selected tile, which per-pair gather does not; and skipped work that is
  enormous at 10^5 tokens against a factor of two at the agent counts here. It
  cites BigBird's own statement that accelerators need coalesced block access
  rather than sporadic lookups.
- **Remaining limitation.** No comparability is claimed across four orders of
  magnitude of sequence length, and none should be inferred.

### No empirical escape case for Proposition 2

- **Action.** Build one, and charge it for its selector.
- **New experiment.** `src/kernels/prescore_selection.py`: two selectors that
  fix the retained set before any attention score exists — k nearest peers by
  distance in the position space, which is the rule neighbourhood-based
  communication already uses, and a Z-order window, which is sub-quadratic.
  Selector cost is timed separately from attention cost.
- **Location.** Section 5, Figure 4, Table 6.
- **Remaining limitation.** This is a cost demonstration. The selectors attend
  to a different peer set than top-k does, so the comparison is of cost and not
  of quality, and whether a pre-score selector preserves task performance is not
  answered here.

### No explicit per-stage arithmetic-intensity and byte accounting

- **Action.** Report it per stage, with attainment against measured machine
  limits.
- **New experiment.** `scripts/stage_accounting.py`.
- **Location.** Section 4.5, Table 3, Figure 3.
- **Result.** **This corrected an explanation the manuscript had offered.** We
  had attributed the sparse contraction's slowness to a batched matrix--vector
  product having lower arithmetic intensity and being unable to reach the same
  hardware paths. Measured, the sparse contraction attains a *higher* fraction
  of achievable bandwidth than the dense multiply it replaces. Both are
  bandwidth-saturated; the sparse one is slower because it moves several times
  the bytes. The replacement is a checkable criterion: gathering is byte-neutral
  only when the compacted operand is no larger than the score row it avoids
  re-reading.
- **Remaining limitation.** The byte counts are analytic. They ignore cache
  residency, which at the smaller agent counts they materially overstate.

### The paper is dense and digresses into implementation detail

- **Action.** The learning study's inferential machinery, the pre-registration
  chronology, the allocator procedure, the backend probe, the audit protocol and
  the full parameter grid all move to five topic-based appendices. Eleven
  appendix subsections, five of which existed only to point at one float, become
  thirteen organised by subject.
- **Location.** Throughout; see `writing_revision_report.md`.

---

## Errors of fact corrected

| What | Where it was | Correction |
|---|---|---|
| "At twelve agents neither trained variant beats a uniform-random policy" | archival `07_limitations.tex` | False, and contradicted by the paper's own appendix. The random baseline is −758.9; the dense variant is −864.9 and is worse, but the sparse variant is −717.9 with an interval entirely above the random mean. The corrected statement supports the same conclusion. |
| Margin between the permutation floor and the first Holm threshold given as 5×10⁻⁴ | archival `05_training.tex` | It is 3.97×10⁻⁴. Wrong at the only significant figure quoted, in a sentence about the tightness of the margin. |
| Selection reported as 51% of the gathered forward pass | archival `04_measurement.tex` | 52.4% against the measured total; the denominator was not reconstructible from the released artifact, which the new stage accounting fixes. |
| CPU bandwidth probe read an untouched buffer | `scripts/roofline.py` | An untouched allocation is backed by the shared zero page, so the host probe re-read one cache line. Every CPU ridge point derived from it was wrong, in the direction that protected the argument. Fixed, and two ridge points are now reported per device. |
| Byte model counted the score matrix once | `scripts/roofline.py`, manuscript Section 3.3 | Each elementwise step on it is out of place, so it is written and read once per step. Corrected. |
| Released pre-registration asserts that at most one comparison can reach corrected significance | `results/PREREGISTRATION.md` | False: Holm is a step-down procedure. The manuscript was already correct; the deviation is now logged against the artifact rather than the artifact silently edited. |
| Hardware and provenance records describe a superseded software stack and an unavailable host | `HARDWARE.md`, `results/provenance.json` | Corrected, and the reproduction appendix now points at what exists. |

## Framing changes

| What | Why |
|---|---|
| The headline agent count moves down | The manuscript made 512 agents its headline while its own related work said this literature operates at fewer than a hundred. The attention-communication methods it cites run between two and roughly thirty agents. The larger counts are now a stress test. |
| The absence claim becomes a taxonomy | "No audited paper selects peers by attention score after computing all of them" is defensible as scoped and vulnerable as read; handshake-based collaborative perception does exactly that. Restating Proposition 1 as the special case of Proposition 2 in which the selector is a full score matrix places every mechanism on one axis and replaces an absence claim with a result. |
| Existing work is credited for supporting the thesis | The exact top-k attention of Gupta et al. reports that its mechanism is as fast as vanilla attention and buys memory rather than time — an independent instance of this paper's central finding, in the sequence regime, published in 2021. Routing Transformer states the pre-score/post-score distinction explicitly. Both were previously under-used. |
