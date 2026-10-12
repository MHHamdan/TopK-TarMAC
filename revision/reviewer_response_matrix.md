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
- **Result.** **This overturns one of the manuscript's claims.** Fusion removes
  most of the measured penalty, and at one retained peer the sparse path becomes
  faster than the best dense implementation available to us — including a search
  over roughly a hundred tilings of a dense kernel of our own. The advantage is
  confined to one retained peer, to the narrow head widths where no
  implementation reaches half the device's arithmetic rate, and it is gone by
  four retained peers. The structural obstacle survives: ranking
  candidates costs more on the units that do the ranking than aggregating over
  the survivors saves, in all three selection strategies we implemented.
- **Remaining limitation.** The kernels are Triton rather than tuned assembly,
  and they hold the whole score row in registers, which caps them at 1024 agents
  and limits occupancy at the largest sizes. A better kernel would narrow the
  remaining gap further without removing the requirement to form and rank the
  score matrix.

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
- **Remaining limitation.** Matched effort turned out to be the wrong condition;
  see the second-round entry below, which replaces it with matched efficiency
  and a tiling search.

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

---

# Second round: a fresh review of the revision

The revised manuscript was given to an independent skeptical reviewer who had
not seen it before, together with the artifacts and the kernel source. They
returned *major revisions required* and identified defects that the first round
had introduced or missed. Everything below was found by that review and acted on
before submission.

### The crossover might be an artifact of a weak dense baseline

**The objection.** The paper's only positive claim rested on a hand-written
sparse kernel beating a hand-written dense kernel by the same author. Matched
authorship does not identify the algorithm; matched *efficiency against the
machine* does. Neither kernel's achieved efficiency was reported anywhere, and
the dense kernel ran at a fraction of what the vendor library reaches at wider
head widths, so the claimed margin could lie entirely inside the baseline's
deficit.

**Action.** `scripts/autotune_kernels.py` searches roughly a hundred tilings for
each kernel — query tile, key tile or head chunk, warp count, pipeline depth —
with the same budget, takes the better of that search and the vendor library as
the dense baseline, and reports achieved arithmetic rate as a fraction of the
peak the device attains in the same process.

**Result.** The crossover survives tuning and is not produced by an unusually
good sparse kernel: across the head widths where it exists, the sparse kernel
reaches 21–33% of the device's rate against 30–47% for dense, so it wins only by
doing about half the arithmetic and any residual engineering bias runs against
it. **The objection is partly vindicated all the same.** Dense efficiency climbs
with head width to 77% at $d{=}128$, and the crossover disappears exactly where
dense passes about half the device's rate. We cannot exclude that a dense kernel
reaching at $d{=}16$ what the library reaches at $d{=}128$ would remove it
entirely. That is now stated in the section, the limitations and the abstract,
and the positive result is labelled provisional.

### Section 5 used the baseline the paper itself declares invalid

**The objection.** The pre-score experiment quoted every ratio against PyTorch's
library kernel, not against the matched dense kernel — the exact pitfall the
paper's own measurement-error section names. Against the correct baseline, the
claim that the pre-score attention stage is "roughly twice as fast as dense"
reversed.

**Action and result.** Re-run against the better of both dense kernels. The
claim as stated was wrong and was withdrawn: composed from library operators the
pre-score attention stage is slower than dense at every configuration but one,
running 0.97–5.11× it. A later objection (below) showed that this corrected
figure was itself measuring the composition rather than the design; the ratios
in the paper are now fused, and every one of them is still against the better of
both dense kernels.

### The pre-score negative result measured an unfused bit loop, not a selector

**The objection.** The ~0.19 ms selector cost, attributed in the text to a sort,
was roughly a hundred kernel launches from a Python per-bit loop computing
Morton codes — the same composition overhead the paper wrote fused kernels to
eliminate for the other half of its argument. A double standard.

**Action.** Replaced with the standard branch-free bit-spreading interleave,
verified bit-identical to the loop. Selector cost falls by about a third. The
conclusion is unchanged in direction and the text no longer attributes the cost
to the sort. The remaining constant is still a composed-operator upper bound and
is now labelled as one.

### The equivalence claim was false as implemented

**The objection.** The paper stated that every implementation is checked against
the reference at every configuration whose latency is reported. The check ran
once per (batch, head width, agent count) at the widest $k$ — which at several
agent counts is $k = N-1$, where selection retains every peer and is never
exercised. There was no correctness check at the small $k$ that produce the
positive result.

**Action.** The check now runs at every $(N, d, k)$ whose latency is reported,
and the campaign was re-run so the released artifact matches the claim.

### The operation-count ceiling was used as a validity check on wall-clock

**The objection.** A multiply–accumulate bound constrains time in neither
direction, and using it as a plausibility check contradicts the paper's own
thesis. It is also one-sided and blind to a sub-2× baseline deficiency.

**Action.** Removed from the abstract, the contributions and Section 4.3. It
survives only in the measurement-error section, as the smoke alarm that caught
one real error, and is described as that rather than as a criterion. The checker
now bans the phrase.

### Prose contradicted a table, and an appendix was a placeholder

The text quoted 91% attainment for dense aggregation where the table said 81%
(a stale number from a superseded run); the checker now re-derives both from the
artifact. The literature-audit appendix rendered as "[pending]"; it is now
generated from the released audit sheet.

### Training-step timing was contaminated by warm-up

**The objection.** At six agents the first two implementations showed six times
the action-selection cost of the last three — warm-up, not a property of the
implementations — and it was that artifact which produced the "the
communication-free variant is not the fastest" observation.

**Action.** Two discarded iterations instead of one. All five implementations
now agree within 2% at every agent count with consistent phase shares, and the
null is clean.

### Smaller items acted on

Table 5's raw code identifiers replaced with the names used elsewhere; the
"widest $k$" column's varying fraction-of-peers made explicit; the claim that
implementation was "held constant" across two different kernels removed; the
"sparse arithmetic or cheap selection but not both" claim softened to a
statement about the implementations we obtained; the disagreement between two
machine probes explained (peak varies between 53 and 77 TFLOP/s on this shared
host, so every fraction-of-peak is an upper bound); and the figure panel at 96
agents labelled as supporting no claim.

### "Sparse arithmetic or cheap selection but not both" was an assertion, not a measurement

**The objection.** The manuscript said it could not write a kernel with both
sparse aggregation and selection cost independent of the retained count, and
conceded in the same sentence that a radix or bitonic select would narrow the
gap and had not been implemented. A claim whose own statement names the
experiment that would test it should carry that experiment.

**Action.** Wrote a third fused kernel. Scores go through the same
matrix-multiply units as the reference, are mapped to order-preserving integers,
packed against their own peer index into one 64-bit word, and sorted once per
query row; the top k words are unpacked and exactly k value rows are read.
Selection is one bitonic sort, logarithmic in the agent count and independent of
k; aggregation is sparse. Verified exact at 123 shapes and at every
configuration of the equivalence sweep.

**Result. The claim is refuted and replaced by a measurement.** Both properties
can be had at once: selection does not grow with k and the aggregation reads
exactly k value rows, so at 192 agents the kernel runs in 0.164 ms at one
retained peer and 0.186 ms at sixteen — a 14% rise, against eightfold for the
argmax kernel. It is also five times dense attention across that range, and
slower than the argmax kernel below sixteen retained peers. Having both
properties does not help, which is a stronger statement than being unable to
obtain them. Section 4 and Table 3 now report this;
`results/attention_core_sort.json`.

### The pre-score negative result was measured on a composed attention stage

**The objection.** Section 5's attention stage was gather-keys, gather-values,
contract, softmax, contract — a composition writing and re-reading a
$(B,N,k,d)$ tensor twice. The paper's entire argument about the score-then-select
path is that such a constant is an upper bound rather than a property of the
design, so reporting the composed figure as a property of pre-score selection
applied one standard to the mechanism the paper criticises and another to the
mechanism it proposes.

**Action.** Wrote `fused_prescore_attention`: one program per tile of queries
walks that tile's retained peers, reading one key row and one value row per peer
and accumulating with a running maximum. Nothing intermediate is written.
Verified exact against the composed path at all 40 configurations and both
selectors.

**Result. This overturns the section's conclusion.** Fused, attention over a
pre-chosen peer set is faster than the best dense implementation at every agent
count when at most four peers are retained — 0.76, 0.43, 0.25, 0.12 and 0.09
times dense from 48 to 512 agents at four retained peers, best case 0.060,
nearly seventeen times faster than dense. Fusing is worth 3.9–17.4×. With the
sub-quadratic selector in front of it the whole path costs 16.8, 8.8, 4.0, 1.28
and 0.91 times dense: it crosses below dense attention at 512 agents, by about
ten per cent, at two and four retained peers, and the sweep's repeat puts those
cells within 1.5% of each other. The explanation the manuscript gave for the negative
result — that the gathered operand loses the matrix-multiply shape — is
withdrawn. The constructive claim is now that escaping the ceiling is necessary
and not sufficient, and that what decides the outcome is whether the selector
avoids a quadratic intermediate and whether the attention left behind is fused.
Abstract, introduction, Section 5, conclusion; `results/prescore.json`.

### Two sweeps quoted against an untuned dense kernel, and one kernel was flat by construction

Both found in an adversarial re-read of the two new experiments above.

**The objections.** (1) The sorted kernel padded its aggregation axis to a floor
of sixteen, so for every k below sixteen it read sixteen value rows and its cost
was flat in k by construction; the "flat in k" measurement was therefore not
evidence about the design, and "reads exactly k value rows" was false. (2) The
pre-score and sorted-select sweeps quoted their ratios against the vendor
library and our own dense kernel at its default tiling, while Section 4 commits
the paper to quoting against the better of the library and a searched dense
kernel — a weaker standard for the mechanism the paper proposes than for the one
it criticises. (3) The two sweeps had no repeat, and Section 5 read claims off
agent counts the paper's own policy excludes.

**Action.** (1) The block floor is now the power of two the axis requires and
nothing more; re-verified exact and re-measured. (2) `dense_tuned` is an
implementation in both harnesses, re-timed inside the same interleaved group,
and the tiling search was extended to 48 and 96 agents so every reported agent
count has a searched configuration. All sweeps re-run. (3) The pre-score sweep
is now run twice and the appendix reports how far the two runs agree, which is
also what licenses quoting it below 192 agents.

**Result.** The sorted kernel is not flat: it rises 14% from one to sixteen
retained peers, against eightfold for the argmax kernel, and the conclusion is
unchanged — having both properties does not help. Against the searched dense
kernel the escape case weakens and survives: 18.1× becomes 16.8×, and the
end-to-end crossing at 512 agents goes from 0.76 to 0.90, so the claim is now a
ten per cent margin restricted to two and four retained peers rather than an
unqualified crossing. The negative result is unchanged: at one retained peer the
fused sparse kernel is still faster than the best dense implementation.

### Raised and not acted on

- **Moving the learning study wholly to an appendix.** We compressed it and
  removed the non-vacuity claim's stronger form, but kept the design and the
  primary comparison in the body, because a reader deciding whether the
  sparsified architecture is inert should not have to go looking.
- **A dense kernel at $d{=}16$ reaching library-at-$d{=}128$ efficiency.** We
  could not write one and neither could the search. This is the single
  outstanding threat to the positive result and it is stated as such.

---

## TMLR review, 2026-10-09 (minor revision)

Locations name sections and appendix subsections of
`paper_submission_revision/main.pdf`. "Measured" marks a new experiment;
"analytic" marks a response that is argued, not measured.

### W1 / Q1. Single GPU architecture; portability to A100/H100

- **Response.** Measured. The headline sweeps were repeated on an A100-SXM4-40GB
  (sm_80), an H100 80GB HBM3 SXM (sm_90) and an H200 (sm_90), one GPU allocated
  exclusively per job, every tiling searched again on that GPU
  (`scripts/run_portability_campaign.sh`). Device SKU, memory, clocks, power
  limit and driver: `results/portability/<gpu>/device.csv`, `nvidia_smi_q.txt`;
  the Blackwell card is the RTX PRO 6000 Blackwell Server Edition (HARDWARE.md).
- **Location.** Appendix "Other GPU generations" (Table "headline quantities on
  four NVIDIA GPUs"); abstract; Introduction (Findings, Contributions); Section 4
  Setup, Q3 ("Streaming the selection"), Q4 ("bf16 on four GPUs"), Q5 ("The
  attention stage", end-to-end paragraph), Q6; Discussion (takeaways,
  Limitations).
- **Evidence.** Replicates on all four: composed gathered path slower than dense
  everywhere (≥6.6×); finding the set is 88–97% of fused time from N=384; both
  fused kernels faster than dense at k=1 in fp32; kernel count orders latency at
  N≤96 (ρ 0.91–0.95 vs ≤0.10 for operations). Does not replicate: streaming at
  k=2 is faster everywhere on Blackwell and A100 but at 7 of 8 configurations on
  H100/H200; at k=4 it is slower on all three non-Blackwell GPUs (1.03–1.50);
  bf16 k=1 is slower than dense on all three (1.03–1.52); the Z-order end-to-end
  crossing (0.85 on Blackwell) becomes 1.12–1.45.
- **Limitation.** No CPU or non-NVIDIA accelerator. The A100 runs the cluster's
  PyTorch 2.11 / Triton 3.6 build against CUDA 12.9 (12.8 elsewhere). We do not
  attribute the cross-GPU differences to a single hardware feature.

### Q2. Sensitivity of the fp32 crossover to batch and multi-head packing

- **Response.** Measured, on A100, H100, H200; B ∈ {32, 256, 1024, 4096}, both
  kernels searched per batch. Heads are independent, so H heads over batch B is
  one launch over batch BH; the batch sweep is the packing sweep.
- **Location.** Appendix "Other GPU generations", paragraph "Batch and heads"
  and Table "streaming ... as the batch varies".
- **Evidence.** Over 256–4096 the window does not widen (k=1 0.57–0.65, k=2
  0.83–0.96). At B=32 the result is less favourable and less uniform (k=1
  0.60–1.17; k=2 no longer faster at every N).
- **Limitation.** Not measured on the Blackwell card above B=1024; packing was
  not measured as a separate multi-head kernel, only through the equivalence
  to batch.

### Q3. CUDA-graph capture of the streaming kernel under ragged N or per-row k

- **Response.** Partly measured, partly analytic. Measured: the streaming kernel
  is captured and replayed at fixed (B, N, k) in every graph-mode sweep on all
  four GPUs. Analytic: k is a compile-time constant and N uniform per launch, so
  per-row k and ragged N are unsupported; we did not implement or measure them,
  and make no claim about their small-N latency.
- **Location.** Discussion, Limitations ("Its retained count is a compile-time
  constant ...").

### W3 / Q4. Hardware-counter evidence for the byte argument

- **Response.** Measured (`scripts/ncu_counters.py`) on H100 and H200; the A100
  cluster's nodes do not expose counters to users.
- **Location.** Appendix "Other GPU generations", paragraph "Hardware
  counters"; Section 3 "Byte criterion" (last sentences).
- **Evidence.** At N=512, k=4: the composed path moves 4.4 GB in 27 kernels
  against 28 MB for tuned dense; 17 kernels are torch.topk's multi-block radix
  selection (4 passes), 69% of the bytes and 81–82% of the time. The fused and
  streaming kernels reach 0.9–1.6% of peak DRAM throughput and 52–60% of peak SM
  throughput at 18–22% achieved occupancy; the fixed-set kernels take 4.1–4.8%
  of the streaming kernel's time.
- **What it does and does not show.** It shows the fused kernels are not
  bandwidth-bound and that selection dominates their time. It does not show
  them compute-bound in the sense of saturating a unit, and the manuscript says
  so. It also shows the byte criterion is necessary but not sufficient, since
  library top-k reads the scores several times; Section 3 now says this.
- **Limitation.** Counters at one shape (and 44 others in the artifact), under
  ncu's cold-cache, base-clock replay.

### Q5. Modern exact top-k kernels; amortised indexers

- **Response.** Partly measured. The composed baseline is PyTorch's exact
  torch.topk in the evaluated version, which the counters show is a multi-block
  radix selection; the manuscript describes it as such and not as the fastest
  exact selection possible. The fixed-set split bounds what any faster selector
  could recover: attention over a given set is 0.04–1.02× dense for k = 1–32
  and N ≥ 192 across the four GPUs.
- **Location.** Appendix "Other GPU generations" ("Hardware counters"); Section
  4 Q3 (Table "One retained set, reached two ways").
- **Limitation.** Sample-guided exact top-k, other radix-fused designs and
  amortised or cross-layer indexers are not benchmarked.

### Q6. Effect of bf16/TF32 retained-set deviations on training

- **Response.** Not measured. We measured the deviations, not their effect on
  training: bf16 changed the retained peers in at most 6 of 10 million query
  rows checked on any GPU (none on Blackwell); TF32 configurations changing more
  than 0.1% of rows are excluded.
- **Location.** Section 4 Q4 ("bf16 on four GPUs"); Discussion, Limitations
  ("We did not measure how the retained-set changes ... affect training").

### Q7. int32 indices and other dtypes in the byte criterion

- **Response.** Analytic. int32 halves the index term: k(d+2) ≤ N exactly;
  narrower value formats raise the index term's relative weight; kd ≤ N at
  leading order is unchanged. The composed path cannot use int32 indices
  (PyTorch's gather takes int64); the fused kernels never write indices.
- **Location.** Section 3, "Byte criterion".
- **Limitation.** Not measured.

### W2. Single-round, single-head; backward via dense autograd

- **Response.** Unchanged and stated. Multi-head is covered for latency through
  the batch equivalence above; the backward pass is measured for composed paths
  only (fused kernels are forward-only).
- **Location.** Discussion, Limitations; Appendix "Backward pass and training
  throughput".

### Missing related work / production sparse kernels

- **Response.** FlexAttention is the production sparse path compared (Section
  4 Q4). No additional library comparison was added in this revision.
- **Limitation.** Stated in Limitations.
