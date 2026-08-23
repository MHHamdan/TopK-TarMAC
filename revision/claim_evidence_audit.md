# Claim and evidence audit

Every headline claim in the revised manuscript, the evidence for it, the scope
it is stated within, and whether it survives.

Claims are quoted in the form the manuscript makes them, not in a stronger form.
Where a claim in the archival manuscript did not survive the revision, it appears
here as WITHDRAWN with what replaced it.

---

## Analytic claims

**A rule that selects communication partners by their attention scores can
reduce attention's multiply--accumulate count by at most a factor of two,
whatever the sparsity.**
Evidence: Proposition 1, proved. Scope: score-then-select rules — those whose
retained indices are fixed only after the complete score vector exists; per
head; single round; excluding the shared projections; excluding the cost of
selection itself, whose inclusion would only lower the ratio. Artifact: the
proposition is arithmetic and needs none, but the numeric identity is checked in
the regression tests. **Defensible: yes.**

**Proposition 1 is the special case of Proposition 2 in which the selector is a
full score matrix.**
Evidence: substituting $C_{\mathrm{sel}} = N^2d$ into the escape ratio recovers
the ceiling. Scope: an identity, not a measurement. **Defensible: yes.** New in
this revision; it converts an absence claim about the multi-agent literature
into a taxonomy of selectors by what they cost.

**A selector that fixes the retained set without forming the complete score
vector is not bound by the ceiling, and its count ratio is asymptotic to $N/k$
when the selector costs $o(Nkd)$.**
Evidence: Proposition 2, proved; the asymptotic statement is stated as such and
exact equality is claimed only for a free selector. Scope: operation counts
only. **Defensible: yes**, and Section 5 measures what it does and does not
imply about time.

---

## Measurement claims

**A sparse implementation assembled from general-purpose tensor operations is
slower than every dense implementation at every agent count measured.**
Evidence: `results/attention_core_d16.json`, Table 1. Scope: one GPU
architecture, `fp32`, head width 16, batch 256, agent counts 12 to 512, the
device shared with other tenants. **Defensible: yes.**

**The cause is memory traffic, not arithmetic: gathering is byte-neutral only
when the compacted operand is no larger than the score row it avoids re-reading.**
Evidence: `results/stage_accounting.json`. The sparse contraction attains a
higher fraction of achievable bandwidth than the dense multiply it replaces, so
neither is on a worse hardware path and the sparse one is slower because it
moves more bytes. Scope: the analytic byte counts ignore cache residency, which
at small agent counts they overstate; the table flags any stage whose attainment
exceeds 100% as cache-served. **Defensible: yes.**

**WITHDRAWN — "the sparse contraction is slower because a batched
matrix--vector product has far lower arithmetic intensity than a real GEMM and
cannot use the same hardware paths."**
This was the archival manuscript's explanation and the measurement contradicts
it. Replaced by the byte criterion above.

**WITHDRAWN — "the gathered path is slower than unfused dense in all 46 cells
and slower than a fused kernel in 45 of them," read as a claim about sparse
attention rather than about one implementation.**
A purpose-built fused kernel is faster than the best dense implementation at the
sparsest settings. Replaced by the conditional claim below.

**With a purpose-built fused kernel, sparse attention is faster than the best
dense implementation only when a single peer is retained, and is 14 to 26 times
slower at the sparsity multi-agent communication uses.**
Evidence: `results/attention_core_d16.json` and `results/autotune.json`,
Tables 2 and 3, Figure 2. The dense baseline is the better of the library fused
path and a search over roughly a hundred tilings of a dense kernel written in
the same framework. Scope: agent counts from 192 upward, where the sweep
reproduces on repetition; head widths 16 and 32, where the crossover exists at
all; `fp32`. **Defensible: yes, and provisional.** The crossover exists only
where no implementation reaches half the device's arithmetic rate, and
disappears at the head widths where the library does. A dense kernel closing
that gap at narrow head widths would remove it. We could not write one and
neither could the search, and the manuscript says so in the abstract, the
section and the limitations.

**The sparse kernel is the less efficient of the two, so residual engineering
bias runs against sparsity.**
Evidence: `results/autotune.json`. Across the head widths where the crossover
exists the sparse kernel attains 21–33% of the device's arithmetic rate against
30–47% for dense. **Defensible: yes.**

**WITHDRAWN — "the measured speedup stays inside the twofold analytic ceiling",
used as a validity check on a wall-clock ratio.**
A multiply--accumulate bound constrains time in neither direction, so this is
not a test, and using it as one contradicts the paper's own thesis. It survives
only in the measurement-error section as the smoke alarm that caught a real
error, described as that. The checker bans the phrase.

**WITHDRAWN — "a fused score-then-select kernel can have sparse aggregation
arithmetic or selection cost independent of the number retained, and we could
not obtain both."**
This was scoped as a statement about our implementations rather than a theorem,
and it was still an assertion standing in for a measurement. A third kernel
obtains both: it packs each score against its own peer index into one machine
word, sorts the row once, and reads exactly the retained value rows. Replaced by
the claim below.

**Both properties can be had at once and having both does not help.**
Evidence: `results/attention_core_sort.json`, Table 3. The sorted kernel's
selection cost does not grow with the retained count and its aggregation reads
exactly the retained rows: at 192 agents it runs in 0.164 ms at one retained
peer and 0.186 ms at sixteen, a rise of 14% against an eightfold rise for the
argmax kernel. It is five times dense attention across that range, overtakes the
argmax kernel only at sixteen retained peers where both already cost more than
five times dense, and at 384 and 512 agents is thirteen to sixteen times dense
at every sparsity below thirty-two retained peers, since a row-wide sort needs
the whole row resident. Scope: one architecture, `fp32`,
head width 16, batch 256, and three implementations rather than a proof about
all possible ones. **Defensible: yes.** New in this revision, and it replaces an
assertion with a measurement.

**Exact score-then-select cannot stream keys and values the way dense attention
can, because the complete score row must exist before it can be ranked.**
Evidence: structural, and visible in the block-size policy the sparse kernels
require. Scope: exact selection; an approximate or hierarchical rule is not
constrained. **Defensible: yes.**

**The ordering survives the backward pass with a smaller margin, because
selection has almost no adjoint.**
Evidence: `results/attention_core_d16.json`, Table 4. Scope: the differentiable
implementations only; the hand-written kernels are forward-only and excluded.
**Defensible: yes.** New in this revision.

**End to end, the implementations cannot be separated: a full training iteration
is dominated by the environment and by per-step action selection, and the
communication implementations differ by less than the variation between
iterations.**
Evidence: `results/training_step_timing.json`, Table 5. All five agree within
2% at every agent count, and the ordering among them is unstable: the
communication-free variant is fourth of five at six and twenty-four agents, and
the gathered sparse implementation is fastest at twenty-four. Scope: this
pipeline, few agents, a CPU-bound environment. **Defensible: yes**, and it is
reported as a failure to resolve rather than as an equality.

---

## Escape-case claims

**Both pre-score selectors deliver the operation-count saving the accounting
promises; one is an order of magnitude slower than dense attention end to end
and the other overtakes it.**
Evidence: `results/prescore.json`, Table 7 and Figure 4. Every ratio is against
the better of the two dense kernels. Scope: a cost comparison; the selectors
attend to different peers than a top-$k$ rule, so no quality claim is made.
**Defensible: yes.**

**A pre-score selector whose decision is reached through a quadratic structure
reproduces the cost it was introduced to avoid, and gets worse with scale.**
Evidence: the nearest-neighbour selector saves 15 times the operations at 512
agents and its path costs 30.8 times dense attention there, up from 9.2 at 48
agents. **Defensible: yes.**

**WITHDRAWN — "a sub-quadratic selector makes the attention itself faster than
dense attention," as measured against the vendor library.**
The comparison was against the library dense kernel, which is the pitfall the
paper's own measurement-error section names. Every ratio in the section is now
against the better of the library kernel and ours.

**WITHDRAWN — "attention over a pre-chosen peer set is not cheaper than
attention over all peers, despite up to 104 times fewer operations, because the
gathered operand loses the matrix-multiply shape."**
Measured against the right baseline but composed from library operators, which
write and re-read a $(B,N,k,d)$ tensor twice. The paper's own argument about the
score-then-select path is that such a constant is an upper bound and not a
property of a design, so this applied one standard to the mechanism the paper
criticises and another to the mechanism it proposes. A fused kernel over the
same pre-chosen peer sets refutes both the claim and its explanation.

**Fused, attention over a pre-chosen peer set is faster than the best dense
implementation at every agent count measured when at most four peers are
retained, by up to nearly seventeen times, and a sub-quadratic selector in front
of it crosses below dense attention end to end at 512 agents.**
Evidence: `results/prescore.json`, `results/prescore_repeat.json`, Table 7,
Figure 4. At two retained peers the attention stage costs 0.67, 0.35, 0.19, 0.08
and 0.06 times dense from 48 to 512 agents, at four 0.76, 0.43, 0.25, 0.12 and
0.09; the best case over the grid is 0.060. At sixteen retained peers it is
slower than dense at the smallest agent count. Composed from library operators
the same stage is slower than dense at all 40 configurations, so fusing is worth
3.9 to 17.4 times. The Z-order path's total costs 16.8, 8.8, 4.0, 1.28 and 0.91
times dense at four retained peers; the crossing holds at two and four retained
peers, is level at eight and gone at sixteen. Every ratio is against the best of
the vendor library, our own dense kernel and our own dense kernel at the searched
tiling. Scope: one architecture, `fp32`, head width 16, batch 256, centralised
inference, and cost only — the selectors attend to different peers than a
top-$k$ rule and no quality claim is made.
**Defensible: yes, and provisional in the same way as the negative result**: it
is what we could build; the crossing is a ten per cent margin, it sits at the top
of our agent range, and it holds only at the two sparsest settings. Both runs of
the sweep put the cells that carry it within 1.5% of each other.

**Escaping the operation-count ceiling is necessary and not sufficient, and what
decides the outcome is two things a count cannot express.**
Evidence: both selectors satisfy Proposition 2 by between 7 and 107 times in
operations at four retained peers; the quadratic one ends an order of magnitude
behind dense attention and the sub-quadratic one overtakes it, and the
sub-quadratic one overtakes it only once its attention stage is fused. The two
conditions are whether the selector avoids a quadratic intermediate and whether
the sparse attention left behind is executed by a kernel written for it.
**Defensible: yes.** This is the revision's main constructive contribution, and
it exists only because the escape case was measured — against the right
baseline, and with the same implementation standard the paper applies to the
mechanism it criticises.

## Learning-study claims

**The communicating architecture improves return on both positive controls.**
Evidence: `results/reduced_design_summary.md`. Both controls separate completely
and attain the permutation floor. Scope: **the architecture, not the channel** —
the communicating variants carry 7–19% more parameters and one control task
supplies a communication channel of its own. **Defensible: yes**, with the scope
stated in the same paragraph.

**Six pre-registered sparsification comparisons are under-powered nulls.**
Evidence: as above. Scope: no equivalence claim is available, because no
equivalence margin was registered. **Defensible: yes.**

**CORRECTED — "at twelve agents neither trained variant beats a uniform-random
policy."**
False. The random baseline returns −758.9; the dense variant returns −864.9 and
is worse, but the sparse variant returns −717.9 with a bootstrap interval
entirely above the random mean. The manuscript now says that the dense variant
is worse than random and the sparse variant barely better, which supports the
same conclusion — that no comparison at that agent count is interpretable —
without asserting an absence the data contradicts.

---

## Claims deliberately not made

- No energy or power claim. The device is shared; board-level power would
  attribute other tenants' work to our kernels.
- No claim about a second GPU generation. The host has two cards of the same
  model.
- No claim about bandwidth-constrained distributed deployment. Every cost result
  is centralised single-device inference.
- No claim that no score-then-select implementation can ever cross over at
  realistic sparsity. We report the best three we could write, including one
  that has both properties the other two trade off, and bound what remains.
- No claim that the pre-score crossover extends below or above the agent range
  measured. It is reported where it was observed, at the top of that range.
- No equivalence claim anywhere in the learning study.
- No claim of comparability between our agent-count regime and the long-context
  regime of block-sparse attention, in either direction.
- No claim that the pre-score selectors preserve task performance. They were
  measured for cost.
