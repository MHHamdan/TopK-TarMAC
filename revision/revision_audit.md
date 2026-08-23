# Revision audit

Three parts. Part I maps every reviewer concern to an action and says whether it
threatens a central claim. Part II is the writing audit: the patterns that must
not survive into the submitted version, with the evidence for each. Part III
records what a second round of independent review — of the revision itself —
found, and which of the revision's own claims did not survive it.

The archival manuscript is `paper/`, unchanged, at commit `40d06e7`. The
revision is `paper_submission_revision/`. Checksums of the archival sources are
recorded so that byte-for-byte recoverability can be demonstrated rather than
asserted.

---

# PART I — Reviewer feedback mapped to action

## Summary of what changed, and what it cost the paper

The revision's most important finding is one the reviewer's criticism produced
directly. Asked for a purpose-built fused sparse kernel, we wrote one; it is much
faster than the composed PyTorch path, and at one retained peer it is faster than
any dense implementation we could obtain. **The manuscript's unconditional claim
that the sparse path is slower in every measured configuration does not survive
and has been replaced by a measured, conditional statement**, itself qualified by
a tiling search over both kernels (Part III). The operation-count ceiling is
untouched; what changes is the empirical claim built on top of it.

Two errors of fact found by the audit are corrected. Neither was caught by the
number checker, and both are in the direction of overstating a negative result.

---

## Scientific and experimental limitations

### Forward-pass-only benchmarking; no backward-pass timing

**Current evidence.** The harness already recorded a `forward_backward` field
for every configuration in `results/microbenchmark_comm.json` and
`results/microbenchmark_ksweep.json`. The manuscript never reported it. This was
unreported evidence, not missing evidence.
**Location.** Section 4 throughout; the omission is not acknowledged.
**Threatens a central claim?** Potentially. The gathered path's backward
involves a scatter-add whose relative cost is not obviously the same as the
forward's.
**Addressable by writing?** No. **By existing data?** Partly. **New experiment?**
Yes, for the corrected implementation set.
**Cost.** Included in the main sweep; no additional runtime.
**Action.** Report forward, backward and forward+backward separately for the
differentiable implementations. The hand-written kernels are forward-only and
are excluded from that comparison, which is stated rather than elided.

### No end-to-end training throughput evidence

**Current evidence.** None. The manuscript asserts that end-to-end wall-clock is
environment-bound and declines to measure it (Section 5.1).
**Threatens a central claim?** No, but it leaves the practical stakes of the
whole paper unquantified, and an assertion in place of a measurement is exactly
what the paper criticises elsewhere.
**Action.** RUN. New script `scripts/bench_training_step.py` times one complete
MAPPO iteration by phase. If the environment dominates and the implementations
are indistinguishable, that is the finding and it is reported as one, with the
iteration-to-iteration spread that bounds it.

### No purpose-built fused sparse kernel

**Current evidence.** An analytic copy-equivalent counterfactual, and a
limitations paragraph naming a fused kernel as the most informative follow-up
not run.
**Threatens a central claim?** **Yes, and it did.**
**Action.** RUN. `src/kernels/fused_topk_attention.py` provides two fused
kernels that write no score matrix, index tensor or gathered value tensor to
memory: one selecting by $k$ argmax rounds with the score product on the
matrix-multiply units, one selecting by an exact bisection on the integer key of
the score, whose cost does not grow with $k$ but which pays for that by
aggregating densely. Both verified against the reference at every configuration
whose latency is reported.
**Consequence.** The composed-implementation penalty is shown to be mostly
implementation. The remaining penalty at realistic $k$ is shown to be
algorithmic. The paper is stronger and one of its sentences is now false and has
been removed.

### The dense baseline was not of matched engineering effort

**Not raised by the reviewer; found while addressing the previous item, and it
is the more dangerous of the two.** Timing a hand-written sparse kernel against
a vendor library's dense kernel measures sparsity and authorship at once. At
$d{=}16$ PyTorch's fused attention path is well short of what a straightforward
Triton kernel achieves, so the first fused-sparse measurements showed an
apparent speedup larger than Proposition 1 permits from sparsity — which is the
signal that something other than sparsity produced it.
**Action.** `src/kernels/fused_dense_attention.py`: a streaming dense attention
kernel in the same framework, by the same author, in the same session. The
reported dense figure is the better of it and the library path. Added to the
measurement-error section as a distinct pitfall, since we would have published
the unmatched comparison.

### Incomplete exploration of very small $k$

**Current evidence.** `results/microbenchmark_ksweep.json` sweeps $k$ as a
fraction of $N-1$, so the smallest $k$ reached is 3 at $N{=}384$ and 5 at
$N{=}512$. The reviewer asks for $k \in \{1,2,4\}$, which is unmeasured at the
two largest agent counts.
**Threatens a central claim?** Yes: extreme sparsity is where the sparse path is
most favourable.
**Action.** RUN, at absolute $k \in \{1,2,4,8,16,32,64,128\}$ across the agent
count, head width and batch grid. Reported as a line plot of latency ratio
against retained peers, with the analytic count model overlaid, rather than as
another heatmap.

### Limited head-width range

**Current evidence.** The sweep stops at $d{=}256$ with the two most favourable
cells skipped for memory, and the manuscript declares the head-width trend
unresolved.
**Action.** RUN to $d{=}512$. Report the measured endpoint; do not extrapolate.

### One GPU generation and one software stack

**Action.** DO NOT RUN — unavailable. The host has two cards of the same model
and the older hardware is gone. Stated as a limitation without a same-model
second card being presented as architectural replication.

### CPU allocator dependence

**Current evidence.** The manuscript pins the glibc heap, counts minor faults
inside the timed region, and retains the superseded un-pinned run.
**Action.** No new experiment. Swapping allocators would establish that a
different allocator has a different page-return policy, which is not a fact
about attention.

### No energy or power measurement

**Action.** DO NOT RUN. Both GPUs carry other tenants' work throughout;
board-level power would attribute their computation to our kernels. `pynvml` is
absent and per-process energy accounting is unavailable to this account. The
manuscript states that no energy claim is made and why. A software proxy would
satisfy the request while being wrong.

### Insufficient bridge to modern block-sparse and fused methods

**Action.** Addressed by writing, with primary sources verified. The bridge now
names the three properties that separate that regime from ours — selection
amortised over blocks rather than per pair, block structure preserving the dense
matrix-multiply shape, and skipped work that is enormous at $10^5$ tokens and
bounded by a factor of two at $N \lesssim 10^2$ — and cites BigBird's own
statement that accelerators need coalesced block access rather than sporadic
lookups. No comparability is claimed across four orders of magnitude of sequence
length.

### No empirical escape case for Proposition 2

**Current evidence.** The proposition is stated and proved and never
instantiated.
**Action.** RUN. `src/kernels/prescore_selection.py` implements two selectors
that fix the retained set before any attention score exists: $k$ nearest peers
by Euclidean distance in the two-dimensional position space, which is the rule
DGN already uses; and agents ordered along a Morton curve with the $k$ adjacent
in that order, which is $O(N\log N)$ and so satisfies the proposition's
$o(Nkd)$ condition. Selector cost is measured separately from attention cost,
because a proposition that charges for the selector is only tested by an
experiment that charges for it.

### No explicit per-stage arithmetic-intensity and byte accounting

**Action.** RUN. `scripts/stage_accounting.py` reports, per stage, analytic
arithmetic, analytic traffic, intensity, measured time, achieved rate, achieved
bandwidth, and the fraction of whichever machine limit binds it. The last column
is what distinguishes a badly implemented stage from one doing the only thing
its access pattern permits.

---

## Errors of fact found by the audit and corrected

**The claim that neither trained variant beats a uniform-random policy at twelve
agents is false.** `07_limitations.tex:105` states it; the appendix contradicts
it on the same page turn. The random baseline returns $-758.9$; the dense
variant returns $-864.9$ and is indeed worse, but the gathered variant returns
$-717.9$ with a bootstrap interval of $[-756.7, -688.0]$ lying entirely above
the random mean. The correct statement supports the same conclusion — no
contrast at that agent count is interpretable — without asserting an absence the
data contradicts. Independently identified by two reviewers. Not guarded by the
number checker.

**The stated margin between the permutation floor and the first Holm threshold
is wrong.** `05_training.tex:57` gives $5\times10^{-4}$; $0.05/6 - 2/252 =
3.97\times10^{-4}$. Wrong at the only significant figure quoted, in a sentence
whose point is the tightness of the margin.

**The selection share is misquoted.** `04_measurement.tex:169` gives 51%;
$3.4258/6.534 = 52.4\%$. The denominator is not reconstructible from the
released artifact, which is itself the defect: the new stage accounting records
it.

**The CPU bandwidth probe reads an untouched buffer.** `scripts/roofline.py`
allocated the copy source with `torch.empty`, which on Linux is backed by the
shared zero page, so the host probe re-read one cache line and reported a
bandwidth several times the machine's real figure. Every CPU ridge point derived
from it is wrong, and the derived claim that the dense path is compute-bound on
CPU is wrong in the direction that protected the argument. Fixed, and the ridge
is now reported as two well-defined numbers — cache-resident and DRAM-resident —
rather than as a vague range.

**The byte model undercounts the composed implementations.** Each elementwise
step on the score matrix is out of place, so the matrix is written and read once
per step rather than once in total. The dense count was low by roughly a factor
of two, which happened to flatter the sparse path's relative traffic less than
it flattered dense's absolute attainment. Corrected.

**The explanation offered for the sparse contraction being slower is wrong.**
The manuscript attributes it to a batched matrix--vector product having lower
arithmetic intensity and being unable to use the same hardware paths. Against
measured bandwidth the sparse contraction attains a *higher* fraction of
achievable bandwidth than the dense matmul it replaces. Both are
bandwidth-saturated; the sparse one is slower because it moves about four times
the bytes. That yields a checkable criterion the paper now states instead:
gathering is byte-neutral only when $kd \le N$.

**The released pre-registration still asserts a statistical claim the manuscript
was corrected to remove** — that at most one contrast in the family can reach
Holm-corrected significance, which is false because Holm is a step-down
procedure. The manuscript is right and its own artifact is wrong. Recorded in
the pre-registration's deviations log rather than silently edited.

**Provenance is stale.** `HARDWARE.md` still declares the pinned PyTorch build
unable to execute on this hardware and quotes a lockfile hash that no longer
matches the file it names; `provenance.json` describes training runs from the
superseded host while the appendix calls them microbenchmark phases, and the
per-run configuration files it redirects to contain only a seed. Corrected, and
the appendix now points at what exists.

---

## Framing changes the audit forced

**The headline agent count.** The manuscript makes $N{=}512$ its headline while
its own related work says multi-agent communication operates at
$N \lesssim 10^2$. The attention-communication methods cited operate at between
two and roughly thirty agents. The headline moves to the top of the range those
methods occupy, and the larger counts are presented as a stress test.

**The absence claim becomes a taxonomy.** The manuscript claims that no audited
multi-agent communication paper selects peers by attention score after computing
all of them. That is defensible as scoped and vulnerable as read. Two genuine
instances exist outside the audited set — handshake-based collaborative
perception computes a matching score for every ordered pair and then thresholds
— and several methods run a pairwise selector before running pairwise attention,
paying a selector of the same order as the score matrix they avoid. Restating
Proposition 1 as the special case of Proposition 2 with $C_{\mathrm{sel}} = N^2d$
places every mechanism on one axis and replaces an absence claim with a result.

**Existing work already supports the thesis and was under-used.** The exact
top-$k$ attention of Gupta et al. reports that its mechanism is *as fast as*
vanilla attention and buys memory rather than time — an independent instance of
the paper's central finding, in the sequence regime, published in 2021. Routing
Transformer states the paper's own pre-score/post-score distinction explicitly.
Both are now cited for what they establish.

---

# PART II — Writing audit

Measured over the archival sources: 13,045 words of prose carrying 118 em
dashes, 34 sentences whose grammatical subject is a table or figure, 20
paragraph-level `\paragraph{}` headings, and 12 sentence-initial ordinals. The
problems below are structural rather than lexical, which is why the revision
rewrites rather than paraphrases.

## Patterns that must not survive

**Outline prose in place of an abstract.** The abstract is organised as
`First, an analytic ceiling. / Second, what executes. / Third, what it costs. /
Constructively:` — four sentence fragments used as section markers. Rewritten as
continuous prose with no mathematical notation.

**Artifact narration.** Fifteen sentences exist only to point at a float:
*Table 2 reports the outcome.* / *Table 6 gives the numbers behind Figure 3.* /
*Table 7 gives the per-stage numbers behind Figure 4.* / *Table 1 says that the
sparse path loses; the following says where its time is spent.* /
*Figure 4 and Table 7 give two results...* Three appendix subsections consist of
nothing but such a sentence and a float. Every one is deleted; the finding leads
and the float is cited in support.

**Opaque identifiers as prose subjects.** *P1, P3, P6 and P7 are instrumentation
errors. P2 is a design error. P4 and P5 are specification errors.* requires the
reader to hold seven arbitrary codes in working memory. The measurement-error
section is reorganised by the concept behind each failure. Likewise *C1 is the
contrast the design exists for* becomes a description of what is being compared.
Codes remain in the artifact repository and in the tables; they no longer carry
prose.

**Sentence-shaped section headings.** *The count is smaller than it looks.* /
*Even the reduced count need not be executed.* / *And the executed saving need
not be time.* / *Selection is tuned, and it matters how much.* / *Our sparse
implementation is not optimal, and we can bound by how much.* / *The
communicating architecture helps.* Twenty in total, eight of them inside a
1,052-word limitations section. Replaced by noun-phrase labels or dissolved into
running prose.

**The three-beat figure, four times.** The same tricolon organises the abstract,
the introduction's three headings, the conclusion's three sentence openers, and
the "three levels of abstraction" framing in both introduction and conclusion.
One statement of a three-part structure is appropriate; four is what makes a
manuscript read as generated. Reduced to one.

**A defensive register.** Twelve instances of asserting candour rather than
exercising it: *we state this plainly*, *we should be direct*, *we do not
obscure*, *we are careful about which*, *we flag that explicitly*, *we withdraw
any suggestion*, *rather than taken on trust*, *a reader is entitled to*, *a
reader who weights architectural replication heavily should discount*. Every
underlying qualification is correct and is kept; the meta-frame is removed.

**Lab-diary framing.** *We built one and report why we abandoned it.* / *This
paper is a report on why we abandoned that design.* The project history informs
the motivation and no longer supplies the scholarly frame.

**Duplicated passages.** Five statements appear near-verbatim in both the main
text and the limitations or appendix, with the same numbers and the same clause
structure: the FlashAttention substitution, the copy-equivalent counterfactual,
the same-device drift caveat, the timestamps concession, and the
device-independence chiasmus. Limitations now cross-references rather than
restates.

**Aphorism.** *X is only as good as Y* fires three times with three different
pairs. *Cannot rank what it has not scored*, *decide where to look before
looking*, *the arm meant to save cost*, *the hypothesis they were meant to test*
are memorable once and manufactured by the fourth repetition.

**Harness vocabulary in the narrative.** *arm* and *cell* appear throughout,
*cell* eighteen times in Section 4 alone, alongside *timing group* and *headline
cell*. Replaced by *implementation* and *configuration*.

**One substantive defect found while auditing prose.** `05_training.tex:132`
announces two secondary observations and lists three.

## What the writing audit does not change

The manuscript's habit of stating exactly what a measurement does and does not
support is its principal virtue and is preserved. The null-result language was
audited for slips into equivalence and none were found, in either the manuscript
or the pre-registration that bound it in advance. The scoping of Propositions 1
and 2, the separation of assumptions, and the refusal to make a channel-specific
causal claim from a parameter-unmatched control are all kept as they stand. The
revision removes the frame that announces those virtues, not the virtues.

---

# PART III — What a second round of review changed

The revision described above was given, complete, to an independent reviewer who
had not seen it, and to an independent editor told only to find what still
sounded machine-made. Both returned substantial findings. The full accounting is
in `reviewer_response_matrix.md` and `writing_revision_report.md`; the summary
is that three of the revision's own claims did not survive.

**The positive result was not identified.** The revision's one new positive
claim — a fused sparse kernel faster than a matched dense one — rested on
matched *authorship*, which does not identify the algorithm. Neither kernel's
efficiency against the machine was reported anywhere, so the claimed margin
could have lain entirely inside the dense baseline's deficit. A tiling search
over roughly a hundred configurations per kernel, with achieved rate reported as
a fraction of the device's own peak, shows the crossover surviving and shows the
sparse kernel to be the less efficient of the two — but also shows the crossover
existing only where nothing reaches half the machine, and disappearing where the
vendor library does. The claim is kept and labelled provisional.

**The escape-case section used the baseline the paper declares invalid.** Every
ratio in it was quoted against the vendor library rather than the better of the
two dense kernels — the pitfall the paper's own measurement-error section names,
committed in the paper that names it. Corrected, the claim that the pre-score
attention stage is faster than dense reversed. A second pass then found that the
corrected measurement still composed that stage from library operators, which is
the constant the paper's own argument says is an upper bound rather than a
property of a design; a fused kernel over the same pre-chosen peer sets is up to
nearly seventeen times faster than the best dense implementation, and the
Z-order path crosses below dense attention end to end at 512 agents with at most
four peers retained. Both corrections stand
in the paper: every ratio is against the better of the two dense kernels, and
both the composed and the fused columns are reported.

**A negative result measured a hundred kernel launches rather than a selector.**
The pre-score selector's cost, attributed in the text to a sort, was dominated
by a per-bit Python loop computing Morton codes — the same composition overhead
the paper wrote fused kernels to remove for the other half of its argument.
Replaced with a branch-free interleave, verified bit-identical.

**Two claims were asserted where an experiment was available.** The manuscript
said it could not write a kernel with both sparse aggregation and selection cost
independent of the retained count, naming in the same sentence the radix or
bitonic select that would test it; and it explained the pre-score negative
result by the gathered operand losing the matrix-multiply shape, having measured
only a composed implementation of it. Both were written. A sort-based kernel has
both properties and is not faster, so the claim becomes a measurement rather
than a limitation; a fused pre-score kernel refutes the explanation and reverses
the section's conclusion. `results/attention_core_sort.json` and
`results/prescore.json`.

**And the two new experiments were themselves audited before being believed.**
An adversarial re-read found that the sorted kernel's aggregation axis was
padded to a floor of sixteen, so its cost was flat below that by construction
rather than by design; that both new sweeps quoted their ratios against an
untuned dense kernel while Section 4 commits the paper to a searched one; and
that neither sweep had a repeat. All three were fixed and everything re-measured.
The sorted kernel is not flat (it rises fourteen per cent from one to sixteen
retained peers) and the conclusion drawn from it is unchanged; the escape case
weakens from eighteen to nearly seventeen times and from a twenty-five per cent
to a ten per cent end-to-end margin, and is now stated with the retained counts
it holds at.

Two further defects were structural rather than substantive and are also fixed:
the claim that equivalence is checked at every reported configuration was false
as implemented, and the operation-count ceiling was being used as a validity
check on wall-clock ratios, which a multiply--accumulate bound cannot support and
which contradicts the paper's own thesis.

The pattern across both rounds is the same one the paper is about. Every defect
found was a measurement that answered a different question from the one it was
labelled with, and in every case the label was more favourable to us than the
measurement was.

---

# PART IV — A third pass, on the rebuilt PDF

The compiled submission was read end to end against the artifacts once more,
after the second round's corrections had landed. Six defects survived that
round; none touches a central claim, and two of them were introduced by the
corrections themselves.

**Five LaTeX macros had been wrecked into literal text.** `\ref` and `\texttt`
arrived through an editing script as a carriage return or a tab plus the rest of
the macro name, so the PDF printed "Appendix efapp:verification", "Section
efsec:fused" twice, "extttfp32" and "extbf{Right:}". LaTeX reports no error for
any of it — the text is valid — and the build gate counted zero undefined
references because no reference was ever made. `check_revision_numbers.py` now
scans for control characters and for macro names missing their backslash, and is
verified to fail on an injected instance.

**The withdrawn matched-search claim was still standing in three places.** The
second round replaced "the same tiling space ... with the same budget" in
Section 4.3 and in Table 3's caption, but the identical sentence remained in
Limitations, and Section 7 and the introduction still asserted one shared tiling
space. All three now say what the artifact shows, and both phrases are banned by
the checker.

**A number came from a superseded artifact.** The gap between the two dense
baselines at 512 agents is quoted in Section 7 and in Table 9 as 4.6x. That is
`microbenchmark_comm.json`, which this campaign supersedes; the campaign's own
figure is 5.4x. Corrected, and now re-derived by the checker.

**The Z-order selector's discontinuity is explained rather than reported.** Its
cost falls from 0.37 to 0.13 ms at one cell of the sweep and stays there. The
dense baselines timed in that same cell fall with it -- the Triton kernel by 29%,
which has nothing to do with Morton codes -- so the shift is the device's state
and not the selector's. Ratios are formed within a cell and survive; the absolute
range does not, and the text now says so.

**Two pointers were wrong.** The reproduction appendix sent readers to
`check_paper_numbers.py`, the archival paper's checker, rather than the one that
validates this manuscript. And the literature audit described reading "the full
text, including appendices and the official supplement" without saying that the
reading was done through ar5iv where the paper is on arXiv, which the artifact's
own method field records.

The pattern of the first two rounds does not repeat here. These are production
defects and stale pointers, not measurements answering a different question from
the one they are labelled with -- with the single exception of the 4.6x, which is
that same failure in miniature and was caught by asking where each number's
artifact is.
