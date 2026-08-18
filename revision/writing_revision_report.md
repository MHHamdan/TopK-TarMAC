# Writing revision report

Counts are over the LaTeX sources with float environments excluded, so tables do
not inflate prose statistics.

## Counts

| | archival `paper/` | revision `paper_submission_revision/` |
|---|---|---|
| Abstract, words | 334 | 301 |
| Abstract, mathematical notation | 11 symbols and one displayed ratio | none |
| Prose words | 10,561 | 12,287 |
| Em dashes | 118 | 50 |
| Words per em dash | 89 | 245 |
| `\paragraph{}` mini-headings | 20 | 6 |
| Sentences whose subject is a float | 13 | 8, all cross-references to appendix detail |
| Sentence-initial `First,`/`Second,`/`Third,`/`Finally,` | 12 | 3 |
| Occurrences of "rather than" | 49 | 33 |
| Opaque experiment codes in prose (P1–P7, C1–C6, G1/G2, tier-*n*) | 31 | 0 |
| *arm* / *cell* as nouns for implementations and configurations | 63 | 0 in the main text |
| Total pages | 26 | 29 |
| Main text / appendix pages | 20 / 6 | 23 / 6 |
| Appendix subsections | 11, five of which existed only to point at one float | 13 under 5 topic-based sections, none of which is a pointer |

The revision is longer because it contains a section, two tables and a figure
that did not exist — the fused-kernel comparison, the tiling search, and the
pre-score measurements. The parts that were criticised as dense are shorter: the
training study falls from 1,209 to about 800 words in the body, and the
seven-pitfall section's prose from a code-indexed list to a conceptual account
that needs no lookup table.

## Sections reorganised

| archival | revision |
|---|---|
| 2 Related Work **and a Reporting Audit** | 2 Related Work in three subsections; the audit protocol and the completed sheet move to Appendix D |
| 3 What Sparsifying Attention Can and Cannot Save | 3 Computational Limits of Score-Then-Select Attention, intuition before algebra, Proposition 1 restated as a special case of Proposition 2 |
| 4 Measuring the Cost | 4 Measured Cost, organised by question: what the composed implementations cost, what a purpose-built kernel changes, sensitivity, where the time goes, backward and training throughput, robustness |
| — | 5 Selecting Before Scoring (new) |
| 5 Does Sparsification Cost Return? | 6 Learning Study, compressed, inferential machinery to the appendix |
| 6 Seven Pitfalls in Attention Benchmarking | 7 Sources of Measurement Error, organised by the four kinds of failure, with an eighth pitfall added |
| 7 Limitations | 8 Limitations, five coherent paragraphs instead of eight run-in headings |
| 8 Conclusion | 9 Conclusion, five paragraphs with no tricolon |
| Appendix A, eleven subsections | Appendices A–E: additional results; implementation and measurement validation; additional training analysis; literature audit; reproduction and provenance |

## Representative before and after

**Abstract opening.**

> *Before:* "We built one and report why we abandoned it, at three levels. First, an analytic ceiling. ... Second, what executes. ... Third, what it costs. ... Constructively: a successor method must economise on the selector, not the aggregation."

> *After:* "Attention-based communication between agents is routinely sparsified on the grounds that attending to a few peers rather than to all of them removes most of the work, and the case is almost always made by counting operations rather than by measuring execution. We show that the count is smaller than it appears, and that what it buys in time is smaller still."

**Artifact narration.**

> *Before:* "Table~\ref{tab:headline} says *that* the sparse path loses; the following says where its time is spent."

> *After:* "Low arithmetic intensity in a batched matrix--vector product is the natural explanation for the sparse contraction's slowness, and it is not the right one."

**Opaque identifiers.**

> *Before:* "P1, P3, P6 and P7 are instrumentation errors ... P2 is a design error ... P4 and P5 are specification errors about *what is being compared to what*, and they are the dangerous ones."

> *After:* "A diagnostic that synchronises the device inside the timed region measures the synchronisation. ... The two most dangerous errors are of a different kind, because the resulting numbers are internally consistent, reproducible, and stable under every check the others would fail. They answer a different question from the one being asked."

**Experiment codes.**

> *Before:* "C1 is the contrast the design exists for: \texttt{gathered} and \texttt{random-}$k$ are matched on $k$, on executed FLOPs and on every hyperparameter."

> *After:* "The comparison the design exists for holds the communication budget, the executed aggregation cost and every hyperparameter fixed, and varies only whether peers are chosen by attention score or uniformly at random."

**Defensive register.**

> *Before:* "We should be direct that this was a design choice rather than a hard constraint ... and a reader is entitled to weigh them accordingly."

> *After:* "That was a budget allocation and not an unaffordable experiment: the 60-run study consumed 31.9 GPU-hours and a properly powered primary comparison would have cost about 37.9 more."

**Lab-diary framing.**

> *Before:* "This paper is a report on why we abandoned that design."

> *After:* "This paper is about the second. It asks what a particular and natural restriction — keeping, for each agent, the peers with the highest attention scores — saves in local computation, and whether the saving appears as elapsed time."

## The second writing pass

After the scientific rewrite the manuscript was read by an independent editor
who was told only to find what still sounded machine-made. Their three
strongest findings were acted on.

**Nineteen references to the manuscript's own earlier version.** The revision
narrated its own history — "the archival version of this work observed…", "which
the earlier version could only attribute to…", "we report the correction rather
than the claim". A reader of the accepted paper reads only the accepted paper,
and naming a previous version is a double-blind risk besides. All nineteen are
gone. Three technical statements survive where the earlier state is a fact about
the artifact rather than about the manuscript: the buffer-initialisation defect
in the bandwidth probe, the summation-order disagreement in an early kernel, and
the correction logged against the released pre-registration.

**One sentence written four times.** "Two limits bound the model's use." / "Two
limits of the environment bound what the study can say." / "Two properties of
this host bound what any of it can mean." / "Two properties bound what it can
support." — one per section, nouns swapped. The editor's judgement was that a
writer working over weeks does not converge on this and a per-paragraph
optimiser does. All four are rewritten and the wider family of enumerative
announcers is thinned.

**Virtue-marking.** Eleven clauses whose only function was to certify the
authors' own conduct: *we report the correction rather than the claim*, *we
report it as the null it is*, *the deviation is recorded rather than the artifact
silently edited*, *wrong in the direction that flattered the argument*. Every
underlying qualification is kept; the clause announcing it is gone.

Also acted on: "rather than" reduced from 49 to 33 by giving genuinely different
relations different constructions; eight sentences over 55 words split; the
`no_comm` / `dense_sdpa` code identifiers in Table 5 replaced with the names used
in every other table; and the paragraph-final epigram removed from roughly two of
every three paragraphs that carried one.

## Regression protection

`scripts/check_revision_numbers.py` fails the build on any of the banned phrases
above, on any table that regenerating would change, on any implementation whose
deviation from the reference exceeds floating-point tolerance, on mathematical
notation appearing in the abstract, on harness vocabulary appearing more than
twice in the main text, and on the absence of the three scoping phrases the
paper's claims depend on. It also re-derives the prose numbers that generation
cannot reach. It is not a style checker: ordinary uses of "first" or "Table" are
untouched, and the purpose is regression protection against text that was
already removed once.
