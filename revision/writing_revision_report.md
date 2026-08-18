# Writing revision report

Measured over the LaTeX sources, with float environments excluded from word and
sentence counts so that tables do not inflate prose statistics.

## Counts

| | archival `paper/` | revision `paper_submission_revision/` |
|---|---|---|
| Abstract, words | 334 | 279 |
| Abstract, mathematical notation | 11 symbols and one displayed ratio | none |
| Prose words, all sections | 10,561 | see note below |
| Em dashes | 118 | 43 |
| `\paragraph{}` mini-headings | 20 | 6 |
| Sentences whose subject is a float | 13 | 7, all genuine cross-references to appendix detail |
| Sentence-initial `First,`/`Second,`/`Third,`/`Finally,` | 12 | 3 |
| Appendix subsections | 11 | 5 sections with 13 subsections, consolidated by topic |
| Opaque experiment codes in prose (P1–P7, C1–C6, G1/G2, tier-*n*) | 31 occurrences | 0 |
| Uses of *arm* and *cell* as nouns for implementations and configurations | 63 | 0 in the main text |

Note on total word count: the revision adds a section that did not exist (the
pre-score measurements) and a fused-kernel subsection, and removes an equivalent
volume from the appendix and the training study. Main-text density is lower
because the diagnostic material moved: the training study falls from 1,209 to
about 700 words in the body, and the seven-pitfall table's supporting prose from
605 to a conceptual account that no longer requires a lookup table.

## Sections reorganised

| archival | revision |
|---|---|
| 2 Related Work **and a Reporting Audit** | 2 Related Work, in three subsections; the audit protocol and completed sheet move to Appendix D |
| 3 What Sparsifying Attention Can and Cannot Save | 3 Computational Limits of Score-Then-Select Attention, with the intuition before the algebra and Proposition 1 restated as a special case of Proposition 2 |
| 4 Measuring the Cost | 4 Measured Cost, reorganised around questions: what the composed implementations cost, what a purpose-built kernel changes, sensitivity, where the time goes, backward and training throughput, robustness |
| — | 5 Selecting Before Scoring (new: the empirical escape case) |
| 5 Does Sparsification Cost Return? | 6 Learning Study, compressed, with the inferential machinery moved to the appendix |
| 6 Seven Pitfalls in Attention Benchmarking | 7 Sources of Measurement Error, organised by the four kinds of failure rather than by numbered code, with an eighth pitfall added |
| 7 Limitations | 8 Limitations, four coherent paragraphs instead of eight run-in headings |
| 8 Conclusion | 9 Conclusion, four paragraphs with no tricolon |
| Appendix A, eleven subsections | Appendices A–E: additional results; implementation and measurement validation; additional training analysis; literature audit; reproduction and provenance |

## Representative before and after

**Abstract opening.**

> *Before:* "We built one and report why we abandoned it, at three levels. First, an analytic ceiling. ... Second, what executes. ... Third, what it costs. ... Constructively: a successor method must economise on the selector, not the aggregation."

> *After:* "Attention-based communication between agents is routinely sparsified on the grounds that attending to a few peers rather than to all of them removes most of the work, and the case is almost always made by counting operations rather than by measuring execution. We show that the count is smaller than it appears, and that what it buys in time is smaller still."

**Artifact narration.**

> *Before:* "Table~\ref{tab:headline} says *that* the sparse path loses; the following says where its time is spent."

> *After:* "Timing each stage in isolation and charging it with its own arithmetic and traffic corrects an explanation the earlier version of this manuscript offered."

**Opaque identifiers.**

> *Before:* "P1, P3, P6 and P7 are instrumentation errors: they inflate or flatten a measurement and they leave traces. P2 is a design error that lets a shared environment leak into a comparison. P4 and P5 are specification errors about *what is being compared to what*, and they are the dangerous ones."

> *After:* "A diagnostic that synchronises the device inside the timed region measures the synchronisation. ... Synchronising without naming the device is the complementary error. ... The two most dangerous errors are of a different kind, because the resulting numbers are internally consistent, reproducible, and stable under every check the others would fail. They answer a different question from the one being asked."

**Experiment codes in the training study.**

> *Before:* "C1 is the contrast the design exists for: \texttt{gathered} and \texttt{random-}$k$ are matched on $k$, on executed FLOPs and on every hyperparameter."

> *After:* "The comparison the design exists for holds the communication budget, the executed aggregation cost and every hyperparameter fixed, and varies only whether peers are chosen by attention score or uniformly at random."

**Defensive register.**

> *Before:* "We should be direct that this was a design choice rather than a hard constraint ... and a reader is entitled to weigh them accordingly."

> *After:* "That was a budget allocation rather than an unaffordable experiment: the whole 60-run study consumed 31.9 GPU-hours and a properly powered primary contrast would have cost about 37.9 more."

**Lab-diary framing.**

> *Before:* "This paper is a report on why we abandoned that design."

> *After:* "This paper is about the second. It asks what a particular and natural restriction — keeping, for each agent, the peers with the highest attention scores — saves in local computation, and whether the saving appears as elapsed time."

## Mechanical announcements removed

Fifteen sentences whose entire content was a pointer to a float were deleted and
their references folded into the substantive sentence they supported. Five of
them were the sole prose in an appendix subsection; those subsections are gone,
their floats absorbed into topic-based appendices. The remaining seven
float-first sentences in the revision are cross-references directing a reader to
detail that was deliberately moved out of the main text, which is the pattern
the brief asks for rather than the one it bans.

## What the rewrite deliberately preserved

The archival manuscript's habit of stating exactly what a measurement does and
does not support is its principal strength. Every scoping qualification, every
refusal to convert a null into an equivalence claim, and every acknowledgement
of a confound is retained. What was removed is the frame that announces those
qualities — the sentences asserting the authors' candour rather than
demonstrating it — not the qualities themselves.
