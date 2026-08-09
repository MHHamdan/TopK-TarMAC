# Baseline Revision Report — Phase 1

Snapshot of `paper/` taken before any editing, at commit `7d525cf`.

## Build

| Item | Value |
|---|---|
| Toolchain | TinyTeX, pdflatex + bibtex |
| Missing package at start | `eso-pic.sty` (required by `tmlr.sty`); installed via `tlmgr install eso-pic` |
| Compile after fix | clean: `pdflatex → bibtex → pdflatex → pdflatex` |
| LaTeX errors | 0 |
| Undefined references | 0 |
| Undefined citations | 0 |
| Multiply-defined labels | 0 (in `main.tex`; see note on orphaned `tables/`) |
| Overfull/underfull boxes | 0 |

**Blocker for a clean clone:** `eso-pic` is not vendored and was not installed. Anyone
building from a fresh TeX install hits a fatal error. Either vendor it or document it.

## Page accounting

| Item | Pages |
|---|---|
| Total PDF | 17 |
| Main content (title → end of Broader Impact) | 1–15 (References heading falls mid-page 15) |
| References | 15–16 |
| Appendix | 16–17 |
| **Main content before references** | **≈14.5** |

Target is ≤12 pages of main content. ~2.5 pages need to come out.

## Word counts (source, `wc -w`)

| File | Words |
|---|---|
| `sections/00_abstract.tex` | 427 |
| `sections/01_intro.tex` | 819 |
| `sections/02_related.tex` | 719 |
| `sections/03_analysis.tex` | 1248 |
| `sections/04_measurement.tex` | 1637 |
| `sections/05_training.tex` | 1027 |
| `sections/06_pitfalls.tex` | 572 |
| `sections/07_limitations.tex` | 540 |
| `sections/08_conclusion.tex` | 440 |
| `sections/09_appendix.tex` | 743 |
| `main.tex` (incl. Broader Impact) | 255 |
| **Total** | **8427** (excl. orphaned `tables/`) |

The abstract at 427 words is roughly double a normal TMLR abstract and is the single
clearest length defect.

## Structural finding: `paper/tables/` is orphaned and stale

`main.tex` and every file in `sections/` were grepped for `tables/`. **No file in the
manuscript inputs anything from `paper/tables/`.** All tables are inlined in the section
files.

The three files in `paper/tables/` are artifacts of an abandoned earlier version of this
project:

- `table_headline.tex` — reports "Adaptive TopK (ours)" against MAPPO on *fully observable*
  `simple_spread` at N ∈ {3,6,12}, with Welch t-tests and n = 3 seeds at N = 12. It also
  claims `\label{tab:headline}` — the **same label** the current Section 4 headline latency
  table uses. It is only harmless because it is never included.
- `table_ablation.tex` — contains two rows with `--` and `n = 0` seeds, i.e. cells that were
  never run.
- `headline_stats.json` — the data behind both, from the superseded study.

These are inconsistent with the current manuscript in framing (the paper no longer proposes
a method), in statistics (Welch t-test vs. the pre-registered exact permutation test), and
in environment (fully observable vs. the PO variant). They are audited as defective in
`AUDIT.md` (entries D-08, D-12, D-20, D-21). Shipping them in a submission source tree is a
liability. **Action: delete from `paper/`.** (`scripts/make_tables.py`, which generates them,
belongs to the same superseded pipeline.)

## Section inventory (as found)

| § | Title | Content |
|---|---|---|
| 1 | Introduction | Problem, two-senses framing, mechanism preview, 6 contribution bullets, scope |
| 2 | Related Work: Who Measures, and Who Counts | Chronological MARL list; 12-paper latency-reporting audit table; efficient-attention precedent; sparse/budgeted mechanisms; MARL rigour |
| 3 | What Sparsifying Attention Can and Cannot Save | masked/gathered/fused taxonomy + Fig 1; Prop 1 (2× ceiling) + A1–A5; Prop 2 (structural escape); roofline + Fig 4 + Table 2 |
| 4 | Measuring the Cost | Protocol; headline latency table; robustness; crossover surface + Fig 2; component decomposition + Fig 3 + Table 4; O(BN³) gate memory + Fig 5; drift control + Table 5 |
| 5 | Does Sparsification Cost Return? | Design; PO env characterisation; pre-registration; results table; positive control; six nulls; descriptive mechanism |
| 6 | Five Pitfalls in Attention Benchmarking | P1–P5 as prose paragraphs |
| 7 | Limitations | 7 paragraphs |
| 8 | Conclusion | Recap + constructive direction + methodological lesson |
| — | Broader Impact | In `main.tex`, 1 paragraph |
| A | Appendix | Dispersion; crossover surface tabulated; implementation verification; reproduction |

Note that the roofline analysis currently sits in Section 3 (analysis) while the
measurements it interprets are in Section 4, so Figure 4 and Table 2 are presented before
the latencies they explain.

## Pre-identified consistency suspects (carried into Phase 2)

Recorded here before artifact verification; each is dispatched to a specialist audit.

1. **Roofline contradiction.** Table 2's caption says "Both paths sit far below both ridge
   points", but the same table reports a CPU ridge of 3.68 FLOP/byte and dense arithmetic
   intensity of 3.98 (N=192) and 4.18 (N=512) — above it. Figure 4's caption says the
   opposite of Table 2's ("on the CPU ... the dense path reaches the ridge"). The two
   captions cannot both be right.
2. **"46 of 46".** §4.3 states 0/46 vs unfused dense and 1/46 vs fused. The abstract says
   "slower than dense in 46 of 46 cells"; the conclusion says slower "in every one of 46
   cells ... against a fused dense baseline", which directly contradicts §4.3.
3. **Envelope conflation.** Intro and abstract attach "on two architectures, at two
   precisions, under three execution modes" to the 46-cell count. The 46-cell sweep is one
   device/dtype/mode; the robustness runs are separate.
4. **Minimum ratio 1.14×.** Claimed as the smallest ratio "anywhere". The headline table's
   smallest vs. unfused is 1.17 and the crossover surface's smallest is 1.16. The 1.14 must
   come from a run not named in the sentence; its baseline and universe need pinning.
5. **Directional drift claim.** "*Every* drift moves in the direction that makes the sparse
   path look worse" is a universal over per-cell signs and needs counting.
6. **Internal contradiction on kernel headroom.** §4.5 says the loss "is not an engineering
   defect a better kernel would remove. It is intrinsic"; §7 says a fused custom kernel
   "could plausibly close part of the gap".
7. **`architecture-independent`.** Asserted in §3.3 on the strength of one GPU
   architecture, a same-model second device, one CPU, and an analytic argument.
8. **`O(BN³)` gate memory** is derived for one natural implementation but stated in the
   abstract as a property of "a learned per-agent gate".
9. **Proposition 1 indexing.** Setting masks self-attention and the abstract says "all N−1
   peers", but the counts use N²d and Nkd.
10. **Voice.** "does not survive contact with the hardware", "was never going to help",
    "Two honest caveats", "the arm whose purpose is to reduce cost is the first to run out
    of memory", "where we would spend the next unit of effort" — conference rhetoric in a
    journal submission.
