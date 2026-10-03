# Cross-document consistency audit (post-revision)

Compared: main text (`sections/00`–`09`), appendices (`10`–`12`, `05`, `07`),
generated tables, figure scripts, code, artifacts (`results/*.json`), README,
Makefile, supplement builder.

## Mechanical gates (all pass at the revision commit)

| Gate | Command | Result |
|---|---|---|
| Tables regenerate byte-identically from artifacts | `scripts/check_revision_numbers.py` §1 (now restores a stale table instead of silently fixing it) | 15 tables, no diff |
| Prose numbers re-derived from artifacts | `scripts/check_revision_numbers.py` | 391 conditions pass |
| Main-text number manifest, file-scoped, incl. spelled-out numbers | `--verify-paper-numbers` → `scripts/paper_numbers.py` → `results/paper_numbers.json` | 45/45 pass |
| Bibliography | `scripts/check_bib.py` (now on the submission bib) | 46 entries, all cited, none dangling |
| LaTeX build | `tectonic main.tex` | no undefined references or citations |
| Tests | `pytest -q` | see commit message |
| Anonymity of supplement | `scripts/make_supplement.sh` grep gate | no identifying strings |

## Items reconciled by hand

| Item | Before | After | Where |
|---|---|---|---|
| Fused k=4 slowdown | "1.3 to 2.1" (precision sweep only) in §4/App. B; "1.3 to 28" in abstract/conclusion | "1.3 to 2.3" everywhere (precision + main sweep, N ≥ 192, fp32) | 00, 04, 12 |
| Reduced-precision crossover | abstract "beyond two hundred agents", intro "from 384" | "from 384 agents" everywhere | 00, 01, 04, 09 |
| Median wording | "paired median" (not paired) | "ratio of medians"; "paired" kept only where per-repetition pairing is real (fixed-set sweep) or for interleaved ratios | 04, 12 |
| Z-order median at k=4, first sweep | 2.06 (k=2 value) | 2.10 at k=4 in §4; 2.06 kept in App. B where the context is k=2 | 04, 12 |
| Agent-count literature claim | "two to thirty, DGN low hundreds" | "two to roughly thirty; DGN and ATOC about one hundred; mean-field one thousand" | 02, 12 |
| Measurement-error count | "Seven errors" (§5) vs "Eight ways" (App. C table) | "eight" | 08 |
| Ops-saved range | "between 7 and 108" | "7 and 107" | 12 |
| Byte thresholds | "k ≤ 32 / k ≤ 4" as exact | leading-order with exact k ≤ 25 / k ≤ 3 | 03, 12 |
| Training bound | "visible in the measurements" | bound is for a custom backward; measured autograd backward is 2N/(N+k) | 03, 11, 12 |
| Assumption list | six (A1–A6) | seven (A7 element-wise work) | 03, 11 |
| Figure 1 semantics | implementation colours | stage-role colours shared with the per-stage figure (`figstyle.ROLE_COLOR`) | fig1, fig3 |
| Hardware for new sweep | — | fixed-set sweep on cuda:0; main sweeps mostly cuda:1; both stated | 04, 08, App. I |
| Tiling-search timing protocol | "every claim is from an interleaved sweep" | exception stated; 30 reps, sequential, peaks 58 vs 71 TFLOP/s | 08, App. B, App. I |
| README / Makefile | old paper, deleted `paper/` dir, "0 of 46 cells" | current paper and entry points | README.md, Makefile |

## Known residual inconsistencies (documented, not hidden)

- `results/provenance.json` predates the revision campaign (2026-08-05) and
  `results/sdpa_backend_probe.json` has no generating script in `scripts/`; both
  are excluded from the supplement and noted in App. I's provenance caveats.
- `scripts/reproduce.sh` reproduces the learning study pipeline and still writes
  to the legacy `paper/` tree; the measurement paper's entry points are the
  Makefile targets.
- Script names contain "revision" (`make_revision_tables.py`, etc.); this marks
  the paper as revised but does not identify the authors.
- Equivalence references are fp32 composed paths; float64 references exist only
  in the precision study. Stated in App. C.
