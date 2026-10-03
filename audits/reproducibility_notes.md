# Reproducibility audit: paper_submission_revision

Audited 2026-10-03 against working tree on `phase-c/experiments` (HEAD 53ef0f4).
All runs used a scratch copy (`rsync ... /tmp/claude-2011/repro_copy/`), CPU only
(`CUDA_VISIBLE_DEVICES=""`), `.venv/bin/python`. No tracked file was modified.
Matrix: `audits/reproducibility_matrix.csv`.

## 1. Run log summary

| Step | Command (in copy) | Exit | Wall | Result |
|---|---|---|---|---|
| Tables | `python scripts/make_revision_tables.py` | 0 | 0.06 s | 14 files written; `diff -r` against committed `tables/`: **byte-identical** |
| Number check | `python scripts/check_revision_numbers.py` | 0 | 0.14 s | `checked 383 conditions across 12 manuscript sections and 14 tables`; `manuscript agrees with every measurement artifact` |
| Figures | `python scripts/make_revision_figures.py` | 0 | 3.4 s | 5 PDFs; each differs from committed by 8-10 bytes = `/CreationDate` only (content identical). Set `SOURCE_DATE_EPOCH` / `metadata={'CreationDate': None}` for byte-identity |
| PDF | `tectonic --keep-logs main.tex` | 0 | 8.1 s | 34 pages (same as committed), no undefined refs/cites; warnings: 211 (Underfull boxes), `internal consistency problem when checking if main.bbl changed`, `TeX rerun seems needed, but stopping at 6 passes`. Committed PDF was produced by pdfTeX (TeX Live 2026), not tectonic; text equivalent modulo ligature extraction |
| Tests | `pytest -q tests/` | 0 | 4 s | 40 passed |
| Legacy: `make tables` | Makefile | 0 | - | re-creates `paper/tables/table_headline.tex`, `table_ablation.tex`, `headline_stats.json` (old MARL tables; unused by the manuscript) |
| Legacy: `make paper` | Makefile | would fail | - | `cd paper` — directory deleted in working tree |
| Legacy: `check_bib.py` | `make check-bib` | 1 | - | reads `paper/` → `cited keys: 0 uncited: 67`; `make check` therefore fails |
| Legacy: `check_paper_numbers.py` | - | 1 | - | `FileNotFoundError: paper/sections/04_measurement.tex` |
| Legacy: `reproduce.sh --dry-run` | - | 0 | - | plan still builds MARL tables/figures and compiles `paper/main.tex` |

Verdict: the revision pipeline (tables + figures + checker + tectonic) passes and is
current. Every repo-level entry point (`Makefile`, `reproduce.sh`, `README.md`) points
at the deleted `paper/` manuscript and does not reproduce this submission.

### Checker design weaknesses found while running it
- `check_revision_numbers.py:83-91` regenerates tables **in place** and compares to the
  pre-run copy. A stale table fails once, but the file is left overwritten, so a second
  run passes silently. It should regenerate into a temp dir.
- `check_num` (`check_revision_numbers.py:156-168`) takes the **first** regex match over
  all sections joined in `Path.glob` order (unsorted). When a number appears in both main
  text and appendix, only one instance is checked — in practice the appendix one
  (e.g. 0.85, 2.06, 4–17, 0.10–0.14, 16.8 are checked in `12_app_measurement.tex`, not
  in `04_measurement.tex`).
- Spelled-out numbers ("sixteen", "thirty-seven", "three to eight", "five hundred") are
  never checked except three `_word()` cases in appendices.
- Figures are not checked at all. `fig4_roofline.pdf` is generated but never included.
- `make_revision_tables.py:4-6` docstring still names the legacy `check_paper_numbers.py`.

## 2. Unchecked prose numbers (abstract, intro, measurement, limitations, conclusion)

Coverage: the checker's prose regexes hit 15 values in these five files, all in
`04_measurement.tex` (lines 111, 212, 264-282). Everything else below is **unchecked**
(~45 distinct quantitative claims; analytic constants such as "at most half" and
`kd <= N` derivations are listed separately). Values I re-derived during the audit are
marked; ✗ marks a disagreement or an ambiguous definition.

### Checked (for reference)
04:111 4.62 / 0.139 / 33; 04:212 99.93%; 04:264 0.32 / 5.69; 04:265 37; 04:268 0.51 / 1.67;
04:270 10; 04:274 0.86; 04:276 1.6; 04:282 0.92. Plus semantic guards: the
"slower than dense ... every configuration" regex (01/04/09 only — **not** the abstract),
grid slower than all-pairs at k=16 (04:269-272), sort kernel 5x/13-16x bounds (04:145-146,
hard-coded ranges, not the prose).

### Gap list

| file:line | number in prose | likely source artifact | audit re-derivation |
|---|---|---|---|
| 00_abstract.tex:12 | composed slower "at every configuration" | attention_core_d16.json (topk_gather vs best dense) | guard regex excludes 00_abstract |
| 00_abstract.tex:13-14 | faster only at one peer, narrow heads, no tensor cores | autotune.json, precision.json | — |
| 00_abstract.tex:14-15 | advantage disappears "beyond two hundred agents" | precision.json (k=1, tf32/bf16) | intro/conclusion say "from 384" — consistent only because grid jumps 192→384 |
| 00_abstract.tex:15-16 | **1.3 to 28** times slower from k>=4 | autotune.json (1.35 at d=32,N=512,k=4) + attention_core_sort.json | ✗ 28 matches only `topk_fused_gather` N=384 k=64 in attention_core_sort.json (27.96); best-fused over d16 sweep max is 18.8, single-kernel max is 42.8 (N=192,k=128), batch sweep min is 1.20 (b=32). Range is not a well-defined statistic |
| 00_abstract.tex:17 | "up to sixteen times faster" | prescore.json (1/min attention ratio = 16.8) | appendix 16.8 checked; abstract word not |
| 01_intro.tex:31 | six implementations | attention_core_d16.json impl list | — |
| 01_intro.tex:35 | three numerical formats | precision.json | — |
| 01_intro.tex:39 | **thirty-seven** (composed max) | attention_core_d16.json N=512 k=128 | ✓ 37.3 (not checked; 04:265's checked "37" is a different quantity) |
| 01_intro.tex:41 | one peer (two at a single configuration) | autotune.json "k that pays" | ✓ d=32,N=512 `<= 2` |
| 01_intro.tex:42 | head widths 16 and 32 | autotune.json | ✓ |
| 01_intro.tex:43 | dense gains **three to eight** times | precision.json | ✓ 2.73–3.98 TF32, 3.92–8.08 bf16 |
| 01_intro.tex:44 | gone from **384** agents | precision.json | ✓ k=1 ratios ≥1.07 at 384/512 |
| 01_intro.tex:47 | up to sixteen times faster | prescore.json | see abstract |
| 01_intro.tex:52 | parity at **five hundred** agents | prescore.json / prescore_selectors.json (0.85/0.86) | numeric value checked in 04/12_app; word not |
| 01_intro.tex:83 | head widths 16 to 512, at most 512 agents | attention_core_dsweep.json | — |
| 04_measurement.tex:28 | changes winner at **one of 24** configurations under medians | attention_core_d16.json dispersion | not re-derived |
| 04_measurement.tex:47 | 75% at N=512, k=127 | analytic (1-128/512) | ✓ analytic |
| 04_measurement.tex:67 | thirty-seven at 512, quarter of peers | attention_core_d16.json | ✓ 37.3 |
| 04_measurement.tex:72-73,80 | byte-neutral k<=32; four times past; k<=4 at d=128 | analytic (kd<=N) | ✓ analytic |
| 04_measurement.tex:76 | "about four times the bytes" | stage_accounting.json | not re-derived |
| 04_measurement.tex:78 | selection "under a tenth" of bandwidth | stage_accounting.json | not re-derived |
| 04_measurement.tex:83-84 | **18.4** fwd, **6.2** fwd+bwd | attention_core_d16.json (topk_gather vs dense_fused, N=512) | ✓ 18.42 / 6.17 |
| 04_measurement.tex:86 | environment takes **78 to 96%** | training_step_timing.json | ✗ table column "environment" is 48–92%; 78/96 is environment+acting (as 12_app_measurement.tex:312 says). Main-text wording is wrong |
| 04_measurement.tex:87 | within **2%** | training_step_timing.json (÷ fused 0.992–1.013) | ✓ |
| 04_measurement.tex:132 | **21 to 33%** vs **30 to 47%** | autotune.json % peak (d=16,32) | ✓ from table |
| 04_measurement.tex:134-135 | three vs **28 to 106** configs | autotune.json n_configs | ✓ from table (also 08:26-27) |
| 04_measurement.tex:142 | **1.3 to 2.1** at k=4 | autotune.json | not re-derived |
| 04_measurement.tex:145-146 | five to sixteen times dense (sort) | attention_core_sort.json | hard-coded bounds only |
| 04_measurement.tex:178-179 | **2.7–4.0** TF32, **3.9–8.1** bf16 | precision.json | ✓ |
| 04_measurement.tex:180-182 | **0.67–0.91**, **1.07–1.38**, **0.83–1.01** | precision.json k=1 | ✓ 0.67–0.91 fp32 |
| 04_measurement.tex:189-190 | **0.7%** rows differ; 0.1% exclusion | precision.json deviation `rows_differing` | not re-derived |
| 04_measurement.tex:193 | **nine of the twelve** | precision.json best_dense | ✓ 4 TF32 + 5 bf16 |
| 04_measurement.tex:195 | **8 to 218** | precision.json flex_topk:1 | ✓ 7.7–218.4 at k=1 only (234 at k=16; scope implicit) |
| 04_measurement.tex:251 | **0.06**, more than sixteen | prescore.json | appendix 0.060 checked; this instance not |
| 04_measurement.tex:257 | **4 to 17** | prescore.json | appendix 4.0/17.3 checked; this instance not |
| 04_measurement.tex:269 | about a hundred candidates | prescore_selectors.json / analytic | — |
| 04_measurement.tex:273 | **0.10 to 0.14** ms | prescore.json | appendix checked; this instance not |
| 04_measurement.tex:275 | first sweep's **0.85** | prescore.json | appendix checked; this instance not |
| 04_measurement.tex:283 | median **2.06** | prescore.json | appendix checked; this instance not |
| 04_measurement.tex:286 | **16** times slower at 48 agents | prescore.json | not re-derived |
| 08_limitations.tex:1 | **Seven** errors | 07_pitfalls.tex table rows | — |
| 08_limitations.tex:16-18 | envelope: sm_120, PyTorch 2.11, CUDA 12.8, Triton 3.6, d 16–512, N<=512, batch 32–1024, fp32/TF32/bf16 | JSON headers + campaign args | ✓ torch/triton/cuda recorded in JSONs |
| 08_limitations.tex:22 | no claim below **192** | design choice / launch_overhead.json | — |
| 08_limitations.tex:26-27 | N<=**1024** cap; three vs 28–106 | kernel code / autotune.json | — |
| 08_limitations.tex:33 | from **384** upward | precision.json | ✓ |
| 08_limitations.tex:38 | five seeds | results/red_* | — |
| 09_conclusion.tex:6-7 | bounded by two; N/k | analytic | ✓ analytic |
| 09_conclusion.tex:13-14 | three to eight times; from 384 | precision.json | ✓ |
| 09_conclusion.tex:15 | **1.3 to 28** in fp32 | as abstract | ✗ ambiguous, see above |
| 09_conclusion.tex:19 | up to sixteen | prescore.json | unchecked word |
| 09_conclusion.tex:21 | five hundred agents parity | prescore.json | unchecked word |
| 05_training.tex:44-58 (main-text table) | +4.26, +11.59, +4.03, 0.0079, 0.357, intervals, d=+0.60 | results/reduced_design_summary.md | ✓ values match, but table is hand-typed and unchecked |

Most important unchecked numbers: **1.3 to 28** (abstract + conclusion; upper bound not
reproducible as a defined statistic), **78 to 96%** (04:86; mislabelled), **thirty-seven**
(abstract-level headline; correct but unguarded), **sixteen / three to eight / 384 / five
hundred** (abstract/intro headline words), the precision ranges 04:178-195, and the
hand-typed `tab:training`.

### Structured manifest
None exists. Numbers live as regexes inside `check_revision_numbers.py`. A
`--verify-paper-numbers` manifest (e.g. `paper_numbers.yaml`) would need one entry per
quoted value with:

- `paper_number_id` (stable key, e.g. `abs.fused_k4plus_range.hi`), plus
  `file:line`-independent anchor text (the macro or phrase that carries it)
- `value` as printed and `unit` (ms, ratio×, %, count, word-number)
- `configuration` (artifact filter: impl set, N, k, d, batch, dtype, estimator
  min/median, which dense baseline set)
- `source_artifact` and a named `derivation` (function in the checker, e.g.
  `ratio_best_fused_over_best_dense`, `range_over(k>=4, N>=192)`)
- `tolerance` (abs/rel, plus rounding rule for spelled-out words)
- `occurrences` — every location the number is quoted (abstract, intro, body,
  appendix, conclusion), each verified, not just the first match

Coverage needed: every row in the gap table above (~45 main-text claims), the
~25 appendix values already checked, all spelled-out numbers, all range endpoints, and
the hand-typed `tab:training`. Cleanest implementation: emit `\newcommand` macros from
the manifest into `tables/macros.tex` (the mechanism already exists for `\SelectorsK`)
and have prose use macros, so values cannot drift.

## 3. Timing protocol: appendix vs code

Appendix (`10_appendix.tex:388-393`): fp32, CUDA events, interleaved, **20** warm-up,
**40 to 60** timed reps per block, block repeated **3** times, minimum over all samples,
spread of the three per-run medians; tiling search Q/K tiles 16–128, 2–8 warps, 1–3
stages, 100 kB shared memory.

| Artifact | Campaign args | Code default | Recorded in JSON | Matches appendix? |
|---|---|---|---|---|
| attention_core_d16 / _repeat / _sort | reps 60, runs 3 | warmup 20 (`bench_attention_core.py:428`) | reps 60 warmup 20 runs 3 | yes |
| attention_core_dsweep / _batch | reps 40, runs 3 | warmup 20 | reps 40 warmup 20 runs 3 | yes |
| stage_accounting | reps 50, runs 3 | warmup 20 (`stage_accounting.py:180`) | reps 50 runs 3 (warmup not recorded) | yes |
| prescore / _repeat / _selectors | reps 50, runs 3 | warmup 20 (`bench_prescore.py:163`) | yes | yes |
| autotune / autotune_small | reps 30 / 40 | warmup **15** hard-coded (`autotune_kernels.py:38`), single run | reps only | **no**: 30 reps, 15 warm-up, no 3x repeat |
| precision | (defaults) | reps 40, warmup **15**, runs 3 (`bench_precision.py:314-316`) | reps, runs (warmup **not recorded**) | precision warm-up undocumented (appendix scopes its sentence to "outside the precision study" but gives no precision parameters) |
| training_step | iterations 5, warmup 2 | same | iterations only | wall clock with sync, not CUDA events; documented only in 12_app, not app:repro |
| launch_overhead | defaults | reps 500, warmup 200 | yes | not in app:repro |

Seeds: input-tensor generator seeds are fixed in code (`bench_attention_core.py:337`
`1234+7N+d`, `:370` 99; `bench_prescore.py:174` `4242+N`; `bench_precision.py:335`
`77+N+d`; `autotune_kernels.py:177` 7; `stage_accounting.py:92` 0) and are **not
documented** in the appendix. They matter for the tie counts in `equivalence.json`.

Devices: campaign uses one `DEV` (default `cuda:0`), but `autotune.json`,
`stage_accounting.json`, `training_step_timing.json`, `sdpa_backend_probe.json` record
`cuda:1`. The appendix says "one ... of the two" devices.

## 4. Anonymisation hits (supplementary archive risks)

Scanned with `/usr/bin/grep` (the shell `grep` is aliased to ugrep with `--ignore-files`,
which hides gitignored files) over everything except `.venv/` and `.git/`.

Tracked files:
- `configs/environment.lock:2` — `-e file:///home/mhamdan/agentic_marl`
- `LICENSE:3` — `Copyright (c) 2026 Mohammed Hamdan`
- `README.md:179-180,189` — bibtex `author = {Mohammed Hamdan}`, name + `mh2022ets@gmail.com`
- `DECISIONS.md:128,329,413` — `git@github.com:MHHamdan/TopK-TarMAC.git`, public repo URL
- `RESEARCH_LOG.md:7` — `/home/mhamdan/agentic_marl`
- `REVISION_REPORT.md:277` — `/home/mhamdan/`
- `revision/TMLR_submission_package.md:151` — `/home/mhamdan/agentic_marl`
- `literature/references.bib:545-548` — `hamdan2026marliotp`, `author = {Mohammed Hamdan and ...}` (not cited by the submission's own `references.bib`, but a self-citation entry in the archive)

Untracked/ignored files that a naive zip of the working tree would include:
- `paper_submission_revision/main.log` — 67 lines with `/home/mhamdan/.TinyTeX/...`
- `.claude/settings.local.json` — name, email, `/home/mhamdan/...` (lines 10-185)
- `src/agentic_marl.egg-info/PKG-INFO:81` — `Mohammed Hamdan.`
- `PAPER_PROMPT.md:15`, `.ipynb_checkpoints/PAPER_PROMPT-checkpoint.md:15` — `/home/mhamdan/agentic_marl`
- `**/__pycache__/*.pyc` (28 files) — embed absolute `/home/mhamdan/TopK-TarMAC/...` source paths

Indirect: `results/provenance.json` (`git.sha` 35f6d10, branch `phase-b/defects`),
`results/MASTER_LOG.csv` and every `results/*/seed_*/config.json` carry git SHAs that
resolve on the public `github.com/MHHamdan/TopK-TarMAC`; the repo name `TopK-TarMAC`
plus README title make the work searchable. Git history: 65 commits authored
`MHHamdan` / `Mohammed Hamdan <mh2022ets@gmail.com>` — strip `.git`. No hostname
(`syrin`) or IP appears in any file; results JSONs contain no paths or usernames.
Clean: `src/kernels/*`, `scripts/*.py` sources, all `results/*.json`, `paper_submission_revision/{main.tex,sections,tables}` (author block is `Anonymous`).

No archive-building script exists; `TMLR_submission_package.md:149-151` lists the archive
and the lockfile scrub as open TODOs.

## 5. README / Makefile / docs staleness

- `Makefile:13-15,43-55,65-66` — `figures`/`tables`/`paper`/`clean` target `paper/` (deleted) via
  legacy `make_fig1_schematic.py`, `make_figures.py`, `make_tables.py`. No target runs
  `make_revision_tables.py`, `make_revision_figures.py` or `check_revision_numbers.py`.
  `check` fails because `check_bib.py:20` reads `paper/`.
- `scripts/reproduce.sh:96-122` — builds MARL tables/figures and compiles `paper/main.tex`;
  never calls the revision campaign or revision generators.
- `README.md` — titles the work "The FLOPs You Save Are Not the Time You Spend" (current
  title is "Score-Then-Select Sparse Attention in Multi-Agent Communication"); points to
  `paper/` and `check_paper_numbers.py` (lines 40-46); states sparse is "slower at every
  agent count and every sparsity level ... 0 of 46 cells" (lines 19-21), which the current
  manuscript contradicts (fused k=1 crossover; pre-score 16x); headline numbers are the old
  N in {3,6,12} MPE table; quick start and layout omit `src/kernels/`, the bench scripts and
  `run_revision_campaign.sh`.
- `results/REPRODUCTION.md` — MAPPO N=3 baseline note; unrelated to this manuscript.
- `main.tex:73-74` "a single script regenerates every table and figure" — it is two scripts;
  and "a second re-derives every number quoted in the text" (also `01_intro.tex:76-77`) is
  overstated per section 2.
- `10_appendix.tex:399-403` says the aggregate provenance record "has been corrected", but
  `results/provenance.json` was last committed 2026-08-05 with sha 35f6d10 (dirty), before
  the revision campaign it would need to cover.
- `10_appendix.tex:192` backend probe: `results/sdpa_backend_probe.json` has no producing
  script in the repo.

---

## Resolution in this revision

| Finding | Status |
|---|---|
| Makefile / `check_bib.py` target deleted `paper/` | Fixed: Makefile rewritten (`make check`, `tables`, `figures`, `paper`, `equivalence`, `fixed-set`, `supplement`); `check_bib.py` checks the submission bib |
| README describes an older paper | Fixed |
| Reproducibility statement overstated ("single script", "every number") | Fixed in `main.tex` |
| Checker matches first occurrence only; spelled-out numbers unchecked | Fixed for the main text: `scripts/paper_numbers.py`, 45 file-scoped entries, `results/paper_numbers.json` |
| Checker rewrites stale tables in place | Fixed: restores and fails |
| `tab:training` hand-typed | Disclosed in App. I (source: `results/reduced_design_summary.md`) |
| "Environment 78–96%" is environment + acting | Fixed in §4.3 |
| Tiling-search / precision protocols differ from the main sweeps | Disclosed in App. B (table caption) and App. I |
| No anonymised archive | `scripts/make_supplement.sh` (grep gate on names, email, home paths, self-citation, PDF metadata); excludes git history, provenance.json, MASTER_LOG.csv, decision logs |
| `provenance.json` stale; no script for `sdpa_backend_probe.json` | Open: excluded from the supplement; noted in `final_consistency_audit.md` |
| `reproduce.sh` still writes to legacy `paper/` | Open: it reproduces the learning-study pipeline only |
