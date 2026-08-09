# TMLR Revision Report

Companion documents: `baseline_revision_report.md` (Phase 1) and
`claim_evidence_audit.md` (Phase 2, 78 audited claims). Commit `b65842c`.

---

## 2. Before / after

| | Before | After |
|---|---|---|
| Total PDF pages | 17 | 22 |
| Main content before references | ~14.2 (refs start p15) | ~17.3 (refs start p18) |
| Appendix | p16–17 | p19–22 |
| Word count (source) | 8,427 | 11,803 |
| Abstract | 427 words | 382 words |
| Compilation errors | 0 (after installing `eso-pic`) | 0 |
| Undefined refs / citations | 0 / 0 | 0 / 0 |
| Overfull boxes | 0 | 0 |
| Unreferenced floats | 6 | 0 |
| `check_paper_numbers.py` | 63/63 pass — with 13 false statements live | 63/63 pass |

The paper got longer, not shorter. Nearly all of the growth is disclosure that
the audit showed was missing: two new benchmarking pitfalls, a cross-architecture
section that had to explain an allocator artifact, the `sorted=False` sensitivity,
the N=12 random-policy floor, the parameter-count mismatch in the positive
controls, and the corrected compute-budget accounting. TMLR sets no page limit
("Submissions may be any length"), and the brief prioritised scientific clarity,
so I did not trade these away. I did cut redundancy: the scope paragraph
appeared four times and now appears twice, and the pitfalls section was
converted from seven prose blocks to a compact table.

---

## 3. Major writing and structural changes

**The target was reframed from a literature to ourselves.** This is the single
largest change. `literature/literature_review.md:244` states verbatim: *"ATOC's
gating is binary, not top-k; no top-k attention in surveyed comm-MARL"*, and
OP-7 in the same file records score-based top-$k$ communication as this
project's own research proposal. The abstract's "a substantial line of work
proposes sparsifying it to the top-$k$ most relevant peers" was uncited and
contradicted by the authors' own notes; "a widely reproduced implementation
pattern applies top-$k$ as a mask" was supported only by this project's own bug
(`DECISIONS.md` D-019). The paper now says plainly that the argument examined is
one the authors made themselves and that the implementation errors reported are
their own. Related Work states that the multi-agent restriction mechanisms it
surveys are mostly *structural* and therefore exempt from Proposition 1 by the
paper's own Proposition 2 — which was previously an unnoticed hole in the
argument, since Table 1 indicted papers the theory does not bind.

**Related Work was restructured conceptually** into four subsections (multi-agent
communication / efficient and sparse attention / sparse routing and MoE /
measuring rather than counting), replacing a chronological list. A new MoE
subsection makes the ceiling sharper by contrast: score-based top-$k$ *does* pay
in mixture-of-experts because routing scores are cheap relative to the expert
FLOPs they gate, and the general principle — whether selection pays depends on
the ratio of scoring cost to skipped work, which is exactly 1 in single-head
attention — is now stated.

**The audit table was reframed from an indictment to a description.** Section
title "Who Measures, and Who Counts" → "Related Work and a Reporting Audit";
table title "Does the paper measure what it claims to save?" → "Type of evidence
offered for efficiency claims"; $n{=}6$ per group now stated in the caption;
CommNet recategorised to `n/a` since it makes no efficiency claim and should not
have been in a table about substantiating them; "the surrounding literature has
largely not been doing this" replaced with a statement scoped to the twelve
papers and explicitly describing *the authors' own evidence base*.

**A new cross-architecture subsection (§4.6)** replaces a single sentence that
was false. **A new §4.4 paragraph** withdraws the "intrinsic" claim using the
paper's own roofline. **The pitfalls section** became a table and gained two
entries (P6 allocator, P7 timing-group composition), and now corrects a belief
the authors held about FLOP counters rather than repeating it.

**Voice.** Removed: "does not survive contact with the hardware", "was never
going to help", "Two honest caveats", "the arm whose purpose is to reduce cost
is the first to run out of memory", "where we would spend the next unit of
effort", "we characterise it rather than asking for trust", "we validate the
derivation rather than asserting it", "we say so rather than implying", "the
most favourable case sparsity can construct for itself" (personification, used
3×), and the twice-bolded "The efficiency claim in the literature is a claim
about gathered". Zero hits remain for the flagged LLM-register phrases
(`moreover`/`furthermore` chains, "it is important to note", "underscores",
"paves the way", "plays a crucial role", "delves"). Em-dash density is ~1 per
150 words.

---

## 4. Scientific corrections

Thirteen statements were **false as written**. Each below gives the original,
the artifact inspected, and the resolution. Full detail with file:line in
`claim_evidence_audit.md`.

| # | Original claim | Evidence inspected | Resolution |
|---|---|---|---|
| 1 | "slower … in every one of $46$ cells … against a fused dense baseline" (conclusion) — contradicted §4.3's own "1 of 46" | `crossover_surface.json`: `wins_vs_sdpa = 1`, $d{=}128,B{=}4096,N{=}12$, gathered 0.35558 vs fused 0.41392 ms → **0.859** | Now "$0$ of $46$ vs unfused dense and $45$ of $46$ vs fused", stated separately everywhere |
| 2 | "Both paths sit far below both ridge points" (Table 2 caption) vs "the dense path reaches the ridge" (Figure 4 caption) — same page | `roofline.json`: CPU ridge **3.6755**, dense AI **3.981** (N=192), **4.182** (N=512). Closed form $N(4d{+}5)/(16(N{+}d)) \to 4.3125$, crosses at $N \approx 92$ | The **figure was right**. Table caption rewritten; the crossing now stated in abstract, §3, §4 and conclusion consistently |
| 3 | "*Every* drift moves in the direction that makes the sparse path look worse" | `drift_report.json`, per-cell signs: 16 ratio cells, **10 adverse / 6 favourable**; loaded 8/8 adverse, idle second device **6/8 favourable** | Split by condition; "contention cannot conceal a crossover" replaced with an explicit margin argument |
| 4 | "the smallest gathered/dense ratio observed anywhere — across all runs, devices, loads, precisions and execution modes — is $1.14\times$" | Cross-artifact scan: 1.137 is `microbenchmark_drift_cuda_1.json` N=6, **GPU-only, vs unfused only**. CPU N=48 = **0.810**; fused min = **0.923** | Scoped per baseline and per device; both counterexamples now reported |
| 5 | "On CPU the ordering also holds, up to $37.9\times$ the fused baseline" | Archived CPU series non-monotone (2.09, 2.06, 2.08, **0.81**, 5.14, 1.94). Diagnosed as a glibc page-fault storm: ~37k minor faults/forward at N=192, up to **8.8×** inflation, unequal across arms | Re-measured with the heap pinned: monotone **2.0–4.8×** vs unfused, **2.6–11.8×** vs fused. The 37.9× was ~3.2× page-fault time. New §4.6 + pitfall P6 |
| 6 | "median-to-minimum ratio stays within $1.02$–$1.06\times$" | `microbenchmark_comm.json`: true range **1.000–1.234**, 16 of 39 rows outside. Worst is unfused dense at N=96 (spread 21.6%) — a headline denominator | True range stated; full median-vs-min sensitivity for all eight headline ratios published in the appendix |
| 7 | "$56.5$ GiB at $N{=}384$" | `memory_analysis.json`: 56,556.5 **MiB** = **55.2 GiB** | Corrected |
| 8 | Two IQR cells in the dispersion table | N=12 masked **0.0041** (not 0.0032); N=12 gate **0.0101** (not 0.0075) — both borrowed from adjacent rows | Corrected |
| 9 | "in `bf16` … all four fused SDPA backends including FlashAttention are reachable" | `mappo.py:164-170` always passes a float mask, which excludes FLASH at every dtype; kernel profile shows `fmha_cutlassF_{f32,bf16}` | Now states FlashAttention served **no** arm, and that a FLASH baseline would widen the gap, not narrow it |
| 10 | P5: "the fused path is invisible to standard FLOP counters, which report zero for it" | torch 2.11 counter reports the fused path **correctly** (5167.4 MFLOP); the *harness* added $4BN^2d$ on top, publishing 9462 — **1.83× too large** | Pitfall rewritten as "instrument the counter, then check the counter" |
| 11 | "the loss is **not an engineering defect a better kernel would remove**. It is **intrinsic**" — contradicted §7 | `torch.topk` at N=512,B=256 moves ~0.47 GB in 3.897 ms = **~8% of the 1448.8 GB/s the paper itself measures**; the gather stage runs near bandwidth | Withdrawn. Replaced with the roofline-relative statement and an explicit projection showing the ordering survives a bandwidth-optimal select (~3.5 ms vs 1.99 unfused / 0.43 fused) |
| 12 | C4 explained by "the gate-variance failure mode reported for learned gates" | `red_po12_dense/seed_0/config.json` = `{"attn_mode":"dense","topk":0}` — **no gate in that arm** | Removed. Replaced with the finding below |
| 13 | "a trained $N$ sweep was projected at $995$ GPU-hours against a $200$-hour budget" | Derived at 10M steps × 10 seeds; at the actual 2M/5-seed protocol it is **99.5 h**. Whole 60-run study cost **31.9 GPU-h**; a properly powered C1 costs **37.8** | The paper now states the nulls are a budget *allocation* choice, not an unaffordable experiment |

### Unsupported generalisations, now scoped

- **"$46$ of $46$ cells, on GPU and CPU, in fp32 and bf16, eager, compiled and CUDA-graph"** conflated one GPU/fp32/eager surface (which excludes the $d{=}16$ used for the headline table) with four separate smaller sweeps. Now stated as one sweep, with robustness reported separately.
- **"That flatness is itself the finding"** — at $N{=}512$ the ratio falls monotonically in head width ($2.43 \to 2.46 \to 2.12 \to 1.71$) and the sweep ends at $d{=}256$ with the two most favourable remaining cells skipped for memory. Now: flat in batch, *not* flat in $d$, trend explicitly unresolved, no claim for $d > 256$.
- **"architecture-independent"** — now: the intensity *ratio* is a property of the algorithms, so the disadvantage should be expected wherever the gathered path stays below machine balance; but an argument constructed to be device-independent cannot demonstrate device-independence. Evidence stated as one GPU architecture + same-model twin + one CPU.
- **"a learned per-agent gate costs $\mathcal{O}(BN^3)$"** — rescoped to the evaluated broadcast implementation, with the note that an $\mathcal{O}(BN^2)$ cumulative-sum formulation computes the same function. Retitled §4.5 and moved conceptually next to the pitfalls.
- **"within $10\%$ of eager"** — true of gathered latency (1.9%), false of ratios (±27% at N=12). Now split.
- **"Correcting the implementation matters"** previously quoted only the small-$N$ range. The paper now reports that gathered is **slower than masked** at N=96 and N=192.

### Newly disclosed, previously absent

- **`torch.topk(sorted=True)`** costs the paper's own sparse path 5–24% (selection stage up to 35%), and `sorted=False` is exactly equivalent (verified to $5.4\times10^{-7}$, spread ≤2.2%). Ratios move 4.97→3.79 at N=192 and 3.61→3.36 at N=512; **ordering unchanged at every N**.
- **N=12 training arms do not beat a random policy** (random −758.9; dense −864.9 with two seeds below −975; gathered −717.9). C4 is now reported as uninterpretable for that reason, independent of its $p$-value.
- **Positive controls are not capacity-matched** (+7.2% to +19.0% parameters), so they establish the communicating architecture helps, not that the channel alone does.
- **`simple_reference` is not "logically unsolvable without the channel"** — the environment has its own discrete utterance channel (`Discrete(50)` = 5 moves × 10 symbols, partner utterance in obs dims 11–20). The control ablates the *learned attention* channel on top of it, and at N=2 its attention entropy is identically zero.
- **The CPU ridge is a cache-bandwidth figure** (32 MiB buffers vs 96 MiB L3); a DRAM-resident probe gives ~2.2. Used only where the conclusion holds under both.
- **Prop. 1 indexing** — exact counting gives $2(N{-}1)/((N{-}1)+k)$, strictly smaller than the stated bound, so the bound is valid but loose. Now stated once, pre-empting the reviewer question.
- **Small-N cells are launch-bound**, and two of the project's own sweeps disagree 5× on one such cell.
- **Bootstrap/permutation tension** now explains *why* the percentile bootstrap is anti-conservative at n=5 ($5^5 = 3125$ distinct resamples).
- **Power** — 45 seeds/arm at $\alpha=0.05$, but ≈70 at the family-corrected $\alpha = 0.05/6$.
- **Pre-registration** — "40" denominator defined (tier-2–4 subset), and stated that the 8 completed runs *were* the primary contrast, with the amendment confined to the deviations log.

### Figures

| Figure | Defect | Fix |
|---|---|---|
| fig2 | In-figure title read *"Sparsity never wins: every cell is >1"*, contradicting §4.3 one page later; denominator (unfused dense) named nowhere | Neutral title naming the denominator; colourbar labelled; skip-cell legend added (hatch was drawn white-on-white and would have been invisible) |
| fig3 | Totals and shares used **4 of 7 stages** — printed 5.81 ms at N=512 against an actual 7.19 ms, inflating the small-N selection share 2.2× | Regenerated over all seven gathered stages. Totals 0.12/0.15/0.26/0.88/3.32/**7.19**; top-k select 13.4%→54.2%, matching the caption |
| fig4 | Markers are `min(peak, BW·AI)` — the roof at analytic intensity, not measured throughput — under a caption leading with "measured machine balance"; N=12 label landed beside the N=192/512 cluster; ~4 empty y-decades | y-axis relabelled "attainable TFLOP/s at analytic intensity" + figure note; leader lines; y-limits tightened; CPU ridge crossing now visible and annotated |

fig1 and fig5 were left alone (both accurate and CVD-safe). Note `scripts/make_figures.py` and `make_fig1_schematic.py` are **stale** and would clobber figures with unrelated content.

### Bibliography

All 48 entries verified against publisher sources. 11 wrong years corrected
(4 cited: `singh2018ic3net`→2019, `jiang2018dgn`→2020, `terry2020pettingzoo`→2021,
`choromanski2021performers`→2021; three of these printed two different years in a
single reference). 12 cited titles had acronyms destroyed by `tmlr.bst`'s
lower-casing (`Flashattention`/`io-awareness`, `Tarmac`, `Pettingzoo`, `Big bird`,
`ppo`, `pid lagrangian`, `gumbel-softmax`, and `$O(n^2)$` being lower-cased to
`$o(n^2)$`) — all brace-protected. Venue strings normalised from 4 NeurIPS and 3
ICLR spellings to one each. DOIs added where the publisher mints one.
`beltagy2020longformer` correctly left as an arXiv preprint — it has no archival
venue, which matters because §2 leans on it as the measurement-discipline
exemplar.

**Warning:** `literature/references.bib` is generated by
`scripts/build_literature.py`; re-running it will clobber these fixes. The file
header now says so.

### Removed

`paper/tables/` (`table_headline.tex`, `table_ablation.tex`, `headline_stats.json`)
— never `\input` anywhere, from the abandoned "Adaptive TopK (ours)" version,
containing Welch t-tests, an `n=0` seeds row, and a duplicate
`\label{tab:headline}`.

---

## 5. Claim–evidence status

Every headline claim is now supported, scoped, or withdrawn:

| Claim | Status |
|---|---|
| $<2\times$ analytic ceiling under A1–A5 | **Supported.** Holds a fortiori under exact counting; verified numerically for $N \in \{6..512\}$ |
| Structural escape reaches $N/k$ | **Supported**, and explicitly limited to operation counts, not speedup |
| Masked ≡ gathered functionally, ≠ in executed work | **Supported.** $<10^{-10}$ in fp64; 2,555,904 vs 2,015,232 FLOPs; trace assertion. Regression-pinned |
| Gathered slower than unfused dense in 46/46 GPU cells | **Supported**, envelope named (GPU, fp32, eager, $d \in \{32..256\}$) |
| Gathered slower than fused in 45/46 | **Supported**, the one exception reported with its coordinates |
| $3.8\times$ traffic for $0.6\times$ arithmetic at N=512 | **Supported** |
| Selection is 15.8× the dense aggregation | **Supported as a measurement**; the inference that it is *intrinsic* is withdrawn |
| Both paths memory-bound | **Corrected** — true on GPU at all N; on CPU dense crosses the ridge for $N \gtrsim 96$ |
| CPU cross-architecture agreement | **Re-measured.** Monotone 2.0–4.8× vs unfused after controlling the allocator |
| $\mathcal{O}(BN^3)$ gate memory | **Supported for the evaluated implementation**, explicitly not for learned gates in general |
| Positive controls pass | **Supported**, with the capacity caveat and the env-channel caveat now stated |
| Six contrasts are under-powered nulls | **Supported.** All 30 table cells reproduce exactly; Holm arithmetic independently recomputed |
| Reproducibility on an idle device | **Supported** (ratios within 3.2%) |

---

## 6. TMLR readiness assessment

**Criterion 1 — Are the claims supported by accurate, convincing and clear evidence?**
**YES**, where it was BORDERLINE-to-NO before this revision.

The measurement core was always sound — an independent re-run reproduced the
headline table to within 6%, and every cell of the training table reproduces
exactly from per-seed artifacts. What failed criterion 1 was the *claims about*
the measurements: thirteen false statements, three of them contradicted by the
paper's own released artifacts, and a rhetorical target that the authors' own
literature review said did not exist. All are now either corrected, scoped, or
withdrawn, and the paper discloses two respects in which its own sparse
implementation is slower than necessary. The residual weakness is that the
headline campaign still times the `sorted=True` path; this is disclosed with a
measured bound, but a reviewer may reasonably ask for the re-run (see §8).

**Criterion 2 — Would some individuals in TMLR's audience be interested?**
**YES**, clearly and without qualification.

The masked/gathered/fused trichotomy, the demonstration that a FLOP model reports
a 91% saving for code executing the dense operation count, the $2\times$ ceiling
with its structural escape, and seven benchmarking pitfalls with measured
magnitudes are useful well beyond multi-agent RL. The pre-registered, honestly
reported null is a good citizen of the literature.

**Estimated editorial outcome: send to review.** Desk-rejection risk is low —
style file is byte-identical to the official one, anonymisation is clean
including PDF metadata, and there is no page limit to violate. I would expect a
reviewer to ask for the `sorted=False` re-run and to press on generality from
one GPU generation; neither is a desk-level concern.

---

## 7. Five questions a skeptical reviewer will still ask

1. **"You disclose that `sorted=False` makes your sparse path 5–24% faster, but your tables still use `sorted=True`. Why not re-run?"** The strongest answer is a re-run. The current answer — an explicit bound showing the ordering is unchanged at every N — is defensible but is the paper's most exposed point.
2. **"Your gathered path is eager PyTorch and your own selection kernel runs at 8% of achievable bandwidth. How is this evidence about the method rather than about `torch.topk`?"** The paper now answers with the bandwidth-optimal projection (~3.5 ms vs 1.99/0.43), but a reviewer may still want a fused-kernel traffic floor. See §8.
3. **"At $N{=}512$ the ratio falls monotonically in head width and your sweep stops at $d{=}256$ — exactly where a crossover would appear. What happens at $d = 512$?"** The paper states this is unresolved and claims nothing for $d > 256$. A reviewer may want the cells run.
4. **"Your own literature review says no surveyed comm-MARL method uses top-$k$ attention. Who is this paper arguing with?"** Now answered directly in the introduction — but a reviewer may ask whether the result is therefore of interest. The MoE contrast in §2.3 and the generality of Proposition 1 are the defence.
5. **"You had 168 of 200 GPU-hours left and a properly powered primary contrast costs 38. Why report six under-powered nulls instead?"** The paper now concedes this explicitly rather than blaming the budget.

---

## 8. Remaining experimental gaps

### Critical before submission

1. **Re-run the GPU campaign with `sorted=False`** on an idle device, and regenerate `tab:headline`, `tab:decomp`, the crossover surface and figs 2–3. This is the only item I regard as genuinely blocking, because the paper currently benchmarks a knowingly sub-optimal sparse path. Cost: minutes of compute; the blocker is device quiescence, not time. (Both GPUs held 51–62 GiB of other tenants' memory during this session, so I did not regenerate headline numbers under contention — the paper's own protocol forbids it.)
2. **Re-run the CPU sweep with the heap pinned** and archive it as the primary CPU artifact. §4.6 currently reports re-measured numbers taken under host load 4.5–6.5; two independent repeats agree on the ratios, but the archived campaign is superseded and should be replaced rather than annotated.

### Useful but not blocking

3. **An analytic traffic floor for a hypothetical fused selection+gather+contraction kernel.** Costs no compute and converts the paper's weakest claim into its strongest: such a kernel reads $Q,K,V$ and writes the output — the same $\mathcal{O}(BNd)$ traffic as fused dense — while still performing the $N^2d$ score matmul, which bounds how close any implementation can come.
4. **Extend the head-width sweep past $d{=}256$** at reduced batch, to resolve the trend in §4.3 rather than truncating it.
5. **A properly powered C1** (≈45 seeds/arm at $\alpha=0.05$, ≈70 under Holm; ~38 GPU-h). Would convert the primary contrast from an under-powered null into a result.
6. **A second GPU generation.** Acknowledged as unavailable; the CPU replication plus the intensity argument is the stated substitute.
7. **Regenerate `provenance.json` / `MASTER_LOG.csv`** to cover the 60 `red_*` runs. They currently describe 45 superseded runs on a 2080 Ti host the limitations section says is gone, so a reviewer checking provenance for the training study finds a different GPU.

### Not recommended

A cross-N trained sweep in the current PO environment — the difficulty confound
makes it uninterpretable until the radius is calibrated per $N$.

---

## 9. TMLR compliance

| Item | Status |
|---|---|
| TMLR stylefile preserved | **Yes** — `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, `math_commands.tex` byte-identical to the official `JmlrOrg/tmlr-style-file`. No margin, font, or spacing changes |
| Double-blind status | **Correct** — `\usepackage{tmlr}` without `[accepted]`/`[preprint]`; author block suppressed |
| Main-content pages | ~17.3 before references (TMLR sets no page limit) |
| Appendix location | After references, p19–22, as the template requires |
| Broader Impact reviewed | **Yes** — shortened; the duplicated scope sentences removed as disproportionate for a measurement study |
| Undefined references | 0 |
| Undefined citations | 0 |
| Compilation errors | 0 (also 0 overfull boxes, 0 BibTeX warnings) |
| Anonymisation leaks | **None** in `.tex`, PDF text, PDF raw streams, or metadata (`/Author`, `/Title`, `/Subject`, `/Keywords` all absent) |

**Build note:** `eso-pic.sty` is required by `tmlr.sty` and was missing — a fatal
error from a clean TeX tree. Installed via `tlmgr install eso-pic`; either vendor
it or document it in the README.

**Supplementary-bundle note:** exclude `paper/*.log`, `*.aux`, `*.out`, `*.blg` —
`main.log` carries absolute font paths under `/home/mhamdan/`.

---

## Suggested follow-up not applied

`scripts/check_paper_numbers.py` passed all 63 checks while every one of the
thirteen false statements above was live, because it checks scalars and never
quantifiers or cross-file minima. Extending it to assert (i) per-cell drift sign
counts, (ii) the global minimum ratio per baseline across *all* artifact files,
and (iii) the win counts against both baselines would prevent this class of
regression. `scripts/check_bib.py` is currently red on 25 uncited entries,
independent of this work.
