# Claim–Evidence Audit (Phase 2)

Consolidated from seven independent specialist audits against the raw artifacts in
`results/`, the implementation in `src/`, the generating scripts in `scripts/`, and the
git history. Every row was checked against an artifact; nothing here is inferred from
the manuscript alone.

Severity: **P0** = false or self-contradictory as shipped; **P1** = unsupported or
mis-scoped; **P2** = imprecise.

---

## A. Statements that are false as written

| # | Claim (location) | Artifact | What the artifact says | Sev |
|---|---|---|---|---|
| A1 | "slower … in every one of $46$ cells … against a fused dense baseline" (`08_conclusion.tex:11`) | `crossover_surface.json` | `wins_vs_sdpa = 1` at $d{=}128,B{=}4096,N{=}12$: gathered 0.35558 ms vs fused 0.41392 ms → **0.859**. §4.3 says "1 of 46" — the conclusion contradicts the body. | **P0** |
| A2 | "Both paths sit far below both ridge points" (`03_analysis.tex:151`, echoed `00_abstract.tex:28`, `01_intro.tex:49`, `08_conclusion.tex:15`, `results/roofline.md`) | `roofline.json` | CPU ridge **3.6755**; dense AI **3.981** ($N{=}192$), **4.182** ($N{=}512$). Closed form $\mathrm{AI}_{\text{dense}} = N(4d{+}5)/(16(N{+}d)) \to 4.3125$, crossing 3.68 at $N\approx92$. Dense is **compute-bound on CPU for $N \gtrsim 96$**. The figure caption says the opposite of the table caption, on the same page. | **P0** |
| A3 | "*Every* drift moves in the direction that makes the sparse path look worse" (`04_measurement.tex:223`) | `drift_report.json` | 16 ratio cells: **10 adverse, 6 favourable**. Loaded 8/8 adverse (+1.06%…+56.44%); idle second device **6/8 favourable** (−3.15%…−0.38%). Self-refuting: the 1.14× quoted in the next clause *is* a favourable drift (1.174 → 1.137). `scripts/drift_report.py:163` scopes it to the loaded condition; the manuscript widened it. | **P0** |
| A4 | "the smallest gathered/dense ratio observed anywhere — across all runs, devices, loads, precisions and execution modes — is $1.14\times$" (`04_measurement.tex:226`) | cross-artifact scan | 1.137 is the min over **GPU fp32, vs unfused dense only** (`microbenchmark_drift_cuda_1.json`, $N{=}6$). Falsified by `microbenchmark_cpu.json` $N{=}48$: gathered 1.4045 vs unfused dense 1.7338 → **0.810** (medians agree at 0.807; spreads ≤1.4%, so not a min artifact). Min vs fused = **0.859**. | **P0** |
| A5 | "finding no crossover anywhere" (`01_intro.tex:68`); "On CPU the ordering also holds" (`04_measurement.tex:70`) | as A4 | Two crossovers exist in released data. | **P0** |
| A6 | "Across the full sweep the median-to-minimum ratio stays within $1.02$–$1.06\times$" (`09_appendix.tex:38`) | `microbenchmark_comm.json` | True range **1.000–1.234**; 16 of 39 rows outside. Worst is unfused dense at $N{=}96$ (1.234, inter-run spread 21.6%) — the *denominator* of the headline $N{=}96$ ratio. On medians that cell moves 1.81× → 1.51×; every other $N$ moves <5.1%; all stay >1. | **P0** |
| A7 | "$56.5$ GiB at $N{=}384$" (`04_measurement.tex:186`) | `memory_analysis.json` | 56 556.5 **MiB** = **55.2 GiB**. Unit relabel. | **P0** |
| A8 | Dispersion table IQRs (`09_appendix.tex:31,33`) | `microbenchmark_comm.json` | $N{=}12$ masked IQR **0.0041** (not 0.0032); $N{=}12$ gate IQR **0.0101** (not 0.0075). Both wrong values are borrowed from adjacent rows. Other 18 cells verify exactly. | **P0** |
| A9 | "in `bf16` … all four fused SDPA backends including FlashAttention are reachable" (`04_measurement.tex:68`) | `src/agents/mappo.py:164-170` + kernel profile | `_forward_sdpa` always passes a **float mask**, which excludes the FLASH backend at every dtype. Profiling shows `fmha_cutlassF_{f32,bf16}_aligned_…_sm80` — mem-efficient CUTLASS in both precisions. FlashAttention served **no** arm. The backend probe at `microbenchmark_comm.py:251` omits the mask, which is why it reported otherwise. | **P0** |
| A10 | P5: "the fused path is invisible to standard FLOP counters, which report zero for it" (`06_pitfalls.tex:44`) | torch 2.11 counter | False. Raw counter gives **5167.4 MFLOP** (correct) for fused. The *harness* then adds $4BN^2d$ on top, publishing **9462 MFLOP, 1.83× too large**. The pitfall as written is wrong; the corrected version (a harness double-count) is a better pitfall. | **P0** |
| A11 | "the gate-variance failure mode reported for learned gates" explains C4 (`05_training.tex:105`) | `red_po12_dense/seed_0/config.json` | `{"attn_mode":"dense","topk":0}` — **the arm has no gate**. Dispersion is real (sd 157.1 vs 44.1) but the named mechanism is absent. | **P0** |
| A12 | "throughput differs by $<3\%$ across arms at every $N \ge 6$" (`05_training.tex:12`) | `throughput.json` | **3.02%** at $N{=}6$, **3.30%** at $N{=}12$. Also `measure_throughput.py:26` maps `"topk" → adaptive_topk`, so the **gathered arm was never in the probe**. | **P1** |
| A13 | "a trained $N$ sweep was projected at $995$ GPU-hours against a $200$-hour budget" (`05_training.tex:13`) | `compute_budget.md`, run `wall_time_s` | 995 h is derived at **10M steps × 10 seeds**. At the paper's actual 2M-step / 5-seed protocol the same sweep is **99.5 h — inside budget**. The whole 60-run study cost **31.9 GPU-h**; C1 at the 45 seeds/arm the limitations section names costs **37.8 GPU-h**. The power excuse does not survive its own arithmetic. | **P0** |

## B. Unsupported generalisations

| # | Claim | Evidence actually held | Sev |
|---|---|---|---|
| B1 | "a substantial line of work proposes sparsifying it to the top-$k$ most relevant peers" (`00_abstract.tex:2`); "$1-k/(N-1)$ … is usually reported" (`01_intro.tex:9`) | **`literature/literature_review.md:244-245` states verbatim: "ATOC's gating is binary, not top-k; no top-k attention in surveyed comm-MARL."** OP-7 shows score-based top-$k$ comm was *this project's own research proposal*. No citation exists for either claim. → reframed first-person. | **P0** |
| B2 | "a **widely reproduced** implementation pattern applies top-$k$ as a mask" (`00_abstract.tex:17`); "the masked variant **in common use**" | No citation, no repo audit, no count. The only documented instance is this project's own bug, recorded at `DECISIONS.md` D-019. | **P0** |
| B3 | "the loss is **not an engineering defect a better kernel would remove**. It is **intrinsic**" (`04_measurement.tex:160`) | Contradicted by `07_limitations.tex:52` ("could plausibly close part of the gap") **and** by measurement: at $N{=}512,B{=}256$ `torch.topk` moves ≈0.47 GB in 3.897 ms = **8% of the 1448.8 GB/s the paper itself measures**, while the gather stage runs at/above DRAM bandwidth. Selection is a *badly-optimised kernel*, not an intrinsic cost. Direction survives a bandwidth-optimal select (≈3.5 ms vs 1.99 unfused / 0.43 fused). | **P0** |
| B4 | "the result should be **architecture-independent**" (`03_analysis.tex:176`) | One GPU architecture, a same-model second device, one CPU, plus an argument that is device-independent *by construction* and therefore cannot itself be evidence of device-independence. | **P0** |
| B5 | "a learned per-agent gate costs $\mathcal{O}(BN^3)$" (`00_abstract.tex:33`) | True of **one broadcast-then-reduce formulation** (`mappo.py` adaptive branch materialising $(B,N,N{-}1,N)$). A cumulative sum along $K$ gives $\mathcal{O}(BN^2)$ for the same function. Not a property of learned gates. | **P1** |
| B6 | "Contention cannot be concealing a crossover" (`04_measurement.tex:227`) | An impossibility claim from **one** load pattern (this project's own training jobs). Correct form is a margin argument. | **P1** |
| B7 | "That flatness is itself the finding" (`04_measurement.tex:100`) | At $N{=}512$ the ratio falls **monotonically** in head width: $B{=}256$ gives $2.43 \to 2.46 \to 2.12 \to 1.71$. The sweep stops at $d{=}256$ with the two most favourable remaining cells skipped for memory. That is a trend toward crossover, truncated where it matters. | **P1** |
| B8 | "$46$ of $46$ cells, on GPU and CPU, in `fp32` and `bf16`, eager, compiled and CUDA-graph captured" (`00_abstract.tex:24`) | The 46 cells are **one** GPU/fp32/eager surface ($d\in\{32..256\}$, excluding the $d{=}16$ used for the headline table and roofline). CPU, bf16, compile and CUDA-graph are separate, smaller sweeps. | **P0** |
| B9 | "the ordering is unchanged under `torch.compile` and CUDA-graph capture (both within $10\%$ of eager)" | Holds for gathered **latency** (1.9%); **not** for ratios (±27% at $N{=}12$). | **P1** |
| B10 | "On CPU … up to $37.9\times$ the fused baseline, which is the cross-architecture evidence Section 3.3 predicts" | §3.3 predicts an **intensity ratio**. The sparse-vs-dense gap is *narrower* on CPU (1.94×) than GPU (3.74×). The 37.9× is entirely the **fused/unfused** ratio: 19.6× on CPU vs 1.38× on GPU — cache blocking (37.7 MB score matrix vs 96 MiB L3 across two sockets), not arithmetic intensity. The CPU dense point sits at **1.5% of both roofs**, so the roofline model does not describe it. | **P0** |

## C. Method / implementation defects affecting reported numbers

| # | Finding | Evidence | Sev |
|---|---|---|---|
| C1 | **`torch.topk(sorted=True)` is used on the op the paper says is 67% of the budget.** `sorted=False` is exactly equivalent (softmax∘gather is permutation-invariant) and strictly cheaper. | `src/agents/mappo.py:194`, `scripts/decompose_components.py:87`. Measured this session (cuda:1, 7 runs, interleaved, spread ≤2.2%, max abs deviation ≤5.4e-7): selection −1.7%…−35.4%; whole gathered arm −5.4%…−23.8%; ratio at $N{=}192$ 4.97 → 3.79, at $N{=}512$ 3.61 → 3.36. **Ordering unchanged at every $N$.** The published campaign benchmarks the pessimized path. | **P0** |
| C2 | **CPU ridge 3.68 is a cache measurement and does not reproduce.** `roofline.py:117` uses 32 MiB buffers → 64 MiB working set against a **96 MiB L3**. | Bandwidth sweep: 1125 GB/s at their buffer size, **331 GB/s** DRAM-resident. Re-running their unmodified code today gives ridge **2.15**, not 3.68 (42% off). GPU ridge reproduces (49.4 vs 53.0; bandwidth to 0.05%). CPU is 2× Xeon 6507P, 2 NUMA nodes — recorded in neither `HARDWARE.md` nor `provenance.json`. | **P0** |
| C3 | AI model's stated assumption ("each tensor read and written once") is false where applied at small $N$: GPU L2 is 128 MB, so $N{=}12/48$ traffic (1.4/12.6 MB) is fully cache-resident. Where it *does* hold ($N{=}192,512$) both paths attain 38–55% of achievable bandwidth and **the byte ratio predicts the latency ratio to within 6%** — the strongest available evidence, currently unreported. | derived from `roofline.json` + `microbenchmark_comm.json` | **P1** |
| C4 | Small-$N$ cells are launch-bound, not bandwidth-bound: dense is flat at ≈0.12 ms for $N{=}6..96$ while work grows 256×. The 1.17–1.23× ratios there reflect kernel count. Both drift runs use the same 5-arm group, so the drift control cannot detect this. | `microbenchmark_comm.json` vs `microbenchmark_execmodes.json` disagree **5×** on the same cell (0.1247 vs 0.0250 ms, eager/fp32/B=256/$N{=}12$) because group size (5 vs 15) changes how much launch latency falls inside the event window. | **P1** |
| C5 | Appendix FLOP counts (2 555 904 / 2 015 232, difference $2BNd(N{-}k)$) verify **exactly** — but only at `in_dim=32`, which the appendix never states; the sweep uses 64. | `09_appendix.tex:60-66` | **P1** |
| C6 | MAC/FLOP factor of 2 is mixed: §3.2 counts MACs, §4 calls FLOPs "multiply–accumulates", the appendix's $2BNd(N{-}k)$ is FLOPs. Ratios unaffected. | | **P2** |
| C7 | Proposition 1 indexing. Exact counting with self-masking gives $2(N{-}1)/((N{-}1)+k)$, **strictly smaller** than the stated $2N/(N+k)$ for all $1\le k\le N{-}1$ — so the stated bound is valid but not tight. The $N^2d$ convention is right for *executed* FLOPs (pinned by `tests/test_comm_module.py:384`), but it is a mixed convention and yields $2N/(2N{-}1)\ne1$ at $k=N{-}1$, where the repo's own test asserts exactly 1.0. | | **P1** |

## D. Training study

| # | Finding | Evidence | Sev |
|---|---|---|---|
| D1 | **The $N{=}12$ arms do not beat a uniform-random policy.** Random = **−755.32** (64 eps, same eval protocol); `red_po12_dense` = **−864.89** (3/5 seeds worse than −830); `red_po12_gather_k3` = −717.88. C4's "+147 favouring sparse" compares a diverged arm to a barely-learning one. `results/random_baseline.json` exists and is cited nowhere. | re-run under `train_mappo.evaluate` | **P0** |
| D2 | "`simple_reference` is logically unsolvable without the channel" is false. Action space is `Discrete(50)` = 5 moves × **10 communication symbols**, and obs dims 11–20 carry the partner's utterance — the "no-comm" baseline has a working environment-level channel. Also $N{=}2$ ⇒ one peer ⇒ logged attention entropy is 0.000 on all five seeds, so G1 validates the *message pathway*, not attention. | env stepping | **P0** |
| D3 | Positive controls are not capacity-matched: comm arm has **+13.9%** (N=6), **+19.0%** (reference), **+7.2%** (N=12) effective trainable parameters. | configs + param count | **P1** |
| D4 | Lagrangian mechanism overstated: $\lambda\to0$ by step 168k (**8% of training**) and stays 0; the gate collapses to $k\approx1.01$ unaided, *below* the 1.25 target, then oscillates. The dual is inactive for 92% of training. | run logs | **P1** |
| D5 | "8 of 40" is **numerically exact** and git-checkable (tier-1 12:18 → plan 12:56 → tier-2–4 19:11), but the denominator is undefined in the text (reader computes 60) and the 8 completed runs *are* the primary contrast (all 5 `gather_k2`, 3/5 `random_k2`). | git + mtimes | **P1** |
| D6 | "45 seeds per arm" is right for two-sided $\alpha{=}0.05$, 80% power, but assumptions are unstated and it is inconsistent with the paper's own Holm family: at $\alpha{=}0.05/6$ the requirement is **≈70 per arm** ($n{=}45$ gives power 0.56). | non-central $t$ | **P1** |
| D7 | Entropy sequence 0.35 → 0.12 → 0.00 sits on **different supports** (ceilings $\ln5$, $\ln2$, $\ln1$); the $k{=}1$ value is arithmetically forced. As fractions of ceiling: 22% → 17% → undefined. | `mappo.py:205` | **P1** |
| D8 | Percentile bootstrap at $n{=}5$ is **anti-conservative** (only $5^5=3125$ distinct resamples per arm) — that is *why* it excludes zero where the exact test does not. Verified not seed-fragile (20/20 seeds). Paper records the tension but never explains its direction. | | **P2** |
| D9 | `provenance.json` (`n_runs: 45`) and `MASTER_LOG.csv` contain **zero** `red_*` runs — they cover only superseded Phase-A/B runs on a 2080 Ti host the limitations section says is gone. A reviewer checking provenance for the training study finds a different GPU. | | **P1** |
| D10 | DGN quote elides "requires high bandwidth and", converting a joint bandwidth+compute claim into a pure-compute one — the exact distinction the Scope section rests on. DGN's spatial neighbourhood is also *structural*, hence exempt by the paper's own Proposition 2. | DGN source | **P0** |

## E. Figures and floats

| # | Finding | Sev |
|---|---|---|
| E1 | `fig2_crossover`'s **in-figure title** reads "Sparsity never wins: every cell is >1" — contradicted by §4.3 three lines below, and the heatmap's denominator (unfused dense) appears in neither title, colourbar nor caption. | **P0** |
| E2 | `fig3_decomposition`'s "total sparse-path latency (ms)" uses **4 of 7 stages** (`qkv`, `scores`, `proj` excluded): printed 5.81 ms at $N{=}512$ vs the module's actual 7.19 ms. The caption's "selection grows from 30% to 67%" is 30→67% *of four stages* but **13→54%** of the real forward pass. Needs regeneration, not a caption fix. | **P0** |
| E3 | `fig4_roofline` markers are `min(peak, bw·AI)` — the roof evaluated at analytic AI, **not measured throughput** — while the caption leads with "measured machine balance". Plus four placement defects (CPU panel's $N{=}12$ label lands beside the $N{=}192/512$ cluster). | **P0** |
| E4 | Six floats never referenced from prose (Figs 1, 3, 5; Tables 5, 6, and appendix 7, 8). Fig 3 gets a full paragraph without being named. Table 5 (drift) is typeset inside §4.5 (memory). | **P1** |
| E5 | `C1$^\star$` is a dangling marker defined nowhere; Table 7 shows 2 of 8 $N$ without saying so; Table 4's `overhead` column cannot be derived from its printed columns. | **P1** |
| E6 | `\Mask`/`\Gath` print as **bold lowercase** in the page-4 description list but small caps elsewhere. | **P2** |
| E7 | `scripts/make_fig1_schematic.py` would silently clobber Fig 1, and `make_figures.py` would clobber Figs 2–4 with unrelated content. | **P1** |

## F. LaTeX / bibliography / compliance

| # | Finding | Sev |
|---|---|---|
| F1 | `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, `math_commands.tex` are **byte-identical** to the current official `JmlrOrg/tmlr-style-file`. Nothing altered for space. Anonymous mode correct; Broader Impact and appendix placement match the template. **TMLR has no page limit.** | PASS |
| F2 | `eso-pic.sty` was missing — a **fatal build failure from a clean TeX tree**. | **P0** |
| F3 | `paper/tables/{table_headline,table_ablation}.tex` + `headline_stats.json` are orphans (never `\input`), present an "Adaptive TopK (ours)" method with $n{=}0$–3 seeds contradicting the manuscript, and re-declare `\label{tab:headline}`. | **P0** |
| F4 | Four cited entries carry arXiv years instead of proceedings years: `singh2018ic3net`→2019, `jiang2018dgn`→2020, `terry2020pettingzoo`→2021, `choromanski2021performers`→2021 (the last two print two different years in one reference). Eight cited entries print mangled acronyms (`Flashattention`, `Tarmac`, `Pettingzoo`, `Big bird`, `ppo`, `pid lagrangian`, `gumbel-softmax`, `$o(n^2)$`). Four spellings of NeurIPS, three of ICLR. All 48 entries verified against publisher sources; none UNVERIFIED. **Longformer genuinely has no archival venue** — arXiv is correct there. | **P1** |
| F5 | Anonymisation clean: zero leaks in `.tex`, PDF text, raw streams, or metadata. (Housekeeping: `main.log` carries absolute font paths — exclude `*.log/*.aux/*.out/*.blg` from any supplementary bundle.) | PASS |
| F6 | Build: 0 errors, **0 overfull boxes**, 0 undefined refs/citations, 0 BibTeX warnings. One visible underfull hbox at `05_training.tex:81`. | PASS |

---

## Claims that survive the audit intact

These are the load-bearing results, and all of them verify:

- **Proposition 1** holds under the stated accounting (and holds *a fortiori* under exact
  counting).
- **Masked ≡ gathered functionally, ≠ in executed work**: max deviation $<10^{-10}$ in
  fp64; FLOP counter 2 555 904 vs 2 015 232; trace assertion that the sparse path issues
  no $(B,N,N)\times(B,N,d)$ product. All pinned by regression tests.
- **The headline ordering.** An independent re-run of the arms reproduces `tab:headline`
  to within 6% (dense $N{=}192$: 0.193 vs 0.206 ms; gathered 0.787 vs 0.771 ms), and it
  survives the `sorted=False` correction at every $N$.
- **The aggregation saving is negative** and **selection dominates** — both reproduce,
  though the *explanation* needed correcting (C1, B3).
- **Every cell of the training table** reproduces exactly from per-seed artifacts; Holm
  arithmetic correct including the step-down; power floor $2/252$ correct; positive
  controls at the design floor with confirmed perfect separation; $\ln 5$ is the right
  entropy ceiling and self-masking is confirmed in source.
- **Pre-registration chronology** is git-verifiable, and the post-hoc amendment touches
  only the deviations log (§§1–6 byte-identical).
- **Drift magnitudes** in `tab:drift` are correct; only the directional gloss was false.
- **PO environment characterisation**: fraction × $(N{-}1)$ = absolute count checks out;
  the masking rule is exactly what the code does.
- **C1 is genuinely FLOP-matched** — `_forward_random_k` does compute the full $N\times N$
  score matmul.
- **Longformer's three-implementation claim** is accurate as described.
