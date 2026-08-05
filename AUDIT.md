# AUDIT — TopK-TarMAC

Ground-truth audit of the repository as of commit `f7d89bf` (branch `phase-a/audit`),
performed 2026-08-05. **No code was changed and no training was run during this phase.**

Every finding below was verified independently against files on disk or against the
primary source. Findings that the task brief listed as "suspected" but that are *not*
present in the current tree are recorded as such, with evidence.

---

## A1. Repo inventory

### A1.1 Module tree

#### `src/` — 690 lines total

| Path | Lines | Purpose |
|---|---|---|
| `src/agents/mappo.py` | 217 | `MAPPOConfig`, `SharedActor` (param-shared MLP policy), `CentralisedCritic` (joint-obs MLP), `AttentionComm` (the whole method: dense / fixed top-$k$ / adaptive Gumbel top-$k$), `MAPPOAgent` wrapper. |
| `src/training/train_mappo.py` | 328 | `TrainerConfig`, serial-Python `VecEnv`, greedy `evaluate()`, the PPO `train()` loop. Writes `metrics.csv`, `config.json`, `final_eval.json`, `final.pt`. |
| `src/training/rollout.py` | 39 | `RolloutBatch` dataclass + `compute_gae`. |
| `src/envs/mpe.py` | 76 | `MPESimpleSpread` — agent-stacked `(N, obs_dim)` wrapper over `mpe2.simple_spread_v3`. **The only environment in the repo.** |
| `src/utils/seeding.py` | 30 | `set_global_seed`. |
| `src/eval/__init__.py` | 0 | **Empty package. No evaluation module exists**; eval logic lives inside `train_mappo.py`. |

#### `scripts/`

| Script | Purpose | Runs today? |
|---|---|---|
| `train.py` | Single run from a YAML config. | Yes |
| `run_seeds.py` | Launch N seeds, optionally across GPUs. | Yes |
| `measure_flops.py` | Analytic message-gather / score op counts + $k_\text{eff}$ from checkpoints → `results/flops_summary.json`. | Yes |
| `measure_random.py` | Uniform-action floor → `results/random_baseline.json`. | Yes |
| `make_tables.py` | `table_headline.tex`, `table_ablation.tex`, `headline_stats.json`. | Yes |
| `make_figures.py` | Figures 2–4. | Yes (**but emits raw LaTeX into a matplotlib title — see D-09**) |
| `make_fig1_schematic.py` | Architecture schematic. | Yes |
| `build_literature.py` | Rebuilds `literature/*` from structured data. | Yes |
| `reproduce.sh` | End-to-end regeneration. | **Runs, but does not reproduce — see D-15.** No `--dry-run` flag exists. |

#### `configs/` — 11 YAML files, all MPE `simple_spread`, $N \in \{3,6,12\}$ × {baseline, dense, adaptive topk} + 2 unrun fixed-$k$ configs.

#### `tests/` — `test_smoke.py`, `test_mappo.py`, `test_envs.py`. 13 tests.

#### `results/` — 45 run directories (`metrics.csv`, `config.json`, `final_eval.json`, `final.pt`), `MASTER_LOG.csv`, `flops_summary.json`, `random_baseline.json`, `REPRODUCTION.md`.

#### `paper/` — `main.tex` + 7 section files, 5 figure PDFs, 2 tables. **Entirely untracked by git (see D-16).**

### A1.2 What actually runs

```
$ .venv/bin/python -m pytest -q
13 passed in 97.77s
```

`bash scripts/reproduce.sh --dry-run` — **the flag does not exist**; `reproduce.sh` has no
argument parsing at all and `set -u` means the script ignores extra argv. Adding a real
dry-run mode is Phase B work.

### A1.3 Environment

Captured to `HARDWARE.md` and `configs/environment.lock` (101 pinned packages,
sha256 `a3d58f4f…02dd1`).

| | Value |
|---|---|
| Python | 3.11.14 |
| torch | 2.5.1+cu124 |
| numpy / scipy / pandas | 2.1.3 / 1.17.1 / 3.0.3 |
| pettingzoo / mpe2 / gymnasium | 1.26.1 / 1.1.0 / 1.0.0 |
| GPUs | 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition, 97 887 MiB each, driver 610.43.02, **compute capability 12.0 (sm_120)** |
| CPU | 2 × Intel Xeon 6507P, 32 logical cores |
| RAM | 503 GiB |

**D-00 (BLOCKER, new — not on the brief's list).** The pinned PyTorch build cannot
execute on this hardware:

```
NVIDIA RTX PRO 6000 Blackwell ... with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
The current PyTorch install supports sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90
>>> torch.randn(512,512,device="cuda") @ x
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

`torch.cuda.is_available()` returns `True`, so `train_mappo.py:155`
(`device = trainer_cfg.device if torch.cuda.is_available() else "cpu"`) will select
`cuda` and then crash on the first kernel launch. **No GPU experiment in Phase C can run
until torch is upgraded to a cu128 build (≥ 2.7).** That upgrade also invalidates
bit-exact reproduction of the existing 45 runs, which were produced on 2080 Ti (sm_75).

---

## A2. Claim-to-artifact traceability matrix

Claim IDs prefixed `C-`; defects prefixed `D-`. "Reproducible" = the exact number can be
regenerated from a file under `results/` by a script in `scripts/`.

### Abstract — `paper/sections/00_abstract.tex`

| # | Claim (truncated) | file:line | Source artifact | Repro | Notes |
|---|---|---|---|---|---|
| C-01 | "A five-seed evaluation … at $N \in \{3,6,12\}$" | `00_abstract.tex:10-11` | `headline_stats.json` | **N** | **D-01.** $N{=}12$ has **3** seeds, not 5. `01_intro.tex:31` correctly says "5 / 5 / 3". The abstract overstates. |
| C-02 | "$d{=}1.06$, bootstrap CI $[+2.3,+26.5]$" | `00_abstract.tex:13-14` | `headline_stats.json["6"]` = 1.0636, [2.295, 26.516] | Y | — |
| C-03 | "large improvement over the MAPPO baseline" | `00_abstract.tex:12-13` | Welch $p=0.159$ | **N** | **D-08.** See protocol conflict below. |
| C-04 | "at $N{=}12$ underperforms MAPPO with $d{=}-1.87$" | `00_abstract.tex:15-16` | `headline_stats.json["12"]["topk_cohens_d"]` = −1.867 | Y | — |
| C-05 | "std grows from $4.6$ at the MAPPO baseline to $40.2$ at $N{=}12$" | `00_abstract.tex:16-18` | 4.566 (MAPPO@N=12), 40.19 (topk@N=12) | **N** | **D-02.** Both numbers are correct but they are *different conditions at the same $N$*, not growth over $N$. The sentence reads as a scaling trend; it is a cross-arm contrast at one $N$. |

### Introduction — `01_intro.tex`

| # | Claim | file:line | Artifact | Repro | Notes |
|---|---|---|---|---|---|
| C-06 | "5 / 5 / 3 seeds" | `01_intro.tex:31` | `headline_stats.json` `*_n` | Y | Correct; contradicts C-01. |
| C-07 | "10 000-resample percentile bootstrap CI" | `01_intro.tex:32-33` | `make_tables.py:bootstrap_ci(n=10000)` | Y | — |
| C-08 | "gate variance scales with $N$ faster than the FLOPs the gate saves" | `01_intro.tex:35-36` | none | **N** | **D-03.** Gate variance is *never measured*. Only between-seed return std exists. No gate-entropy, $k_\text{eff}$ trajectory, or gradient-variance instrumentation exists anywhere in `src/`. This is the paper's central mechanistic claim and it has zero supporting artifact. |

### Method — `03_method.tex`

| # | Claim | file:line | Artifact | Repro | Notes |
|---|---|---|---|---|---|
| C-09 | Complexity table (score $N^2d$, aggregation $Nkd$, gate $N^2 d_g$) | `03_method.tex:83-92` | analytic only | Y (analytic) | Consistent with `AttentionComm.forward`. Not empirically validated — no latency measurement exists. |
| C-10 | "Driving all $p_i$ to one-hot at $N{-}1$ recovers exact dense attention" | `03_method.tex:58-59` | none | **N** | **D-04.** Asserted, never tested. Plausible by inspection of `mappo.py:141-146` but unverified. |
| C-11 | "zeroing the gate output recovers MAPPO" | `03_method.tex:59-60` | none | **N** | **D-04.** As implemented this is *false as stated*: with `use_comm=True` the actor input dim is `obs_dim + msg_dim` and `fc1` is a different tensor from the baseline's `fc1` (`mappo.py:55`). Zeroing $m_i$ recovers *a* MAPPO-shaped policy, not the trained baseline's logits. The claim needs restating as a weight-matched equivalence. |

### Experiments — `04_experiments.tex`

| # | Claim | file:line | Artifact | Repro | Notes |
|---|---|---|---|---|---|
| C-12 | Hyperparameters: lr 7e-4, 32 envs, rollout 25, 10 epochs, 1 minibatch | `04:17-22` | `results/*/seed_*/config.json` (all 45 agree) | Y | — |
| C-13 | "Four RTX 2080 Ti GPUs (11 GB each), ~4–6 GB free" | `04:24-25` | `HARDWARE.md` (old) | **N** | **D-05.** Hardware no longer exists. Every sentence appealing to it is now false. |
| C-14 | "Total compute … $\sim$60 GPU-hours" | `04:27-28` | `MASTER_LOG.csv` sum = **60.93** | Y* | **D-06.** Arithmetically traceable, but the `gpu_hours` column is *exactly* `wall_time_s / 3600` for all 44 rows — it is wall-clock, not GPU-time, and the runs were CPU-bound (the paper itself says so at `05_limitations.tex:21`). Mislabelled. `STATUS.md` separately claims "~20 GPU-h", contradicting both. |
| C-15 | "Five seeds at $N \in \{3,6\}$ and three at $N{=}12$" | `04:26-27` | `headline_stats.json` | Y | — |
| C-16 | "a claim of improvement requires the bootstrap CI to exclude zero" | `04:35-36` | — | **N** | **D-08.** The body states a **CI rule**; `table_headline.tex` caption states "Bold = significantly better … Welch's $t$-test, $p<0.05$". Two incompatible decision rules in one paper. Under the caption's rule the headline ($p=0.159$) fails; under the body's rule it passes. Also: **no cell in Table 1 is bolded**, so the caption's legend is dead text. |
| C-17 | "At $N{=}3$ … within one std, $p \gtrsim 0.4$" | `04:47-49` | $p$ = 0.511, 0.447 | Y | — |
| C-18 | "$N{=}6$, Dense mean delta $+12.1$" | `04:51` | −212.911 − (−225.038) = +12.13 | Y | — |
| C-19 | "std of Adaptive TopK (20.0 vs 3.7 for Dense)" | `04:56` | 20.014 / 3.725 | Y | — |
| C-20 | "$N{=}12$ … mean delta $-13.5$, $d{=}-0.89$, CI $[-34.2,+5.0]$" | `04:63-64` | `headline_stats.json["12"]` | Y | — |
| C-21 | "Adaptive TopK is **significantly** worse than MAPPO … delta $-65.4$, CI $[-119.7,-28.5]$" | `04:64-66` | CI ✓; Welch $p = 0.146$ | **N** | **D-07.** The numbers are traceable; the word "significantly" is not licensed under the caption's $p<0.05$ rule. Same problem at `06_conclusion.tex:10`. |
| C-22 | "std $40.2$ vs $4.6$ — a ninefold inflation" | `04:66-67` | 40.19 / 4.566 = **8.80** | Y* | "Ninefold" rounds 8.8 up. Defensible but loose. |
| C-23 | "$1 - 7.83/11 \approx 29\%$" | `04:67-68` | `flops_summary.json` k_eff 7.8277 → 0.7116 | Y | — |
| C-24 | Fig. 2 caption: "Adaptive TopK and Dense Attn-Comm **both improve over** MAPPO" | `04:76-77` | body says TopK is "tied with MAPPO" (`04:54`) | **N** | **D-09. CONFIRMED** (brief's #7). Caption contradicts body. |
| C-25 | ¶Scaling: "The gap between attention-comm methods and the baseline **widens with $N$**" | `04:83-84` | Table 1 | **N** | **D-10. CONFIRMED** (brief's #5). Deltas vs MAPPO: $N{=}3$ dense **−1.34**; $N{=}6$ dense **+12.13**; $N{=}12$ dense **−13.46**, topk **−65.42**. The gap does not widen — it is non-monotone and reverses sign. Directly contradicted by the paper's own table. |
| C-26 | "$k_\text{eff}/(N{-}1) \approx 0.79$ … 21 % reduction" | `04:98-101` | `flops_summary.json` 3.925/5 = 0.785 | Y | — |
| C-27 | Fig. 4 caption: "Adaptive TopK is FLOPs-Pareto-optimal **in the relevant operating regime**" | `04:109-110` | none | **N** | **D-11. PARTIALLY CONFIRMED** (brief's #6). The brief quoted "dominates dense"; the text on disk is milder. But "the relevant operating regime" is undefined and unsupported by any artifact, and the body (`04:101-103`) says *neither* method dominates. Caption still asserts more than the data. |
| C-28 | ¶Ablations: "The fixed-$k$ variants are **within seed variance of** the adaptive one" | `04:115-117` | `table_ablation.tex` rows show `--` and `n = 0` | **N** | **D-12. CONFIRMED** (brief's #3). **A quantitative claim made from zero data.** The two fixed-$k$ configs exist (`configs/mappo_mpe_n6_topk_fixed_k{2,4}.yaml`) but were never run and are not in `reproduce.sh`. |
| C-29 | ¶Ablations: "per-agent adaptivity adds little at $N{=}6$" | `04:117-118` | same empty cells | **N** | **D-12.** Rests entirely on the $n{=}0$ cells. |
| C-30 | ¶Ablations: "the ingredient that breaks at $N{=}12$" | `04:118-119` | `headline_stats.json["12"]` | Y | **Brief's #4 NOT CONFIRMED.** The brief expected an assertion that adaptivity "pays off as $N$ grows"; the text on disk asserts the *opposite* (that it breaks at $N{=}12$), which is what the data show. No defect here. |
| C-31 | "reduce-to-baseline switch … recovers MAPPO logits exactly" | `04:119-120` | none | **N** | **D-04.** Untested. See C-11. |
| C-32 | "EPyMARL MAPPO value of $-133.54 \pm 3.08$ at $N{=}3$" | `04:132-133` | **VERIFIED against arXiv:2006.07869**, Table (MPE Spread row, MAPPO column) | Y | Correct. |
| C-33 | "EPyMARL trains for 20 M steps versus 0.8–1.5 M here" | `04:130-131` | **VERIFIED**: "In MPE and LBF we train on-policy algorithms for 20 million timesteps" | Y | Correct. |
| C-34 | "EPyMARL uses PettingZoo `simple_spread_v2`" | `04:127-129` | EPyMARL §3.2 cites **Mordatch & Abbeel 2017** (original MPE), not PettingZoo | **N** | **D-13 (new).** Probable misattribution of another paper's environment. |
| C-35 | "TarMAC … reporting a **15-percentage-point gain** in traffic-junction success rate" | `04:137-138` | **REFUTED against arXiv:1810.11187 Table 3** | **N** | **D-14 (new, serious).** TarMAC Table 3 (hard): no-comm 74.1 ± 3.9, CommNet 78.9 ± 3.4, TarMAC 1-round 84.6 ± 3.2, TarMAC 2-round 97.1 ± 1.6. Gains available: **+23.0 pp** (2-round vs no-comm), **+18.2 pp** (2-round vs CommNet), **+5.7 pp** (1-round vs CommNet). **No pairing yields 15 pp.** An unverifiable number attributed to another paper. |
| C-36 | "…and a **32 % reduction in time-to-catch** in predator-prey" | `04:138-139` | **VERIFIED against arXiv:1810.11187 Table 5** | Y | 10-agent 20×20: IC3Net 52.4 ± 3.4 → IC3Net+TarMAC 2-round 35.57 ± 3.96 = **32.1 %**. Correct, but the paper must state the configuration (10 agents, 2-round) — the number does not hold for the other two configurations. |

### Limitations — `05_limitations.tex`

| # | Claim | file:line | Artifact | Repro | Notes |
|---|---|---|---|---|---|
| C-37 | "exceed the 48 GPU-hour budget **per agent count**" | `05:6-7` | `DECISIONS.md` D-001 says 48 GPU-h **total** | **N** | **D-17 (new).** Internal contradiction; also obsolete hardware. |
| C-38 | "The 2080 Ti rollout regime is CPU-bound" | `05:21-22` | no measurement | **N** | **D-05.** Plausible but unmeasured, and the hardware is gone. |

### Conclusion — `06_conclusion.tex`

| # | Claim | file:line | Artifact | Repro | Notes |
|---|---|---|---|---|---|
| C-39 | "between-seed std of 20–40 for adaptive-$k$ versus **$\sim$5** for dense and baseline" | `06:11-12` | dense: 3.53 ($N$=3), **3.72** ($N$=6), **16.85** ($N$=12); baseline: 1.37, **13.93**, 4.57 | **N** | **D-18 (new).** "$\sim$5" is true for 3 of 6 cells and badly wrong for dense@$N{=}12$ (16.85) and baseline@$N{=}6$ (13.93). Cherry-picked. |
| C-40 | "overwhelms the 21–29 % FLOPs reduction" | `06:9` | `flops_summary.json` | Y | — |

### Defects found in the brief's list that are **NOT** present on this branch

| Brief # | Status | Evidence |
|---|---|---|
| 1 — `X.X` GPU-hours placeholder | **Already fixed** | `04:27-28` reads "$\sim 60$ GPU-hours"; matches `MASTER_LOG.csv` sum 60.93. |
| 2 — `$N \in \{N_LIST\}$` unexpanded | **Already fixed** | `make_tables.py:141` does `tail.replace("N_LIST", …)`; `table_headline.tex` contains the literal `\{3, 6, 12\}`. |
| 4 — "adaptivity pays off as $N$ grows" | **Not present** | Text asserts the opposite (C-30). |
| 10 — "honest/honestly", "zero fabricated citations", "every number is regenerable" | **Already removed from `paper/`** | `grep -rn -i 'honest\|fabricat\|regenerable\|tune away' paper/` returns only `main.tex:4` ("reproducible across machines", a LaTeX build comment). This branch is named `prose-rewrite` for that reason. **Still present in `README.md:120`** — acceptable per the brief (README is the right home). |
| 13 — compute-matching asserted in `REVIEW_CHECKLIST.md` | **File does not exist** | Deleted in commit `6f796c2`. It was `paper/REVIEW_CHECKLIST.md` and `paper/` is now gitignored. Must be regenerated in Phase E. |

### Additional defects not on the brief's list

| ID | Defect | Evidence |
|---|---|---|
| **D-00** | **PyTorch cannot run on the available GPUs (sm_120).** | §A1.3. Blocks all of Phase C. |
| **D-15** | **`reproduce.sh` does not reproduce the published numbers.** | It reads `total_steps` from the YAML (`reproduce.sh:34`). The YAMLs say $N{=}3$: **1 500 000**, $N{=}6$: **1 000 000**. Every run's `config.json` and every `MASTER_LOG` row says **800 000**. Re-running would train 1.9× / 1.25× longer and produce different numbers. It also never runs the two fixed-$k$ configs, which is *why* Table 2 has $n{=}0$. |
| **D-16** | **The entire manuscript is untracked.** | `git ls-files paper` → empty. `.gitignore:37` ignores `paper/`. `prose-rewrite` is content-identical to `main` (`git diff main..HEAD` empty). No manuscript revision is under version control. |
| **D-19** | **`MASTER_LOG.csv` is incomplete and missing required columns.** | 44 rows for 45 on-disk runs — `mappo_mpe_n3_baseline/seed_0` exists on disk but is **not logged** ("a run that is not in the log does not exist"). Columns present: timestamp, git_sha, config, run_name, seed, total_steps, final_eval_mean/std, wall_time_s, gpu_hours. **Missing per the ground rules: method, env, N, peak VRAM, hardware string, config hash, output path.** |
| **D-20** | **Inconsistent std convention across artifacts.** | `make_tables.py:fmt_mean_std` uses population std (`np.std`, ddof=0) while `cohens_d` uses sample std (ddof=1). `results/REPRODUCTION.md` reports the $N{=}3$ baseline as "−58.93 ± **1.51**" while `headline_stats.json` says **1.365** — the same data under two conventions, presented as if authoritative in both places. |
| **D-21** | **Two mutually inconsistent random-policy floors.** | `results/REPRODUCTION.md` says "−77.76 ± 24.16 (n=40)"; `results/random_baseline.json` says −74.05 ± 18.90 (n=40); `table_headline.tex` uses −74.05. `REPRODUCTION.md`'s number is traceable to no artifact and its derived claim ("+24.2 % over random") is therefore also untraceable. |
| **D-22** | **`src/eval/` is an empty package**, but `REPRODUCTION.md` and the paper both describe an evaluation protocol as if it were a component. |

---

## A3. Scientific-framing audit
<a id="framing"></a>

### A3.1 Can `simple_spread` test the hypothesis at all?

**No — not as the sole environment.** This is the most serious problem in the project,
and the paper concedes the premise while not drawing the conclusion. At
`04_experiments.tex:139-140` it states that "`simple_spread` is solvable by purely local
policies". EPyMARL says the same thing more bluntly (§3.2: "All tasks but
Speaker-Listener … are **fully observable**").

The consequence is structural, not a matter of degree. The paper's thesis is:

> *the attention communication channel can be sparsified from $N{-}1$ to $k$ destinations
> without losing return.*

On a fully-observable, locally-solvable task the null hypothesis and the alternative make
**the same prediction**. If communication carries no task-relevant information, then
removing $(N{-}1)-k$ of the messages costs nothing — and a "successful" sparsification
result is indistinguishable from the channel having been useless all along. The strongest
possible positive result on `simple_spread` is therefore uninformative about the claim,
and the observed negative result at $N{=}12$ is equally consistent with "the gate is
broken" and "the gate is injecting noise into a policy that needed no channel". The
study as constituted **cannot separate those two explanations**, and the $N{=}12$
degradation is additionally confounded with undertraining (0.8–1.5 M steps against
EPyMARL's 20 M on the same task family — a 13–25× shortfall, C-33).

The one piece of evidence in the repo that the channel does *something* — dense beats
MAPPO by +12.1 at $N{=}6$ — is exactly the result whose statistical support is contested
(D-08), sits at $n{=}5$, and does not replicate in sign at $N{=}3$ or $N{=}12$.

**Therefore: at least one communication-critical environment is a precondition for the
paper, not an enhancement.** `simple_reference_v3` / `simple_speaker_listener_v4` are the
cheapest correct choice — they are in the same `mpe2` package already pinned, require no
new dependency, and are *designed* so that the task is unsolvable without the channel.
They function as a positive control: if the dense channel does not beat MAPPO there, the
communication module is misimplemented, and no sparsification result from this codebase
means anything. That control has never been run.

### A3.2 Does the current contribution set clear a journal bar?

**No.** Stated plainly, what a reviewer receives today is:

1. **One environment**, which the paper itself says does not require the capability under study.
2. **One negative result** (adaptive gating fails at $N{=}12$) at $n{=}3$ seeds, with $p=0.146$ — described as "significant" (D-07).
3. **One contested positive result** (dense at $N{=}6$) at $n{=}5$, $p=0.159$, resting on a percentile bootstrap that is known to be anti-conservative at $n{=}5$, and reported under a protocol the paper's own table caption contradicts (D-08).
4. **$N \le 12$** — below any regime where $\mathcal{O}(N^2) \to \mathcal{O}(Nk)$ matters.
5. **The efficiency claim that motivates the entire method is never measured.** There is no latency number, no throughput number, no VRAM number anywhere in the repo. `flops_summary.json` contains an *analytic* op count, and the aggregation-FLOPs "saving" of 21–29 % is a ratio of two arithmetic expressions, not an observation. A top-$k$ gather is very plausibly *slower* than a dense matmul at $N \le 12$.
6. **The central mechanistic claim — "gate-variance dominance" — is never measured** (D-03/C-08). It is inferred from between-seed return std, which is a downstream consequence of everything in the training loop, not a gate property.
7. **An ablation table with two empty cells and a paragraph of prose describing what they show** (D-12).
8. **The baseline is not the method it is named after** (§A4).

Items 5, 6 and 8 are each independently sufficient for rejection at a Transactions venue.
This is currently a workshop-grade negative result with a title that promises a method.

### A3.3 The three highest-value additions per unit of compute

Ranked by (defects retired) × (framing upgrade) ÷ (GPU-hours).

**1. A communication-critical environment + the random-$k$ control, run together.**
*Retires:* the A3.1 unfalsifiability problem, D-12 (partially), and the missing positive
control. *Cost:* lowest in the whole program — `simple_reference_v3` is $N{=}2$, already
installed, and trains in minutes. *Why first:* it is the only addition that changes
whether **any** result in the paper is interpretable. The random-$k$ control (brief's arm
8) is the cheapest possible test of whether the learned gate learns anything: match its
mean $k_\text{eff}$, randomise the selection, and compare. If random-$k$ ties the learned
gate, the paper's contribution becomes a clean, defensible negative-with-mechanism; if it
loses, the gate has value the current design cannot see. Either outcome is publishable;
the current ambiguity is not.

**2. The isolated microbenchmark of the communication module (C4).**
*Retires:* item 5 above, D-11/C-27, and the whole of `05_limitations.tex` ¶"Wall-clock not
measured". *Cost:* **near zero GPU-hours** — it is a forward+backward timing loop over
$N \in \{6,\dots,192\}$ with no training at all, minutes of wall-clock. *Why:* the
motivating claim of the paper is a cost claim, and it is currently unmeasured. Measuring
it converts a hand-waved $\mathcal{O}(\cdot)$ argument into a figure with a crossover
point. A finding that top-$k$ is *slower* below some $N$ is a real, publishable,
reviewer-disarming result — and it is the single highest information-per-GPU-hour
experiment available, ahead of anything requiring training.

**3. The Lagrangian-budgeted arm (brief's arm 5) with mechanism instrumentation (C3).**
*Retires:* D-03/C-08 (turns the diagnosis into a measurement), and is the only addition
with a path to a *positive* headline. *Cost:* moderate — one new arm, but it reuses the
entire backbone. *Why:* the paper's own conclusion (`06:12-15`) names this as the next
step. Making sparsity a controlled input rather than a noisy gradient is what converts
"our gate failed" into "here is the sparsity/return frontier and here is why the naive
gate cannot reach it". Instrumenting gate-gradient variance at the same time is what
makes §A3.2 item 6 go away.

Deliberately *not* in the top three: SMACv2 (highest cost, needs a 30 GB StarCraft II
install with a click-through EULA, and does not retire any defect that 1–3 do not),
and the full $k \in \{1,2,4,8,\lfloor (N-1)/2 \rfloor\}$ grid (necessary, but it is a
control that only becomes interpretable *after* 1 establishes that the channel matters).

---

## A4. Baseline fidelity check — is "Dense Attn-Comm" TarMAC?

Compared `src/agents/mappo.py:93-177` against TarMAC as published
(Das et al., ICML 2019, arXiv:1810.11187v2), read directly from the source PDF.

### What TarMAC actually specifies

- **Message structure** (Eq. 1): $m_i^t = [\,k_i^t \mid v_i^t\,]$ — a **signature** $k_i^t \in \mathbb{R}^{d_k}$ and a **value** $v_i^t \in \mathbb{R}^{d_v}$, with **asymmetric dimensions**: signature/query 16-d, value 32-d.
- **Query source** (Eq. 2): the receiver predicts $q_j^{t+1}$ **from its GRU hidden state $h_j^{t+1}$**, not from a raw observation embedding.
- **Timing** (Eq. 3): $c_j^{t+1} = \sum_{i=1}^{N} \alpha_{ji} v_i^t$ — messages emitted at $t$ are consumed at $t{+}1$. **One-step communication delay.**
- **Self-attention is included and is load-bearing.** Verbatim: *"Note that Equation 2 also includes $\alpha_{ii}$ corresponding to the ability to self-attend, which we empirically found to improve performance."*
- **Policy is recurrent**: a 1-layer GRU, hidden 128-d, encoding the full message-action-observation history.
- **Multi-round**: multiple rounds of communication between actions.
- **Training**: actor-critic with a centralised critic; RMSProp, lr $7\times10^{-4}$, $\alpha = 0.99$.

### What this repo implements

| Aspect | TarMAC | `AttentionComm` | Deviation |
|---|---|---|---|
| Signature/value split | Yes, $d_k{=}16$, $d_v{=}32$ | `qkv = Linear(in, 3*msg_dim)` → all three 16-d (`mappo.py:112`) | **Symmetric dims.** Minor. |
| Query source | GRU hidden state $h^{t+1}$ | `GELU(obs_embed(obs))`, feedforward (`mappo.py:204`) | **Major — no recurrence.** Messages are a function of the current observation only; they cannot encode history, which is TarMAC's stated purpose. |
| Timing | $t \to t{+}1$ (one-step delay) | same timestep $t$ (`mappo.py:203-206`) | **Major.** This repo assumes instantaneous communication; TarMAC does not. |
| Self-attention $\alpha_{ii}$ | **Included**, reported to help | **Masked out**: `scores.masked_fill(eye_mask, -inf)` (`mappo.py:129-130`) | **Major and directly contrary to the cited paper.** |
| Rounds | 1 or 2 (2-round is where TarMAC's headline gains come from) | 1 only, no option | **Major.** TarMAC's traffic-junction result is 84.6 % at 1 round vs **97.1 %** at 2 rounds. |
| Policy backbone | GRU-128 | MLP-64, tanh | Major. |
| Aggregation output | $c_j$ fed to GRU update | `proj(GELU(msg))` concatenated with raw obs (`mappo.py:175-176, 206`) | Moderate. |
| Optimiser | RMSProp | Adam | Minor (lr 7e-4 coincidentally matches). |

### Verdict

**The implemented module is not TarMAC.** It is a single-round, feedforward,
self-masked, same-timestep scaled-dot-product self-attention block over per-agent
observation embeddings. It retains exactly one of TarMAC's defining properties — the
signature/query targeting mechanism — and discards recurrence, multi-round
communication, the one-step delay, and self-attention, the last of which the original
authors explicitly identify as beneficial.

Consequences that must be actioned:

1. The arm called **"Dense Attn-Comm, TarMAC-style"** (`04:11-12`, `03_method.tex:20`) must be renamed to something honest — *single-round observation-conditioned attention communication* — **or** a faithful TarMAC must be implemented as a separate arm. Per the brief, Phase C requires the faithful implementation; the current module should be kept as a distinct, honestly-labelled arm.
2. **The method's name, `TopK-TarMAC`, is not currently earned.** It sparsifies a module that is not TarMAC. Either the faithful baseline lands and the name is justified, or the paper is renamed.
3. Comparing against TarMAC's *published numbers* (C-35, C-36) while running a different architecture is a further reason those cross-paper comparisons are hazardous — and C-35 is already refuted on its own terms.

---

## Summary of defects

| ID | Severity | One-line |
|---|---|---|
| D-00 | **Blocker** | torch 2.5.1+cu124 cannot run on sm_120 GPUs; all Phase C training is blocked. |
| D-14 | **Serious** | "15-percentage-point" TarMAC traffic-junction gain matches no pairing in the source. |
| D-12 | **Serious** | Ablation prose describes two cells that contain zero seeds. |
| D-10 | **Serious** | "Gap widens with $N$" is contradicted by the paper's own Table 1. |
| D-08 | **Serious** | Body (CI rule) and Table 1 caption ($p<0.05$ rule) state incompatible protocols; headline passes one and fails the other. |
| D-03 | **Serious** | Central mechanistic claim (gate variance) has no measurement anywhere in the repo. |
| D-15 | **Serious** | `reproduce.sh` trains for the wrong number of steps and skips the fixed-$k$ configs. |
| D-16 | **Serious** | `paper/` is untracked; no manuscript history exists. |
| D-09 | Major | Fig. 2 caption contradicts the body. |
| D-07 | Major | "Significantly worse" at $p=0.146$. |
| D-04 | Major | Reduce-to-baseline equivalences asserted, never tested; one is false as literally stated. |
| D-19 | Major | `MASTER_LOG.csv` missing one run and 7 required columns. |
| D-05 | Major | Hardware description obsolete; all budget/CPU-bound sentences invalid. |
| D-01 | Major | Abstract says "five-seed"; $N{=}12$ has three. |
| D-13 | Moderate | EPyMARL's environment misattributed to PettingZoo `simple_spread_v2`. |
| D-18 | Moderate | "$\sim$5 std for dense and baseline" is wrong for 3 of 6 cells. |
| D-11 | Moderate | Fig. 4 caption asserts an undefined "relevant operating regime". |
| D-21 | Moderate | Two inconsistent random-policy floors; the one in `REPRODUCTION.md` is untraceable. |
| D-06 | Moderate | `gpu_hours` column is wall-clock; `STATUS.md` contradicts it (20 vs 61). |
| D-02 | Moderate | Abstract's std sentence reads as a scaling trend but is a cross-arm contrast at one $N$. |
| D-20 | Minor | Population vs sample std used inconsistently across artifacts. |
| D-17 | Minor | "48 GPU-h per agent count" contradicts DECISIONS D-001 ("total"). |
| D-22 | Minor | `src/eval/` is empty. |
| — | — | **35 of 70 bib entries are uncited** (verified by parsing `\cite*{}` across `paper/`); the uncited block is dominated by LLM-agent work (InstructGPT, DPO, GRPO, ReAct, Reflexion, AutoGen, MetaGPT, CAMEL, Toolformer, Voyager, AgentBench, WebArena, vLLM, DeepSpeed-Chat, …) that is off-topic for this paper. |
| — | — | `make_figures.py:158` emits `MPE \texttt{simple\_spread} ($N{=}6$): learning curves` into a matplotlib title with `text.usetex=False`; verified it renders the backslashes and braces literally. |
