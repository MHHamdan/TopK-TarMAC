# Comparison with verified published benchmarks

This document records the closest published numbers we could find for MPE
`simple_spread` and attention-communication MARL, and explicitly identifies
which of our results are directly comparable, which are not, and why. Every
number reproduced below was fetched from the published paper via the
ar5iv HTML mirror on 2026-05-21 — citations are entries in
`literature/references.bib`.

---

## 1. EPyMARL benchmark (Papoudakis et al. 2021)

**Source:** `papoudakis2021epymarl` — Benchmarking Multi-Agent Deep RL
Algorithms in Cooperative Tasks; arXiv:2006.07869; Table 3.

**Protocol:** MPE `simple_spread`, 5 random seeds, 20 M timesteps for
on-policy methods, 2 M timesteps for off-policy. Maximum returns with
95 % confidence intervals.

| Algorithm | Return ± CI |
|-----------|-------------|
| MAA2C | **−129.90 ± 1.63** (best) |
| QMIX  | −126.62 ± 2.96 |
| VDN   | −131.03 ± 1.85 |
| MAPPO | −133.54 ± 3.08 |
| IQL   | −132.63 ± 2.22 |
| IPPO  | −133.86 ± 3.67 |
| IA2C  | −134.43 ± 1.15 |
| MADDPG | −141.70 ± 1.74 |
| COMA  | −204.31 ± 6.30 |

**Environment version:** PettingZoo's `simple_spread_v2` (used in EPyMARL's
codebase as of 2021).

## 2. MAPPO benchmark (Yu et al. 2022)

**Source:** `yu2022mappo` — The Surprising Effectiveness of PPO in
Cooperative MARL; arXiv:2103.01955.

**Protocol:** MPE `simple_spread`, 10 seeds, results as Figure 1 (curve,
not tabular). Paper states "MAPPO achieves performance comparable and
even superior to the off-policy baselines."

The MAPPO paper does not include a tabular MPE simple_spread number we
could quote verbatim — they rely on figure curves with min-max
normalisation to a per-task best/worst reference.

## 3. TarMAC (Das et al. 2019)

**Source:** `das2019tarmac` — Targeted Multi-Agent Communication;
arXiv:1810.11187.

**Note:** TarMAC does *not* evaluate on MPE `simple_spread`. Its
cooperative-comm benchmarks are SHAPES, traffic-junction, House3D, and a
custom predator-prey. The closest analogue numbers are:

**Traffic-junction success rate (higher is better):**

| Setting | No-comm baseline | TarMAC (2-round) |
|---------|------------------|------------------|
| Easy ($N_{\max} = 5$) | 84.9 ± 4.3 % | **99.9 ± 0.1 %** |
| Hard ($N_{\max} = 20$) | 74.1 ± 3.9 % | **97.1 ± 1.6 %** |

**Predator-prey time-to-catch (lower is better):**

| 10 agents, 20×20 | IC3Net (no targeted comm) | TarMAC + IC3Net (2-round) |
|------------------|---------------------------|---------------------------|
| Steps to completion | 52.4 ± 3.4 | **35.57 ± 3.96** |

These show ~15-percentage-point and ~32 % relative gains for soft-attention
communication over no-comm. They are not on MPE simple_spread.

---

## 4. Our numbers (this paper)

**Environment:** `mpe2.simple_spread_v3.parallel_env`, max_cycles=25,
local_ratio=0.5. **Training budget:** 800k env steps for $N\in\{3,6\}$;
1.5M env steps for $N{=}12$. **Seeds:** 5 for $N\in\{3,6\}$; 3 for $N{=}12$.

**Random policy baseline (our env, our protocol):** −77.76 ± 24.16
(40 episodes, single trajectory each).

| $N$ | MAPPO | + Dense Attn-Comm | + Adaptive TopK |
|-----|-------|-------------------|-----------------|
| 3   | −58.93 ± 1.37  | −60.27 ± 3.53  | −57.46 ± 3.31 |
| 6   | −225.04 ± 13.93 | **−212.91 ± 3.72** (d=+1.06) | −223.22 ± 20.01 |
| 12  | **−716.98 ± 4.57** | −730.44 ± 16.85 | −782.40 ± 40.19 (d=−1.87) |

---

## 5. Why direct numerical comparison fails

Three reasons our absolute returns cannot be compared to EPyMARL's:

1. **Environment-version drift.** EPyMARL uses PettingZoo's
   `simple_spread_v2` (released 2020-2021); we use the `mpe2` package's
   `simple_spread_v3`, which inherited a different reward-scale and
   landmark-distance penalty when the env was spun out of PettingZoo
   1.25+. EPyMARL's MAPPO at −133 in their env is not the same scalar
   as our MAPPO at −58 in ours.
2. **Different training budgets.** EPyMARL trains 20 M timesteps; we
   train 800k–1.5M. Even within the same env, 25× the steps would push
   our numbers to a different convergence point.
3. **Local ratio and N count.** EPyMARL's table uses one fixed $N$
   (typically 3) and a fixed `local_ratio=0.5`; our scaling experiment
   spans $N \in \{3, 6, 12\}$ to study the asymptotic-N regime.

## 6. Comparable quantities

What we can quote in the paper:

| Quantity | Our value | Published comparator |
|----------|-----------|----------------------|
| Cohen's $d$ of attn-comm over baseline at $N{=}6$ | $d = 1.06$ (dense) | TarMAC reports +15 pp on traffic-junction (no $d$ given, but the 84.9 % → 99.9 % gain on a binary success rate corresponds to $d \approx 5$–7 if std is small). |
| Relative gain from communication at $N{=}6$ | $+5.4\%$ (12.1/225.0 return points) | TarMAC: $+15.0$ pp success rate (easy traffic-junction); $+32\%$ time-to-catch (predator-prey). |
| Between-seed std of MAPPO baseline | 1.37 (N=3), 13.93 (N=6), 4.57 (N=12) | EPyMARL MAPPO 95% CI = ±3.08 (5 seeds), so std $\approx$ 1.74 — comparable to our $N{=}3$ baseline std. |
| Reference benchmark MAPPO | (this paper) | EPyMARL MAPPO −133.54 ± 3.08 (different env) |
| Number of seeds | 5 / 5 / 3 | EPyMARL 5 seeds; MAPPO paper 10 seeds; TarMAC 5 seeds |

## 7. Honest framing for the paper

We will cite EPyMARL `\citep{papoudakis2021epymarl}` as the reference
benchmark, explicitly note that absolute numbers cannot be compared due
to env-version drift, and instead compare on (a) Cohen's $d$ of the
attention-comm gain over MAPPO, (b) between-seed standard deviation
(rough check that our MAPPO reproduces the EPyMARL stability), and
(c) the qualitative direction of effect (does attention help at this $N$?).
TarMAC `\citep{das2019tarmac}` is cited as the closest prior comm-attention
work, with its traffic-junction +15 pp and predator-prey +32 % gains
quoted as the published effect sizes.
