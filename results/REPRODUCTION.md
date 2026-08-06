# Baseline reproduction — MAPPO on MPE simple_spread (N=3)

## Setup

- **Env**: `mpe2.simple_spread_v3.parallel_env(N=3, max_cycles=25,
  local_ratio=0.5, continuous_actions=False)`.
- **Algorithm**: MAPPO (Yu et al. 2022) implemented in
  `src/agents/mappo.py` + `src/training/train_mappo.py`.
- **Hyperparameters** (config: `configs/mappo_mpe_n3_baseline.yaml`):
  hidden=64, lr=7e-4, n_envs=32, rollout_len=25, ppo_epochs=10,
  n_minibatches=1, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2,
  value_clip=100 (effectively disabled — see note below), entropy_coef=0.01,
  per_agent_reward=True (team reward / N for stable scale across N).

## Reproduction protocol

Compared against **published numbers in this exact env config we cannot find a
match** — Yu et al. (2022) report normalized scores over an internal "best"
reference, and EPyMARL (Papoudakis et al. 2021, Table 6) uses an older
PettingZoo version with a different per-step reward scale (their MAPPO on
simple_spread ≈ -132.5). We therefore report an **internal reproduction
target**:

- Random policy baseline (40 episodes, our env): **-77.76 ± 24.16**.
- Reproduction target: monotone improvement past -65 by 800k env steps, over
  5 seeds.

## Numbers (5 seeds, 800 k env steps each)

Filled in by `scripts/run_seeds.py` and the rows of `results/MASTER_LOG.csv`.
A pilot single-seed run reached **-57.82 ± 16.71** (greedy eval, 64 episodes).
The full 5-seed table is appended at the bottom of this file when the parallel
runs complete.

## Note on `value_clip`

Yu et al. (2022) and most CleanRL-style impls use `value_clip = 0.2` — a clip
on the per-update change of the value head. With our team-reward magnitude
(returns of order -75 raw, -25 after per_agent_reward), 0.2 was too tight and
the critic could not track returns; value loss diverged and the actor
flat-lined just above random (see `RESEARCH_LOG.md` 2026-05-20 v1 attempt at
500k steps that reached only ~-73). Increasing `value_clip` to 100 (effectively
off) and using `per_agent_reward` to keep returns in [-3, 0] per step recovered
learning. Both changes are tracked in DECISIONS.md (D-007).

## Reproduction status

- **Working baseline**: MAPPO trains and beats random by ~26 % at 800k steps.
- **Published-number match**: not applicable because no source reports
  comparable numbers under mpe2 v3 + local_ratio=0.5. Our own number serves
  as the reference for the method comparison in Phases 6 and 7.

## 5-seed reproduction (800 k env steps each)

| Seed | Final greedy eval mean | Final greedy eval std (within-eval, 64 eps) |
|------|------------------------|---------------------------------------------|
| 0    | -57.82                 | 16.71 |
| 1    | -60.04                 | 16.40 |
| 2    | -60.08                 | 16.02 |
| 3    | -56.79                 | 16.94 |
| 4    | -59.92                 | 15.67 |

**Across-seed mean ± std: −58.93 ± 1.51 (n=5).**
**Random policy in this env: −77.76 ± 24.16 (n=40).**

Improvement: +18.83 absolute units, +24.2 % over random.

## Verdict

Baseline reproduces stably (1.5-unit between-seed std at n=5). We have a
trustworthy MAPPO reference against which to compare the proposed
TopK-TarMAC method in Phases 6 and 7.

---

# Phase C-2 — reproducing the efficiency measurements (2026-08-05)

These are training-free and take minutes, not GPU-hours. They are the only
instrument in this repository that can test the paper's cost claim: end-to-end
training wall-clock is environment-bound and identical across arms to within
3% (D-018).

```bash
# Full campaign: headline sweep, k-sparsity sweep, batch-4096 (FLOP-bound),
# execution modes, bf16, and CPU. Sequential by design -- a second job on the
# same device would corrupt the between-arm comparison.
scripts/run_microbenchmarks.sh cuda:0

# Reduce to the tables the claim turns on.
.venv/bin/python scripts/summarise_microbenchmarks.py     # -> efficiency_summary.md
.venv/bin/python scripts/analyse_memory.py \
    --inputs results/microbenchmark_comm.json \
             results/microbenchmark_comm_batch4096.json   # -> memory_analysis.md
```

## What the arms mean

`dense` is the unfused reference; `dense_sdpa` the same function through the
fused kernel; `topk_masked` masks the dense weights and still runs the dense
aggregation matmul; `topk_gather` is genuinely sparse. **The paper's claim is
about `topk_gather`.** Comparing against `topk_masked` alone measures an
implementation artifact, which is what the superseded results did (D-019,
D-021).

## Protocol caveats that affect what the numbers mean

- **GPU clocks are not locked.** `nvidia-smi -lgc` requires privileges this
  account does not have. Instead: arms are timed **interleaved rep-by-rep** so
  interference lands on all of them at once, `min_ms` is reported as the
  primary statistic (contention can only add time), and per-cell inter-run
  spread is recorded in every JSON. This moved observed spread from >100% to
  typically <2%.
- **Absolute latencies depend on which arms share the interleaved group**
  (cache interference). Ratios within a group are the comparable quantity.
- **`collect_info` must be off when timing.** The diagnostic entropy term
  calls `.item()`, which synchronises the device inside the forward pass. With
  it on, the measurement is dominated by the synchronisation and dense latency
  looks flat in N.
- **No cross-generation replication.** The 2080 Ti host is gone and both local
  devices are the same Blackwell part (D-028). The CPU run is the only
  cross-architecture evidence.

## Reproducing the reduced-design training arms

```bash
.venv/bin/python scripts/make_reduced_design_configs.py   # 12 arms, N <= 12
scripts/run_reduced_design.sh 1                           # positive control
.venv/bin/python scripts/summarise_reduced_design.py
# Only proceed to tier 2 if the tier-1 contrasts read "A > B".
```

Tier 1 is a gate, not a warm-up: on a fully observable task, "sparsification
is free" and "the channel was never used" make identical predictions, so a
sparsification result measured before the channel is shown to carry
information is uninterpretable (D-011).
