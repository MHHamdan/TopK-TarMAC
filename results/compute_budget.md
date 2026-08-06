# Compute budget — Phase C

Written **before** launching any sweep, per the project compute guardrail
(abort and report if a single sweep is projected above 200 GPU-hours).

All projections are built from **measured** throughput on the current host
(`results/throughput.json`), not from an assumed rate.

---

## 1. Measured throughput

`scripts/measure_throughput.py`, 32 parallel envs, rollout length 25,
device `cuda:1` (RTX PRO 6000 Blackwell). Slowest arm per $N$:

| $N$ | MAPPO | Dense | Adaptive top-$k$ | Used for projection |
|-----|-------|-------|------------------|---------------------|
| 3   | 2 806 | 4 456 | 4 309 | 2 806 |
| 6   | 1 744 | 1 709 | 1 691 | 1 691 |
| 12  |   530 |   528 |   512 |   512 |
| 24  |   141 |   141 |   141 |   141 |
| 48  |   —   |   —   |   —   |    39 (extrapolated, $N^{-1.86}$) |

**The three arms are within 3 % of each other at every $N \ge 6$.** The
pipeline is environment-bound: the serial Python `VecEnv` stepping `mpe2`
dominates, and the communication module contributes nothing measurable to
end-to-end wall-clock. This is a measurement, and it has two consequences:

1. End-to-end wall-clock **cannot** validate or refute the efficiency claim
   in this codebase. Only the isolated microbenchmark (§4) can.
2. GPU-hours here are really CPU-bound wall-clock hours. Adding GPUs does not
   speed up a sweep; adding CPU workers does.

Throughput falls as $N^{-1.86}$, close to quadratic — this is the environment's
per-step cost in $N$, not the attention module's.

---

## 2. Projected cost of the Phase C program as specified

At the specified budget of 10 M env steps per MPE run and 10 seeds per
headline cell:

| $N$ | h / run | h / cell (10 seeds) |
|-----|---------|---------------------|
| 3   |   0.99  |     9.9 |
| 6   |   1.64  |    16.4 |
| 12  |   5.42  |    54.2 |
| 24  |  19.71  | **197.1** |
| 48  |  71.71  | **717.1** |

- **One arm** across the full $N \in \{3,6,12,24,48\}$ sweep, 10 seeds:
  **995 GPU-h**
- **Eight arms** (MAPPO, dense, faithful TarMAC, fixed-$k$ grid, adaptive
  Gumbel, Lagrangian, entmax, distance, random-$k$): **≈ 7 960 GPU-h**
- Plus comm-critical MPE, LBF and RWARE at 20 M steps, and SMACv2:
  conservatively **another 3 000–6 000 GPU-h**

### Guardrail verdict: ABORT

**A single sweep — one arm across the $N$ sweep at 10 seeds — is projected at
995 GPU-h, five times the 200 GPU-h limit.** A single *cell* at $N{=}48$ is
717 GPU-h and a single cell at $N{=}24$ is 197 GPU-h, at the limit on its own.

The full program is ~10 000–14 000 GPU-h. On this host, shared with other
users, at ~24 usable parallel CPU workers, that is **3–5 weeks of continuous
wall-clock**. It is not fundable under the stated guardrail and is not
launched.

A reduced design is proposed in §5.

---

## 3. What was run instead (near-zero cost)

| Measurement | Cost | Artifact |
|---|---|---|
| Throughput probe, $N \in \{3,6,12,24\}$ × 3 arms | 2.1 min | `results/throughput.json` |
| Communication-module microbenchmark, $N \in \{6..192\}$ × 4 arms, 100 reps | **15.3 s** | `results/microbenchmark_comm.json` |

Total: **under 3 minutes**. Both are training-free.

---

## 4. Microbenchmark result — the motivating claim does not survive

`scripts/microbenchmark_comm.py`, batch 256, msg_dim 16, median of 100 timed
reps after 20 warmup, explicit CUDA sync on both sides, `cuda:1`.

| $N$ | Dense fwd (ms) | Top-$k$ fwd (ms) | Adaptive fwd (ms) | Dense VRAM (MiB) | Adaptive VRAM (MiB) |
|-----|---------------|------------------|-------------------|------------------|---------------------|
| 6   | **0.202** | 0.357 | 0.612 |  18.8 |    19.4 |
| 12  | **0.424** | 0.498 | 0.697 |  20.7 |    23.5 |
| 24  | **0.422** | 0.496 | 0.628 |  26.1 |    45.2 |
| 48  | **0.433** | 0.510 | 0.707 |  42.8 |   169.9 |
| 96  | **0.429** | 0.501 | 1.863 | 101.7 | 1 038.3 |
| 192 | **0.479** | 0.917 | 11.829 | 309.3 | **7 509.8** |

Three findings, each directly contradicting a premise of the current
manuscript:

**(a) Top-$k$ is slower than dense at every agent count tested.** There is no
crossover anywhere in $6 \le N \le 192$. At $N{=}6$ top-$k$ costs +77 %; at
$N{=}192$ it costs +91 %. The gather's indexing and launch overhead exceeds
whatever the reduced multiply-accumulate count saves.

**(b) Dense attention latency is essentially flat in $N$** — 0.202 ms at
$N{=}6$ to 0.479 ms at $N{=}192$, a 2.4× rise across a 32× rise in $N$. The
$\mathcal{O}(N^2)$ cost that motivates the entire paper **does not manifest as
wall-clock** in this regime: the kernel is launch- and memory-bound, not
FLOP-bound. An analytic FLOPs saving of 21–29 % is therefore not a saving of
anything measurable.

**(c) The adaptive gate's activation memory grows cubically.** The
implementation materialises a $(B, N, N{-}1, N)$ selection tensor
(`mappo.py:162`), so memory grows as $\mathcal{O}(B N^3)$ against dense
attention's $\mathcal{O}(B N^2)$: 19.4 MiB at $N{=}6$ to 7 509.8 MiB at
$N{=}192$, **24× dense**, while being **24.7× slower**. At $N{=}192$ the arm
that exists to *save* compute costs an order of magnitude more of both.

### 4b. The launch-bound caveat, tested and resolved

At batch 256 the GPU is launch-latency-bound, which could have explained (a)
and (b) away. Repeating at **batch 4096**
(`results/microbenchmark_comm_batch4096.json`) moves the largest case firmly
into the FLOP-bound regime — dense at $N{=}192$ rises from 0.479 ms to
8.928 ms, scaling linearly with batch — and the conclusion does not change:

| $N$ | Dense fwd (ms) | Top-$k$ ($k{=}\lfloor(N{-}1)/4\rfloor$) | Ratio | Adaptive |
|-----|---------------|------------------------------------------|-------|----------|
| 12  | **0.194** | 0.282 | 1.45× | 0.677 |
| 48  | **0.578** | 1.033 | 1.79× | 4.541 |
| 192 | **8.928** | 19.627 | **2.20×** | out of memory |

Top-$k$ remains slower than dense in the FLOP-bound regime — by a *wider*
margin (2.20×) than in the launch-bound one. The adaptive arm exhausts
memory entirely. **There is no crossover at any $(N, \text{batch})$ tested.**

### 4c. Root cause — the saving is never executed

Tracing `torch.matmul` through both code paths at $N{=}48$:

```
dense  k=N-1: aggregation matmul [((8, 48, 48), (8, 48, 16))]
topk   k=4  : aggregation matmul [((8, 48, 48), (8, 48, 16))]
```

The two are **identical**. `AttentionComm.forward` masks and renormalises the
attention matrix and then calls `torch.matmul(attn, v)` with `attn` still
shaped $(B, N, N)$ (`mappo.py:167-175`). The zeros are *materialised*, not
skipped. Top-$k$ therefore executes every multiply-accumulate that dense
executes, and pays the `topk` sort, `scatter_`, multiply and renormalise on
top. It is dense attention plus overhead, by construction.

Executed MACs at $N{=}48$, $d{=}16$, $B{=}8$: **294 912 for dense and 294 912
for top-$k$**. The counter in `scripts/measure_flops.py` reports
$1 - k/(N{-}1) = 91\,\%$ saved.

**This means the "21–29 % FLOPs reduction" reported throughout the manuscript
(abstract, §Experiments ¶FLOPs, Fig. 4, conclusion) describes a quantity the
code never computes.** It is a property of the analytic formula, not of the
implementation. Realising it would require gathering the selected values into
a $(B, N, k)$ tensor and contracting that — a different implementation, which
the microbenchmark above suggests would still lose to a single dense GEMM at
these sizes.

Pinned by `tests/test_comm_module.py::test_topk_aggregation_executes_a_dense_matmul`.

---

## 5. Proposed reduced design (≈ 60 GPU-h, fits the guardrail)

Ordered by information per GPU-hour. Every item retires a specific audit
defect.

| # | Experiment | Cost | Retires |
|---|---|---|---|
| 1 | Batch sweep on the microbenchmark (batch $\in \{32 .. 16384\}$), to locate the FLOP-bound crossover if one exists | **~2 min** | the §4 caveat |
| 2 | Comm-critical positive control: `simple_reference_v3` + `simple_speaker_listener_v4`, MAPPO vs dense vs adaptive, 10 seeds, 2 M steps | **~6 h** | D-011, the A3.1 unfalsifiability problem |
| 3 | Random-$k$ control at $N{=}6,12$, 10 seeds, matched mean $k_\text{eff}$ | **~7 h** | "does the gate learn anything" |
| 4 | Fixed-$k$ grid $k \in \{1,2,4,8\}$ at $N{=}6,12$, 5 seeds | **~14 h** | D-12 (the $n{=}0$ ablation cells) |
| 5 | Mechanism instrumentation (gate entropy, $k_\text{eff}$ variance, gate-gradient variance) logged on runs 2–4 | included | D-03 / C-08 |
| 6 | Entmax / sparsemax arm at $N{=}6,12$, 5 seeds | **~14 h** | tests the paper's own diagnosis |
| 7 | Lagrangian-budgeted arm, $\kappa/(N{-}1) \in \{0.25,0.5,0.75\}$ at $N{=}6$, 5 seeds | **~12 h** | the only route to a positive headline |
| 8 | $N{=}24$ extension for the two surviving arms, 5 seeds, 2 M steps | **~6 h** | the $N$-ceiling limitation |

**Total ≈ 60 GPU-h**, comfortably inside the guardrail, and ~3 days of
wall-clock at 24 parallel workers.

Deliberately dropped, with reasons recorded before seeing any result:

- **$N{=}48$** (717 h/cell). The microbenchmark already covers $N$ up to 192
  for the cost question, and the environment-bound training loop makes
  $N{=}48$ training cost 70 h/run for a return number that $N{=}24$ already
  bounds.
- **LBF and RWARE** at 20 M steps. Both need new env integrations plus
  ~2 000 GPU-h. The comm-critical control (item 2) buys the same
  interpretability guarantee for ~6 h.
- **SMACv2.** 30 GB StarCraft II install behind a click-through EULA, and it
  retires no defect that items 1–8 do not.
- **The JaxMARL port.** It is the only way to afford the *specified* design
  (on-device vectorised envs would plausibly give 100–1000× throughput and
  make 10 seeds × 5 $N$ cheap). It is also a multi-day rewrite with real
  numerical-equivalence risk, and it cannot be validated against the existing
  PyTorch results without re-running them. Not attempted under this budget;
  recorded as the top infrastructure recommendation.

---

## 6. Accounting note

The `gpu_hours` column in the historical `MASTER_LOG.csv` was wall-clock
seconds ÷ 3600, and those runs were CPU-bound (AUDIT.md D-06). The same is
true of every projection above: these are **resource wall-clock hours**, and
running $W$ seeds in parallel divides elapsed time by $W$ while leaving the
resource total unchanged. Peak VRAM is now sampled per run
(`train_mappo.py:peak_vram_mib`), so Phase C runs will populate the column
that historical runs left blank.
