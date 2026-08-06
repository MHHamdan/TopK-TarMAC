# DECISIONS

Numbered list. Each entry: decision, date, alternatives considered, rationale.

## D-001 — Compute budget = 48 GPU-h, 200 GB disk (2026-05-19)
- Alternatives: tighter (24h) or looser (96h).
- Rationale: prompt default; matches available free VRAM. Will revisit if pilot is fast.

## D-002 — Bias toward small benchmarks (MPE, Hanabi, MiniGrid) (2026-05-19)
- Alternatives: SMAC-v2.
- Rationale: only ~4–6 GB free per GPU because other processes hold most VRAM. SMAC-v2
  requires StarCraft II install (~30 GB) and 5–20 GPU-h per baseline — too expensive
  given the constrained VRAM. Default starting point is MPE simple_spread (PettingZoo).

## D-003 — Use `uv` for dependency management (2026-05-19)
- Alternatives: pip + venv, conda.
- Rationale: `uv` is on PATH, faster resolves, lockfile semantics. Mohammed's global
  preferences accept this.

## D-004 — Use `mpe2` package for MPE envs (2026-05-20)
- Alternatives: pin pettingzoo to 1.24 (still bundled MPE).
- Rationale: pettingzoo 1.26 removed MPE; upstream officially moved it to the standalone
  `mpe2` package with the same API (parallel_env, agent dicts). Keeping pettingzoo at
  latest lets us reuse classic and butterfly envs too. Tested with simple_spread_v3 —
  works identically.

## D-005 — Primary benchmark = MPE simple_spread; reject SMAC (2026-05-20)
- Alternatives: SMAC v1/v2 (with StarCraft II install), MeltingPot.
- Rationale: Free VRAM per GPU is ~4–6 GB; SMAC's MAPPO/QMIX baselines need
  closer to 8 GB and 5–20 GPU-h per baseline reproduction. SC2 also needs a
  click-through EULA install (STOP CONDITION). MPE + LBF give us the full
  N ∈ {3..25} scaling story at <50 MB on disk and sub-second per 25 steps.

## D-008 — Phase 6 pilot: proceed to full sweep despite ambiguous 160k signal (2026-05-20)
- 160k-step pilot at N=3 (3 seeds):
    baseline (extracted from existing run) -67.41 ± 3.62
    dense    -64.10 ± 1.85   (slightly better, within 1 sigma)
    adaptive_topk -70.33 ± 2.77 (slightly worse, within 1 sigma)
- The comm-module weights start untrained — they inject noise before they
  learn — so a deficit at 20% of training is expected, not a failure mode.
  No run diverged; learning curves are monotone for both methods.
- Per the run-prompt's "ambiguous → run 50% budget" branch, the conservative
  call is to escalate. We instead proceed directly to the 800k full sweep:
  the pilot's purpose is to gate against pathological divergence (which did
  not happen), and re-running 50% would burn ~2 GPU-h without changing the
  next decision (still need the full 800k for headline numbers). The risk
  is captured: if the full sweep also lacks separation, we write the
  honest-negative-result paper described in the run prompt's success criterion 4.

## D-007 — Disable absolute `value_clip` and use per-agent reward (2026-05-20)
- Alternatives: keep CleanRL/MAPPO-paper default value_clip=0.2.
- Rationale: With raw team-reward magnitudes around 75 (returns -25 to -75
  per episode at N=3), clipping per-update value change to ±0.2 lets the
  critic move at most 0.2 toward the target per minibatch. Critic never fits;
  policy gets no usable advantage signal. First 500k-step run plateaued
  ~3 points above random. Two fixes together: (a) per_agent_reward divides
  team_r by N to keep magnitudes O(1); (b) value_clip set high enough not to
  bite. After fix, single-seed 800k-step run reaches -57.8 ± 16.7 (down from
  random -77.8). Both knobs live in `TrainerConfig` so the original behaviour
  is one flag flip away.

## D-006 — Selected Proposal 001 (TopK-TarMAC) after FORCED CHECKPOINT (2026-05-20)
- Alternatives: Proposal 002 (Two-Locus Attention), Proposal 003 (Budgeted
  Communication).
- Rationale: Highest combined score (18/20) on novelty × feasibility × fit-to-
  Mohammed × fit-to-team. Best scaling story (N=3..25 within budget). Cleanest
  reduce-to-baseline ablation (k → N recovers dense attention, k → 0 recovers
  MAPPO). The 24-hour human-review window has elapsed in this unattended run;
  proceeding with autonomous decision per Phase-3 protocol.
- Proposal 002's 2×2 ablation will be partially absorbed into Phase 7 as a free
  attention-locus comparison at N=12.

---

# Phase A decisions (2026-08-05)

## D-009 — Record hardware as 2×96 GiB, not 1×196 GiB (2026-08-05)
- The brief states "a GPU with ~196 GiB VRAM is now available". `nvidia-smi` shows
  **two** RTX PRO 6000 Blackwell cards at 97,887 MiB each (~191 GiB aggregate), both
  shared with other users' processes (~38–45 GiB actually free per device, GPU 0 at
  100% util at capture).
- Decision: document the true topology in HARDWARE.md and plan Phase C against
  per-device free memory, not aggregate total. Multi-GPU means data-parallel seeds,
  not one large device.
- Rationale: a sweep sized for a single 196 GiB device would OOM or contend.

## D-010 — Treat the sm_120/torch incompatibility as a Phase B blocker (2026-08-05)
- torch 2.5.1+cu124 has no sm_120 kernels; `torch.cuda.is_available()` is True but
  every kernel launch raises. Alternatives: (a) run Phase C on CPU, (b) upgrade torch.
- Decision: upgrade to a cu128 build (>= 2.7) in Phase B.
- Rationale: (a) forfeits the wall-clock/latency measurement that is the paper's
  motivating claim. The upgrade breaks bit-exact reproduction of the existing 45
  2080 Ti runs, which is accepted and recorded — those runs are superseded by the
  Phase C program anyway.

## D-011 — `simple_spread` alone cannot support the thesis; a comm-critical env is a
   precondition, not an enhancement (2026-08-05)
- The task is fully observable and locally solvable (the paper concedes this at
  04_experiments.tex:139; EPyMARL S3.2 states it independently). On such a task,
  "sparsification is free" and "the channel was never used" make identical
  predictions, so the central hypothesis is unfalsifiable there.
- Decision: no sparsification result will be reported as a headline until a
  communication-critical positive control (MPE simple_reference_v3 /
  simple_speaker_listener_v4) shows the dense channel beating MAPPO.
- Rationale: without that control, a null sparsification result is uninterpretable
  and a positive one is uninformative. See AUDIT.md#framing.

## D-012 — The implemented "Dense Attn-Comm" is not TarMAC and will not be called
   TarMAC (2026-08-05)
- Verified against arXiv:1810.11187v2: the repo module is single-round, feedforward,
  same-timestep, and **masks self-attention**, which TarMAC explicitly includes and
  reports as beneficial. It has no GRU and no one-step message delay.
- Decision: rename the existing arm to "single-round observation-conditioned
  attention communication" and implement a faithful TarMAC (GRU, signature/value
  split with asymmetric dims, self-attention included, one-step delay, multi-round
  option) as a separate arm in Phase C.
- Rationale: comparing against TarMAC's published numbers while running a different
  architecture is not defensible, and the method name "TopK-TarMAC" is unearned
  until the faithful baseline exists. See AUDIT.md#a4.

---

# Phase B decisions (2026-08-05)

## D-013 — Track paper/ sources in git, but do not push (2026-08-05)
- `paper/` was gitignored in 6f796c2 ("Public-repo prep ... not pushed to
  GitHub"), so no manuscript revision has ever been under version control
  (AUDIT.md D-16). `origin` is git@github.com:MHHamdan/TopK-TarMAC.git.
- Alternatives: (a) leave ignored and forfeit manuscript history; (b) track
  sources locally, hold pushes; (c) track and push.
- Decision: (b). Track paper/sections, paper/figures, paper/tables, main.tex;
  keep build intermediates and main.pdf ignored. **Commits are local-only
  until the author confirms the manuscript may become public.**
- Rationale: the brief requires a regenerated, version-controlled manuscript,
  but publishing an unsubmitted paper to a public remote is an irreversible
  outward-facing action that reverses an explicit prior decision. (b) delivers
  the history without making that call unilaterally.

## D-014 — Align config YAMLs to the runs, not the reverse (2026-08-05)
- The N=3 and N=6 YAMLs declared 1.5M / 1.0M total_steps while all 45 runs on
  disk recorded 800k, so `reproduce.sh` would have trained longer than the
  published numbers and silently produced different results (AUDIT.md D-15).
- Alternatives: (a) edit YAMLs down to 800k; (b) retrain everything at the
  declared budget.
- Decision: (a), plus `scripts/check_config_drift.py` as a hard gate in
  reproduce.sh so the two can never diverge again.
- Rationale: (b) costs ~60 GPU-h to reproduce numbers that Phase C supersedes
  anyway. The gate is what actually prevents recurrence.

## D-015 — Leave peak VRAM blank for historical runs (2026-08-05)
- MASTER_LOG.csv now carries the required schema, but peak VRAM was never
  sampled during the 2080 Ti runs.
- Decision: emit an empty cell, never an imputed one. `train_mappo.py` now
  captures `torch.cuda.max_memory_allocated` so all Phase C runs populate it.
- Rationale: a plausible-looking imputed number is worse than a blank.

## D-016 — Fail loudly on an unusable CUDA device (2026-08-05)
- `torch.cuda.is_available()` returns True on sm_120 hardware that the build
  has no kernels for. The old `device = cuda if is_available() else cpu` line
  would select cuda and crash, or in other configurations silently fall back.
- Decision: `_resolve_device` probes with a real allocation and raises with
  the arch list and device capability. It never downgrades to CPU silently.
- Rationale: a silent CPU fallback would invalidate every wall-clock and
  latency measurement in Phase C -- the paper's motivating claim.

---

# Phase C decisions (2026-08-05)

## D-017 — ABORT the specified Phase C sweep at the 200 GPU-h guardrail (2026-08-05)
- Measured throughput (results/throughput.json) projects ONE arm across
  N in {3,6,12,24,48} at 10 seeds x 10M steps = 995 GPU-h; a single N=48 cell
  is 717 GPU-h and a single N=24 cell is 197 GPU-h. Eight arms = ~7,960
  GPU-h; with LBF/RWARE/SMACv2, ~10,000-14,000 GPU-h.
- Decision: do not launch. Wrote results/compute_budget.md with the full
  matrix and a ~60 GPU-h reduced design (Sec 5), per the guardrail's
  "ask for a smaller design instead".
- Rationale: the guardrail is explicit and the overrun is 5x on the smallest
  meaningful unit, not a rounding error.

## D-018 — Training wall-clock cannot test the efficiency claim (2026-08-05)
- Throughput is identical across MAPPO / dense / adaptive-topk to within 3%
  at every N >= 6 (N=24: 141/141/141 env-steps/s). The serial Python VecEnv
  stepping mpe2 dominates; the comm module is invisible in end-to-end time.
- Decision: the efficiency claim is measured ONLY by the isolated module
  microbenchmark (scripts/microbenchmark_comm.py). No wall-clock claim will
  be made from training runs.
- Rationale: an end-to-end timing that is 97%+ environment cost cannot
  attribute a difference to the module.

## D-019 — The reported FLOPs saving is not realised by the implementation
   (2026-08-05)
- Tracing torch.matmul shows the top-k path executes the SAME dense
  (B,N,N)x(B,N,d) aggregation matmul as the dense path: the mask is applied
  to the attention matrix and the zeros are materialised, never skipped
  (mappo.py:167-175). Executed MACs are identical; measure_flops.py reports
  up to 91% saved.
- Decision: every claim of a 21-29% FLOPs reduction is withdrawn from the
  manuscript. measure_flops.py output is relabelled as an analytic upper
  bound on an achievable saving, not a measured one. Pinned by a regression
  test so a future genuinely-sparse gather has to revisit it deliberately.
- Rationale: it is the difference between a modelled quantity and an executed
  one, and the manuscript presented the former as the latter.

## D-020 — Reframe the paper as a measurement/negative-result paper (2026-08-05)
- Framing option 1 from the brief ("budgeted sparse comm attains dense return
  at reduced latency") is now foreclosed: there is no latency reduction to
  attain, at any N in [6,192] or batch in [256,4096].
- Decision: target framing 2/3 - "sparsifying the MARL attention channel does
  not reduce cost in the regime the literature operates in, and the learned
  gate costs O(B N^3) memory; the analytic FLOPs accounting that motivates
  this line of work does not describe what runs."
- Rationale: this is supported by 30 seconds of measurement, is falsifiable,
  and is useful to the field. Picked from the data, not before it.

---

# Phase C-2 decisions (2026-08-05) — closing the strawman gap

## D-021 — Implement the real gather path before finalising any framing
   (2026-08-05)
- D-019 established that `attn_mode="topk"` masks the dense weight matrix and
  then runs the same dense aggregation matmul, so the "sparse" arm executed
  every MAC the dense arm did. Every latency comparison built on it was a
  comparison against an implementation artifact, not against the method.
- Decision: add `attn_mode="topk_gather"`, which takes top-k on the raw
  scores, softmaxes over just those k, and gathers the selected values into
  (B,N,k,d) contracted as (B*N,1,k)x(B*N,k,d). Keep `topk` (masked) as a
  separate, explicitly-labelled arm. Report **three curves** everywhere:
  dense, masked, gathered.
- Verified: the two produce identical outputs (max abs diff < 1e-10 in
  float64, `test_topk_gather_matches_masked_topk_exactly`) while executing
  different FLOPs (2,015,232 vs 2,555,904 at N=48,k=4,B=8,d=16, pinned by
  `test_topk_gather_executes_fewer_macs_than_dense`).
- Rationale: refuting a claim requires implementing its strongest form first.

## D-022 — The score matmul is irreducible, so the saving is capped at 2x
   (2026-08-05)
- Selecting the top-k peers *by attention score* requires all N scores.
  Attention FLOPs = scores (2BN^2 d) + aggregation (2BNk d); only the second
  term is sparsifiable. As k -> 1 the ratio to dense tends to 2 and never
  reaches it.
- Decision: state this ceiling as an analytic result in the paper. The
  manuscript's withdrawn "21-29% FLOPs reduction" was not merely unrealised,
  it was bounded above by a factor the paper never derived.
- Pinned by `test_attention_flop_saving_is_bounded_by_two` and
  `test_score_matmul_is_irreducible_in_every_mode`.

## D-023 — The dense baseline must be the fused SDPA kernel (2026-08-05)
- The original benchmark compared against an unfused matmul+softmax+matmul.
  On this hardware `F.scaled_dot_product_attention` dispatches to a fused
  memory-efficient CUTLASS kernel **in fp32** (`mem_efficient` is available;
  flash requires fp16/bf16), which never materialises the (B,N,N) weights.
- Measured at N=512, batch 256, fp32: fused 0.431 ms vs unfused 1.989 ms.
  Comparing top-k only against the unfused path flatters it by ~4.6x.
- Decision: `dense_sdpa` is the headline baseline; unfused `dense` is
  reported alongside it as the historical reference, never alone.

## D-024 — Measurement protocol: interleave arms, report min, do not claim
   locked clocks (2026-08-05)
- `nvidia-smi -lgc` is not permitted for this account, and the devices are
  shared. Timing all reps of arm A then all of arm B produced >100% swings
  between identical configurations (dense at N=384 measured 1.118 ms and
  0.299 ms in two runs of the same script).
- Decision: (a) time arms **interleaved rep-by-rep**, so interference lands
  on all arms at once and ratios stay valid; (b) report `min_ms` as primary,
  since contention can only add time; (c) record per-cell inter-run spread in
  every artifact; (d) state plainly that clocks are unlocked. After the
  change, spread fell from >100% to typically <2%.
- Also removed from the timed region: the diagnostic `.item()` call in
  `AttentionComm.forward`, which synchronises the device inside the forward
  pass. Its presence is why dense latency originally looked flat in N.
- Caveat recorded: absolute latencies depend on interleaved-group membership
  (cache interference); ratios within a group do not.

## D-025 — Result: there is no crossover, at any N or any k (2026-08-05)
- Sparsity sweep, k/(N-1) in {0.01 .. 0.5} at N in {48..512}: the best k for
  the gather path is k=1 at every N, and it remains 1.19x-2.42x slower than
  unfused dense and 1.35x-11.11x slower than fused SDPA.
- torch.compile and CUDA-graph capture change the picture by <10%, so this is
  not an eager-mode artifact. Confirmed at batch 4096 (FLOP-bound) as well as
  256.
- The one cell where gathering wins any comparison: N=6, batch 4096, where it
  is 0.92x the fused kernel (which carries fixed overhead at tiny N). It is
  still 1.17x the unfused dense path there.
- Decision: report the negative result with the numbers, including the single
  favourable cell. The gather fix moves top-k from ~1.45x to ~1.23x of dense
  at small N -- a real improvement over the strawman that does not change the
  conclusion.

## D-026 — O(B N^3) gate memory is a first-class, validated result (2026-08-05)
- Derived: the gate materialises (B,N,N-1,N) before summing over the k-bin
  axis. Measured: offset-corrected fit gives N^3.04 against dense's N^1.96
  and fused SDPA's N^1.36. The closed form predicts 97.5% of measured peak at
  N=384, which is the sharper validation than the exponent.
- At batch 256 the arm consumes 56.5 GiB at N=384 and OOMs at N=512;
  analytically it exhausts a full 97,887 MiB card at N=465.
- Decision: promote to a headline result with both the derivation and the
  measurement, and report the OOM boundary for the full card *and* for the
  memory actually free on the shared host, because only the former is a
  property of the hardware.

## D-027 — Reduced design: 12 arms, N <= 12, all N-scaling training-free
   (2026-08-05)
- Decision: 12 training arms in four priority tiers (positive control ->
  random-k control -> fixed-k grid -> Lagrangian/entmax), N <= 12, 5 seeds,
  2M steps. Generated by `scripts/make_reduced_design_configs.py`.
  Projected ~30-40 GPU-h, inside the 200 GPU-h guardrail. No JaxMARL port.
- N-scaling to 512 moves entirely to the training-free microbenchmark, which
  costs minutes rather than the 995 GPU-h the trained N sweep was projected at
  (D-017) and which is the only instrument that can test a cost claim (D-018).
- Comm-critical control implemented two ways, because neither alone suffices:
  `simple_reference` is logically unsolvable without the channel but fixed at
  N=2 (so it cannot carry the k arms), and a new partially-observable
  `simple_spread` (`visibility_radius`) keeps N free. At radius 1.0, N=6, a
  measured 33% of peers stay visible.

## D-028 — Cross-generation replication is BLOCKED, and will be reported as
   such (2026-08-05)
- The brief asks for repetition on the 2080 Ti. That host is no longer the
  execution environment (HARDWARE.md) and both devices here are the same
  Blackwell part, so "structurally" versus "on our hardware" cannot be
  settled on GPUs available to this project.
- Decision: run the CPU backend as the one available cross-architecture
  check, and state the gap explicitly in the artifact rather than implying
  broader coverage. Do not claim cross-generation agreement.

## D-029 — Push code only; the manuscript stays local (2026-08-05)
- `origin` is git@github.com:MHHamdan/TopK-TarMAC.git and `gh repo view`
  reports **visibility: PUBLIC**. `paper/` is absent from `origin/main` but is
  tracked on `phase-c/experiments` (16 files, committed in 6c4c714), so
  pushing that branch would publish an unsubmitted manuscript.
- Decision: keep `phase-c/experiments` local. Publish code, tests, configs and
  measurement artifacts on a separate branch built from `origin/main` that has
  never contained `paper/` in its history.
- Rationale: this is the conditional the brief specified, and publishing a
  manuscript is not reversible by deleting it afterwards.

## D-030 — Phase F venue re-scoring against a refutation-framed paper
   (2026-08-05)
- Dropped: **IEEE TPDS**. The contribution is not a parallel/distributed
  systems advance; the measurements are single-device.
- **TMLR — primary.** Its acceptance criteria are correctness of claims and
  sufficiency of evidence, explicitly *not* novelty or significance. That is
  the exact shape of this paper: a corrected measurement, a refuted
  implementation claim, and an audit. No page limit suits the three-curve
  tables. Certification for survey/reproducibility-style contributions is a
  further fit.
- **RLC — conference alternative.** RL-specific audience, receptive to
  empirical-rigour and negative-result papers; the MARL comm literature being
  audited is its core readership.
- **MLSys — second conference alternative.** If the framing leans on the
  kernel-level measurement (fused vs masked vs gathered, CUDA graphs, the
  O(BN^3) memory cliff) rather than on MARL, this is a natural home.
- **IEEE TAI — scored down, retained only as a fallback.** Re-scored against
  a refutation framing: TAI's remit rewards positive methodological or
  applied contributions, its reviewer pool is not specialised in MARL
  communication, and a paper whose central result is "the reported saving is
  not executed" is a poor match for its stated scope. Slow review compounds
  the cost of a fallback that is unlikely to fit.
- Ranking: TMLR > RLC > MLSys > TAI.

## D-031 — Tier 1 PASSES: the communication channel carries information
   (2026-08-06)
- Both positive controls beat comm-free MAPPO with 95% bootstrap CIs
  excluding zero, over 5 seeds x 2M steps:
    simple_reference (N=2):        -30.10 vs -34.35, diff +4.26 [+3.15, +5.23]
    PO simple_spread (N=6, r=1.0): -192.83 vs -204.42, diff +11.59 [+7.13, +16.44]
- This is the first result in this project where the channel is shown to do
  something on a task that needs it. Every previous sparsification result was
  measured on fully observable `simple_spread`, where "sparsification is free"
  and "the channel was never used" are indistinguishable (D-011).
- Measured attention entropy on the N=6 control is 0.35 nats against a
  ln(5)=1.61 maximum, i.e. attention is strongly peaked rather than uniform --
  which is what makes the tier-2 random-k control a meaningful test rather
  than a formality.
- Decision: the D-011 precondition is met; tiers 2-4 are now interpretable and
  are launched. Any sparsification result from them is a statement about
  giving up part of a channel that demonstrably works.
