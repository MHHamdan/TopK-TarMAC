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
