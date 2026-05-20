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
