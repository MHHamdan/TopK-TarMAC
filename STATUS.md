# STATUS

**Active phase:** Phase 1 — Literature review
**Last update:** 2026-05-20

## Plan summary (5 lines)
1. Phase 0 sets up repo, deps (uv-managed venv, torch + MARL stack), and smoke tests.
2. Phase 1 builds a structured 40+ paper literature review across cooperative MARL, agentic LLM-RL, distributed RL infra, and game-theoretic MARL.
3. Phases 2–3 pick the smallest viable benchmark and propose 3 ranked contributions; FORCED CHECKPOINT for human review.
4. Phases 4–7 reproduce baseline (±10%), implement method as clean delta with sanity-reduction ablation, run pilot (FORCED CHECKPOINT), then full 5-seed sweep.
5. Phases 8–9 generate figures/tables from tracked scripts and write the 10-page workshop paper.

## Compute budget
- Detected: 4× RTX 2080 Ti (11 GB ea., ~4–6 GB free on each currently), 187 GB RAM, 24 TB disk
- Budget: 48 GPU-hours total, 200 GB disk. Realistic given low free VRAM → bias toward small benchmarks (MPE, Hanabi, MiniGrid)

## Next action
Build structured literature review across 4 lanes: cooperative MARL (Lane A),
agentic LLM-RL (Lane B), distributed RL infra (Lane C), game-theoretic / large-scale
MARL (Lane D). Target ≥40 entries in references.bib, with one note per paper.

## Blockers
None.

## Phase 0 completion
- Commit `ff90ccf` on branch `research`.
- 6/6 smoke tests passing. torch 2.5.1+cu124 sees all 4 GPUs.
- Decision D-004: pettingzoo 1.26 dropped MPE → switched to `mpe2` package (still
  upstream-blessed split). Pinned in pyproject.toml.
