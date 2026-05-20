# STATUS

**Active phase:** Phase 0 — Setup
**Last update:** 2026-05-19

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
Finish Phase 0: write pyproject + venv install, smoke tests, initial commit.

## Blockers
None.
