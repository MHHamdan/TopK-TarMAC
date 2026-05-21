# STATUS

**Active phase:** Phase 9 — Paper draft (DONE, polishing)
**Last update:** 2026-05-21

## Phases

| Phase | Status |
|-------|--------|
| 0 — Setup | ✅ |
| 1 — Literature review (64 papers) | ✅ |
| 2 — Benchmark survey | ✅ |
| 3 — Contribution selection (FORCED CHECKPOINT) | ✅ |
| 4 — Baseline reproduction (5 seeds N=3, dense N=3, topk N=3) | ✅ |
| 5 — Method implementation | ✅ |
| 6 — Pilot (FORCED CHECKPOINT) | ✅ |
| 7 — Full sweep (N=3, 6, 12) | ✅ |
| 8 — Figures + tables | ✅ |
| 9 — Paper draft (8 pages, tectonic build green) | ✅ |

## Headline numbers (2026-05-21)

| $N$ | MAPPO | + Dense Attn-Comm | + Adaptive TopK | $d$(dense) | $d$(topk) | 95% bootstrap CI (topk-baseline) |
|----|-------|-------------------|-----------------|------------|-----------|----------------------------------|
| 3  | -58.93 ± 1.37 ($n{=}5$) | -60.27 ± 3.53 ($n{=}5$) | -57.46 ± 3.31 ($n{=}5$) | -0.45 | +0.52 | [-1.6, +4.7] |
| 6  | -225.04 ± 13.93 ($n{=}5$) | -212.91 ± 3.72 ($n{=}5$) | -223.22 ± 20.01 ($n{=}5$) | +1.06 (CI excl. 0) | +0.09 | [-19.6, +22.5] |
| 12 | -716.98 ± 4.57 ($n{=}3$) | -730.44 ± 16.85 ($n{=}3$) | -782.40 ± 40.19 ($n{=}3$) | -0.89 | -1.87 (CI excl. 0, neg.) | [-119.7, -28.5] |

Headline story: Dense attention helps at moderate $N$ (positive finding at $N{=}6$);
adaptive Gumbel-softmax top-$k$ gating fails to scale ($N{=}12$ result is significantly
*worse* than MAPPO with bootstrap CI excluding 0). FLOPs savings of 21–29 % were
achieved but do not compensate for the gate's seed-variance growth.

## Artifact summary
- `paper/main.pdf` — 8 pages, tectonic-built.
- `paper/REVIEW_CHECKLIST.md` — every checkpoint ticked.
- `results/MASTER_LOG.csv` — every run logged with git SHA + wall time + GPU-hours.
- `bash scripts/reproduce.sh` regenerates every figure, table, and the PDF.

## Compute usage
- ~20 GPU-h cumulative across the run (vs 48 GPU-h budget). Plenty of room
  for the Lagrangian-budget follow-up.

## Blockers
None.
