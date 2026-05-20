# CLAUDE.md — Re-read at every session start

## Mission

Run an unattended multi-day research project that produces a publishable workshop-paper
draft (10 pages, conference style) on the **agentic RL / multi-agent RL** track. The
contribution must land within: cooperative MARL, attention-based agent communication,
scalable training for large agent populations, agentic RL for LLM agents, or distributed
rollout efficiency. The human author is **Mohammed Hamdan**, preparing for an Agentic
RL Researcher role on a FANG distributed-computing team. Contributions should extend
MARL-IoTP work (attention communication / learned protocols / heterogeneous
coordination) rather than start a new line.

## Operating principles (non-negotiable)

1. **Honesty over impressiveness.** No fabricated citations or numbers. If a number is
   reported, it came from tracked code on a fixed seed. Negative results stay in.
2. **Reproducibility from line one.** Every experiment: config file + seed list + git
   SHA + logged command.
3. **Statistical rigor.** Min 3 seeds per condition, 5 for headline claims. Mean ± std,
   paired test for "A beats B" claims.
4. **Compute realism.** 4× RTX 2080 Ti with only ~4–6 GB free VRAM per device. Budget
   48 GPU-hours. No run exceeds the remaining budget by >20%.
5. **Small fast experiments first.** A 1-hour negative result > a 24-hour ambiguous
   result.
6. **Document as you go.** RESEARCH_LOG.md daily; DECISIONS.md for every non-obvious
   choice.

## Anti-patterns (forbidden)

- No citation without having fetched the paper.
- No reported numbers from untracked notebooks.
- No single-seed claims.
- No compute-unmatched comparisons without flagging.
- No pushes to remote.
- No `sudo` / global installs / files outside the project dir.
- No "tuning until reproduction works."
- No skipping limitations section.

## Forced checkpoints

- **Phase 3** — after 3 proposals are ranked, pause 24h for human review, then proceed.
- **Phase 6** — after pilot, decide proceed / pivot / fix-retry.

## Stop conditions

Pause and update STATUS.md if: compute >120 % budget, baseline reproduction fails twice,
two pilots fail, disk >80 % budget, sudo required, license click-through required,
remote push needed, or a result contradicts a major published claim.

## File map

```
agentic_marl/
├── CLAUDE.md, RESEARCH_LOG.md, DECISIONS.md, STATUS.md, HARDWARE.md, README.md
├── pyproject.toml
├── configs/, src/{envs,agents,training,eval,utils}/, scripts/, data/, results/
├── notebooks/  # exploratory only
├── literature/{papers,notes}, literature/references.bib, literature/literature_review.md
├── paper/{main.tex, sections/, figures/, tables/, references.bib}
├── proposals/  # filled in Phase 3
└── tests/
```

## Current phase

See STATUS.md.
