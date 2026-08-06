# TopK-TarMAC: Sparse attention communication for cooperative MARL

Research codebase studying whether the TarMAC-style soft-attention
inter-agent communication channel can be sparsified to top-$k$
destinations — with $k$ adapted per agent per step — without sacrificing
task return in cooperative MARL.

## Contents

- A clean MAPPO baseline and an attention-communication module
  (dense, fixed top-$k$, and adaptive top-$k$ via a Gumbel-softmax gate)
  that all share the same actor/critic backbone.
- A reproducible experimental pipeline (configs, seed runner,
  bootstrap-CI tables, figures) for MPE `simple_spread` at
  $N \in \{3, 6, 12\}$ agents.
- A literature index covering cooperative MARL, agentic LLM-RL,
  distributed RL infrastructure, and large-scale MARL, generated from
  a single Python source.
- Decision logs, hardware notes, benchmark survey, and a published-work
  comparison so the absolute-return scale can be re-anchored under a
  different env version.

## Headline numbers

Soft-attention inter-agent communication beats the MAPPO baseline at
moderate agent count ($N{=}6$, Cohen's $d{=}1.06$ for dense attention,
95 % bootstrap CI $[+2.3, +26.5]$ on the mean delta). The adaptive-$k$
Gumbel-softmax variant fails to scale: at $N{=}12$ it underperforms
MAPPO with $d{=}-1.87$ and a bootstrap CI $[-119.7, -28.5]$ that
excludes zero on the negative side. The diagnostic is that the gate's
between-seed standard deviation grows from 4.6 at the baseline to 40.2
at $N{=}12$, overwhelming the 21–29 % FLOPs savings.

| $N$ | Random | MAPPO | + Dense Attn-Comm | + Adaptive TopK |
|-----|--------|-------|-------------------|-----------------|
| 3   | $-74.05 \pm 18.90$ | $-58.93 \pm 1.37$ ($n{=}5$) | $-60.27 \pm 3.53$ ($n{=}5$) | $-57.46 \pm 3.31$ ($n{=}5$) |
| 6   | $-238.43 \pm 54.41$ | $-225.04 \pm 13.93$ ($n{=}5$) | $-212.91 \pm 3.72$ ($n{=}5$) | $-223.22 \pm 20.01$ ($n{=}5$) |
| 12  | $-758.91 \pm 96.95$ | $-716.98 \pm 4.57$ ($n{=}3$) | $-730.44 \pm 16.85$ ($n{=}3$) | $-782.40 \pm 40.19$ ($n{=}3$) |

See [`COMPARISON.md`](COMPARISON.md) for context against EPyMARL
(Papoudakis 2021), TarMAC (Das 2019), and the MAPPO paper (Yu 2022),
with the env-version caveats that prevent direct numerical comparison
of absolute returns.

## Quick start

```bash
# 1. Create the venv and install pinned dependencies
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Sanity tests (env, agent, training-loop)
PYTHONPATH=. pytest -q tests/

# 3. Single-seed training (~30 min on RTX 2080 Ti)
PYTHONPATH=. python scripts/train.py \
  --config configs/mappo_mpe_n3_baseline.yaml \
  --seed 0 --run-name demo

# 4. Multi-seed run across multiple GPUs
PYTHONPATH=. python scripts/run_seeds.py \
  --config configs/mappo_mpe_n6_topk.yaml \
  --seeds 0 1 2 3 4 \
  --run-name mappo_mpe_n6_topk \
  --total-steps 800000 \
  --gpus 0 1 2 3 --parallel 4

# 5. Regenerate every figure / table from existing results
PYTHONPATH=. python scripts/measure_random.py
PYTHONPATH=. python scripts/measure_flops.py --runs mappo_mpe_n3_baseline mappo_mpe_n3_dense mappo_mpe_n3_topk mappo_mpe_n6_baseline mappo_mpe_n6_dense mappo_mpe_n6_topk mappo_mpe_n12_baseline mappo_mpe_n12_dense mappo_mpe_n12_topk
PYTHONPATH=. python scripts/make_tables.py --n-list 3 6 12 --ablation-n 6
PYTHONPATH=. python scripts/make_figures.py --n 6 --n-list 3 6 12 --flops-summary results/flops_summary.json

# 6. End-to-end reproduce script (skips training when final_eval.json exists)
bash scripts/reproduce.sh
```

## Repository layout

| Path | Purpose |
|------|---------|
| `src/agents/mappo.py` | `MAPPOAgent` = `SharedActor` + `CentralisedCritic` + optional `AttentionComm` (dense / topk / adaptive_topk) |
| `src/envs/mpe.py` | Stacked-tensor wrapper around `mpe2.simple_spread_v3` |
| `src/training/{rollout,train_mappo}.py` | GAE + clipped PPO + multi-env loop, per-seed CSV/checkpoint output |
| `src/utils/seeding.py` | Deterministic seeding helper |
| `scripts/train.py` | Single-seed entry-point taking a YAML config |
| `scripts/run_seeds.py` | Multi-seed parallel launcher; writes `results/MASTER_LOG.csv` |
| `scripts/measure_random.py` | Random-policy floor at each $N$ |
| `scripts/measure_flops.py` | Per-trained-run effective-$k$ and FLOPs ratio |
| `scripts/make_tables.py` | Headline + ablation $\TeX$ tables + JSON of stats |
| `scripts/make_figures.py` | Learning-curve, scaling, FLOPs-Pareto figures |
| `scripts/make_fig1_schematic.py` | Schematic of the TopK-TarMAC architecture |
| `scripts/build_literature.py` | Python source of `references.bib` and per-paper notes |
| `scripts/reproduce.sh` | End-to-end driver |
| `configs/*.yaml` | One config per (env, method, $N$) combination |
| `tests/` | Smoke + unit tests; `test_mappo.py` includes the reduce-to-baseline ablation |
| `literature/notes/*.md` | One file per cited paper; structured metadata |
| `literature/references.bib` | Generated bibliography |
| `literature/literature_review.md` | Synthesised review across 4 lanes + 12 open problems |
| `results/MASTER_LOG.csv` | One row per training run (timestamp, git SHA, config, seed, final eval, wall-time, GPU-hours) |
| `results/<run>/seed_<n>/{metrics.csv,config.json,final_eval.json,final.pt}` | Per-run outputs |
| `RESEARCH_LOG.md`, `DECISIONS.md`, `HARDWARE.md` | Process documentation |
| `benchmark_landscape.md`, `COMPARISON.md` | Benchmark survey + verified-prior-work comparison |

## Method

`MAPPOAgent` exposes three switches in `MAPPOConfig`:

| Flag | Behaviour |
|------|-----------|
| `use_comm = False` | Baseline MAPPO. No communication module instantiated. |
| `use_comm = True, attn_mode = "dense"` | TarMAC-style soft attention over all $N{-}1$ peers. $\mathcal{O}(N^2)$ message-aggregation cost. |
| `use_comm = True, attn_mode = "topk", topk = k` | Fixed-$k$ top-$k$ attention. Cost: $\mathcal{O}(Nk)$. |
| `use_comm = True, attn_mode = "adaptive_topk"` | Adaptive top-$k$ where $k_i$ is sampled per agent per step from a learned Gumbel-softmax over $\{1, \ldots, N{-}1\}$. Reduces to dense when the gate puts all mass on $N{-}1$; reduces to MAPPO when messages are zeroed. |

The reduce-to-baseline property is covered by
`tests/test_mappo.py::test_comm_zero_message_recovers_baseline`.

## Reproducibility

- Every reported number ties back to a row of `results/MASTER_LOG.csv`
  (git SHA, YAML config, seed, final eval mean/std, wall time, GPU-hours).
- All figures and tables are generated by tracked scripts under `scripts/`;
  no untracked notebooks produce reported numbers.
- `bash scripts/reproduce.sh` regenerates every artefact from the current
  commit; training is skipped when the corresponding `final_eval.json`
  already exists.

## License

Code is released under the MIT License (`LICENSE`). Bibliography entries
remain the property of the cited authors and venues.

## Citation

```bibtex
@misc{hamdan2026topktarmac,
  author = {Mohammed Hamdan},
  title  = {TopK-TarMAC: Adaptive Sparse Attention for Scalable Multi-Agent Communication},
  year   = {2026},
  note   = {Preprint}
}
```

## Author

Mohammed Hamdan — <mh2022ets@gmail.com>
