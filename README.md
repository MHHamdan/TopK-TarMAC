# SCALE: When Does Sparse Multi-Agent Attention Pay for Its Selector?

Code, kernels and measurement artifacts for *SCALE: When Does Sparse
Multi-Agent Attention Pay for Its Selector?* (manuscript in
[`paper_submission_revision/`](paper_submission_revision/), under review at
TMLR). **SCALE** — *Selector-Cost Accounting for Latency Evaluation* — charges
every peer-selection rule for the cost of deciding what to keep, and comes with
a measurement protocol for the microsecond-scale kernels that test it.

> The manuscript links this repository anonymously
> (`https://github.com/XXX/TopK-TarMAC`) during double-blind review; the
> repository will be made public after review, for reproducibility.

## What the paper studies

Sparse attention between agents is often justified by the operations it
removes: aggregate over $k$ peers instead of $N-1$ and the work shrinks by about
$N/k$. The paper asks when that saving becomes elapsed time, and separates three
costs an operation count conflates:

1. **What can be removed.** A rule that selects peers by their attention scores
   must form every score first, so it removes at most half of the forward-pass
   multiply–accumulates, $C_\text{dense}/C_\text{top-}k = 2N/(N+k) < 2$.
   Rules that select *before* scoring are bounded instead by the selector's own
   cost, which sorts them into four asymptotic classes.
2. **What the implementation executes.** Masking discarded weights to zero
   executes none of the saving; gathering the retained values executes it but
   writes and re-reads a compacted tensor, which is traffic-neutral only if
   $kd \le N$ (to leading order).
3. **What the hardware spends its time on.** At tens to hundreds of agents and
   narrow heads, attention composed from library operators is memory-bound,
   and inside a fused kernel the comparisons of selection can cost more than
   the arithmetic they remove.

```
dense                 [ scores N²d ][ aggregate N²d ]
score-then-select     [ scores N²d ][top-k][ aggregate Nkd ]   ≥ half remains
select before scoring [ selector ][ scores Nkd ][ aggregate Nkd ]
```

## Central findings (N ≤ 512, or 1024 for the streaming kernel)

Measured on four NVIDIA GPUs: RTX PRO 6000 Blackwell (sm_120), A100 (sm_80),
H100 and H200 (sm_90), every kernel's tiling searched on each GPU.

**Hold on all four GPUs**

- Composed from library operators, score-then-select attention is slower than
  the best dense implementation at every configuration measured (6.6× to 37×).
- Selection dominates: from 384 agents up, 85–97 % of a fused score-then-select
  kernel's time is finding the retained set.
- In fp32, a fused kernel is faster than dense attention with **one** retained
  peer (streaming kernel 0.46–0.68× dense).
- Below 100 agents, the number of kernels a step launches orders latency
  (rank correlation 0.91–0.95); the operation count does not (≤ 0.10).

**Depend on the GPU, the format and k**

- Two retained peers: faster everywhere on Blackwell and A100, at 7 of 8
  configurations on H100/H200. Four: within 15 % of parity on Blackwell,
  slower on the other three.
- bf16: near parity at one retained peer on Blackwell (0.85–1.06×), slower than
  dense on A100, H100 and H200 (1.03–1.52×).
- Selecting before scoring: attention over the chosen set is 4–17× faster than
  dense, but with its selector the Z-order path reaches parity only on
  Blackwell (0.85×; 1.12–1.45× elsewhere).
- Batch 256–4096 leaves the profitable window unchanged; batch 32 is less
  favourable.

The crossover is therefore set mainly by selection cost and moves with the
retained count, the numerical format and the GPU — not a universal sparse
speedup. The manuscript's Section 5 lists the limitations.

## Reproduce

```bash
uv venv .venv --python 3.11 && uv pip install -e ".[dev]"

make check     # tests, bibliography, and every number in the manuscript:
               #   regenerates the tables and fails if any is stale, re-derives
               #   prose numbers, and verifies the main-text manifest
               #   (scripts/paper_numbers.py -> results/paper_numbers.json)
make tables    # paper_submission_revision/tables/*.tex from results/*.json
make figures   # paper_submission_revision/figures/*.pdf from results/*.json
make paper     # tables + figures + compile main.pdf (tectonic)
```

GPU measurements (they overwrite `results/`):

| Command | Produces | Time |
|---|---|---|
| `bash scripts/run_revision_campaign.sh` | every sweep the tables and figures read (Blackwell) | ~3 h |
| `TAG=h100 bash scripts/run_portability_campaign.sh` | the headline sweeps on another GPU, tilings re-searched, into `results/portability/<tag>/` (run from a disposable copy) | ~2–4 h |
| `python scripts/ncu_counters.py` | Nsight Compute counters per kernel (needs counter permission) | ~5 min |
| `make equivalence` | `results/equivalence.json`: every timed configuration vs the reference | minutes |
| `make fixed-set` | `results/fixed_set.json`: fixed-retained-set sweep, eager and CUDA-graph, raw samples | ~2 min |

`make tables` also runs `scripts/summarise_portability.py`, which builds the
four-GPU tables and macros from `results/portability/`.

`make supplement` builds the anonymised review archive and refuses to finish if
any identifying string remains.

## Environment

- NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120), shared with other
  tenants, clocks unlocked; PyTorch 2.11 (CUDA 12.8), Triton 3.6, Python 3.11.
  See [`HARDWARE.md`](HARDWARE.md): only ratios between implementations timed
  in the same interleaved sweep should be compared.
- A100-SXM4-40GB, H100 80GB HBM3 SXM, H200: one GPU allocated exclusively per
  job; same PyTorch/Triton versions (the A100 build uses CUDA 12.9). Device
  descriptions are in `results/portability/<gpu>/device.csv`.
- Pinned packages: `configs/environment.lock`.

## Layout

| Path | Contents |
|---|---|
| `src/kernels/` | Triton kernels: fused score-then-select (argmax, bisection, sort), dense, fixed-set attention; pre-score selectors |
| `scripts/bench_*.py` | Measurement harnesses (interleaved CUDA-event timing) |
| `scripts/make_revision_{tables,figures}.py` | Every table and figure, from `results/` |
| `scripts/make_fig0_methodology.py` | Figure 1, the SCALE schematic |
| `scripts/run_portability_campaign.sh`, `scripts/summarise_portability.py` | The four-GPU replication and its tables |
| `scripts/ncu_counters.py` | Nsight Compute counters per kernel |
| `scripts/check_revision_numbers.py`, `scripts/paper_numbers.py` | Number verification |
| `results/` | Measurement artifacts (JSON); `results/portability/<gpu>/` for A100, H100, H200 |
| `audits/` | Revision audit trail: policy, format, math, citations, benchmarks, equivalence, reproducibility |
| `src/agents`, `src/training`, `configs/reduced` | The MAPPO learning study (Appendix D) |

## License

MIT (`LICENSE`).

## Citation

```bibtex
@misc{hamdan2026scale,
  author = {Mohammed Hamdan},
  title  = {{SCALE}: When Does Sparse Multi-Agent Attention Pay for Its Selector?},
  year   = {2026},
  note   = {Under review at TMLR}
}
```
