# Score-Then-Select Sparse Attention in Multi-Agent Communication

Code, kernels and measurement artifacts for *Score-Then-Select Sparse Attention
in Multi-Agent Communication: Operation-Count Limits and Measured Latency*
(manuscript in [`paper_submission_revision/`](paper_submission_revision/)).

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
   narrow heads, attention is memory-bound.

```
dense                 [ scores N²d ][ aggregate N²d ]
score-then-select     [ scores N²d ][top-k][ aggregate Nkd ]   ≥ half remains
select before scoring [ selector ][ scores Nkd ][ aggregate Nkd ]
```

## Central findings (one Blackwell GPU, N ≤ 512, fp32 unless stated)

- Composed from library operators, score-then-select attention is slower than
  the best dense implementation at every configuration measured (up to 37×).
- A fused score-then-select kernel is faster only at one retained peer (a
  degenerate attention), head widths 16–32, and without tensor cores; with
  TF32/bf16 dense attention the advantage is gone from 384 agents.
- Holding the retained set fixed: from 384 agents up, ≥90 % of the fused
  kernel's device time is finding the set; attending over it is cheap.
- Selecting before scoring makes the attention stage up to 16× faster than
  dense; end to end the selector decides, and only a Z-order selector reaches
  parity, at 512 agents on best-case timings.

These are statements about the measured envelope, not general laws; the
manuscript's Section 5 lists the limitations.

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
| `bash scripts/run_revision_campaign.sh` | every sweep the tables and figures read | ~3 h |
| `make equivalence` | `results/equivalence.json`: every timed configuration vs the reference | minutes |
| `make fixed-set` | `results/fixed_set.json`: fixed-retained-set sweep, eager and CUDA-graph, raw samples | ~2 min |

`make supplement` builds the anonymised review archive and refuses to finish if
any identifying string remains.

## Environment

NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120), shared with other
tenants, clocks unlocked; PyTorch 2.11 (CUDA 12.8), Triton 3.6, Python 3.11.
Pinned packages: `configs/environment.lock`. See [`HARDWARE.md`](HARDWARE.md)
for what a shared device means for the numbers: only ratios between
implementations timed in the same interleaved sweep should be compared.

## Layout

| Path | Contents |
|---|---|
| `src/kernels/` | Triton kernels: fused score-then-select (argmax, bisection, sort), dense, fixed-set attention; pre-score selectors |
| `scripts/bench_*.py` | Measurement harnesses (interleaved CUDA-event timing) |
| `scripts/make_revision_{tables,figures}.py` | Every table and figure, from `results/` |
| `scripts/check_revision_numbers.py`, `scripts/paper_numbers.py` | Number verification |
| `results/` | Measurement artifacts (JSON) |
| `audits/` | Revision audit trail: policy, format, math, citations, benchmarks, equivalence, reproducibility |
| `src/agents`, `src/training`, `configs/reduced` | The MAPPO learning study (Appendix D) |

## License

MIT (`LICENSE`).

## Citation

```bibtex
@misc{hamdan2026scorethenselect,
  author = {Mohammed Hamdan},
  title  = {Score-Then-Select Sparse Attention in Multi-Agent Communication:
            Operation-Count Limits and Measured Latency},
  year   = {2026},
  note   = {Under review at TMLR}
}
```
