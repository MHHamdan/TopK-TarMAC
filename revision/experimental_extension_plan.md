# Experimental extension plan

Written before the extensions were run, and amended below where a decision
changed after a result came in. Every entry records the question, the reviewer
concern it addresses, the code that had to be written, the cost, and the risk
that the result would move against the manuscript's central claim.

The central claim being protected is stated once, so that "threatens the central
claim" has a referent:

> In score-then-select top-k attention over N agents, the score matrix is
> irreducible, so the operation-count saving is bounded by 2x; and in the
> agent-count regime of multi-agent communication, that bounded saving does not
> appear as reduced wall-clock time.

An experiment threatens the claim if it can produce a measured configuration in
which a sparse implementation is faster than the best dense implementation by a
margin larger than measurement noise.

---

## Tier 1 — run

### E1. Explicit small-k sweep at absolute k

| | |
|---|---|
| **Question** | Does the gathered path approach or cross dense latency as k falls to 1? |
| **Reviewer concern** | "Incomplete exploration of very small k; the reviewer asks for k in {1,2,4}." |
| **Existing evidence** | `results/microbenchmark_ksweep.json` sweeps k as a *fraction* of N-1, so the smallest k reached at N=384 is 3 and at N=512 is 5. k in {1,2,4} is unmeasured at the two largest agent counts. |
| **Code change** | A `--k-list` of absolute k values, and a benchmark that times the attention core rather than the whole module so that shared projections do not dilute the ratio. New script `scripts/bench_attention_core.py`. |
| **Cost** | ~4 GPU-minutes for N x k x implementation at d=16; ~10 minutes with the head-width sweep. |
| **Risk to the claim** | Moderate. Extreme sparsity is where the gathered path is most favourable, so if it crosses anywhere it crosses here. |
| **Decision** | **RUN.** |

### E2. Backward pass and forward+backward

| | |
|---|---|
| **Question** | Does the ordering established for the forward pass hold for the backward pass and for the two together? |
| **Reviewer concern** | "Forward-pass-only benchmarking; absence of backward-pass timing." |
| **Existing evidence** | The harness already records a `forward_backward` field, and it is populated for every row of `results/microbenchmark_comm.json` and `results/microbenchmark_ksweep.json`. The manuscript never reports it. This is unreported evidence, not missing evidence. |
| **Code change** | None for the module-level measurement. The attention-core benchmark also times forward+backward for the differentiable implementations. |
| **Cost** | Included in E1. |
| **Risk to the claim** | Low-moderate. The backward pass of the gathered path involves a scatter-add, which is not obviously worse than dense; a reversal is conceivable. |
| **Decision** | **RUN**, and report backward separately from forward rather than folding it into a single number. The Triton kernels are excluded from this comparison: they are forward-only, and timing a hand-written kernel against an autograd graph would compare two different things. |

### E3. Per-stage arithmetic intensity and byte accounting

| | |
|---|---|
| **Question** | For each stage of each path, what is the analytic arithmetic intensity, and how close does the stage run to whichever machine limit binds it? |
| **Reviewer concern** | "Lack of explicit per-stage arithmetic-intensity / byte accounting." |
| **Existing evidence** | `results/component_decomposition.json` gives per-stage latency but no bytes, no intensity and no achieved rate. `results/roofline.json` gives whole-path intensity but not per stage. |
| **Code change** | New script `scripts/stage_accounting.py`: analytic FLOPs and bytes per stage alongside measured time, achieved FLOP/s, achieved GB/s, and the fraction of the binding roof attained. Peak arithmetic and peak bandwidth measured in the same process. |
| **Cost** | ~2 GPU-minutes. |
| **Risk to the claim** | Low. This is an explanation of an already-measured result. It could show that a stage the manuscript blames is in fact running near its limit, which would change the explanation but not the outcome. |
| **Decision** | **RUN.** |

### E4. End-to-end training-step timing

| | |
|---|---|
| **Question** | What share of a real MAPPO training iteration does the communication module occupy, and is the difference between implementations resolvable end to end? |
| **Reviewer concern** | "Absence of end-to-end training throughput evidence." |
| **Existing evidence** | None. The manuscript asserts that end-to-end wall-clock is environment-bound and declines to measure it. |
| **Code change** | New script `scripts/bench_training_step.py`: one complete MAPPO iteration with phase boundaries separated by device synchronisations, per implementation and per agent count. |
| **Cost** | ~15 minutes for three agent counts x five implementations. |
| **Risk to the claim** | Low. If the environment dominates, that is itself the finding and it bounds the practical stakes of the whole paper -- which should be said rather than assumed. |
| **Decision** | **RUN**, and report the null as a null: if the difference is below the iteration-to-iteration spread, say so and give the spread. |

---

## Tier 2 — run, with the outcome allowed to change the paper

### E5. A purpose-built fused sparse kernel

| | |
|---|---|
| **Question** | How much of the measured penalty is the composition of general-purpose operators rather than the algorithm? |
| **Reviewer concern** | "No purpose-built fused sparse kernel"; the manuscript's sparse path is criticised as a general-purpose PyTorch implementation. |
| **Existing evidence** | The manuscript bounds this analytically with a copy-equivalent selection counterfactual and states that a fused kernel is the most informative follow-up it did not run. |
| **Code change** | `src/kernels/fused_topk_attention.py`: three Triton kernels that fuse scoring, selection, gather and aggregation with no intermediate reaching global memory. (i) selection by k sequential argmax rounds, one program per query; (ii) the same with a tile of queries per program so the score product runs on tensor cores through `tl.dot`; (iii) selection by threshold bisection, whose cost does not grow with k. All three verified exact against the PyTorch gathered path to fp32 tolerance across the shape range. |
| **Cost** | Roughly a day of implementation; minutes to run. Triton 3.6.0 and sm_120 were already available. |
| **Risk to the claim** | **High, and it materialised.** A fused kernel could in principle overturn the wall-clock result. |
| **Decision** | **RUN.** |
| **Amendment after the result** | It partly did. The fused kernel removes most of the measured penalty and, at the smallest k and the smaller agent counts, is faster than fused dense attention. The manuscript's unconditional statement that the sparse path is always slower does not survive and must be replaced by the measured, conditional statement. The operation-count ceiling is untouched, and the realised speedup remains far below the 2x the ceiling permits. This is reported as a correction to the manuscript's own earlier claim, not buried. |

### E6. Head-width sensitivity to d = 512

| | |
|---|---|
| **Question** | The manuscript observes a trend towards parity as head width grows and stops at d = 256. Where does it go? |
| **Reviewer concern** | "Limited head-width range." |
| **Code change** | Included in the attention-core benchmark's `--d-list`. |
| **Cost** | ~5 GPU-minutes. |
| **Risk to the claim** | Moderate: the trend at N=512 was downward in d, and extrapolating it would reach parity somewhere. Measuring it is the only way to know whether it does. |
| **Decision** | **RUN**, to d = 512, and state the measured endpoint rather than extrapolating past it. |

### E7. Operator-level tuning of the gathered path

| | |
|---|---|
| **Question** | Is the PyTorch gathered path leaving easy performance on the table beyond the `sorted=False` fix already applied? |
| **Reviewer concern** | "Operator-level tuning: index dtype, layout, index_select vs gather, bmm vs einsum, compiled variants." |
| **Decision** | **OPTIONAL.** Superseded in value by E5: a fused kernel bounds the implementation question far more tightly than any operator substitution can, and E5 was completed. Run only if the schedule allows, and report only differences that survive repeated measurement. |

---

## Tier 3 — run

### E8. A pre-score selector, as an empirical test of Proposition 2

| | |
|---|---|
| **Question** | Proposition 2 says a selector that fixes the retained set before any score exists escapes the two-fold ceiling and can approach an N/k operation-count ratio. Does that ratio appear as time? |
| **Reviewer concern** | "Lack of an empirical escape case corresponding to Proposition 2." |
| **Existing evidence** | None. The proposition is stated and proved but never instantiated. |
| **Code change** | `src/kernels/prescore_selection.py`: two selectors that use agent positions rather than attention scores. (i) k spatially nearest peers by Euclidean distance -- the rule underlying distance-based neighbourhood communication, with a selector that is still pairwise but in the two-dimensional position space rather than the d-dimensional head space; (ii) agents ordered along a Morton curve, each retaining the k adjacent in that order -- a genuinely sub-quadratic O(N log N) selector satisfying the o(Nkd) condition of case (i) of the proposition. Both are followed by scoring the retained pairs only. |
| **Cost** | ~half a day of implementation; ~5 GPU-minutes to run. |
| **Risk to the claim** | Low, and a favourable result would strengthen rather than threaten the paper: the paper's constructive half predicts that this is where savings live. An *unfavourable* result -- a pre-score selector that also fails to convert its much larger operation-count saving into time -- would be a stronger and more interesting finding, and would have to be reported as such. |
| **Decision** | **RUN.** Frame as empirical validation of the escape mechanism, not as a new method. The selectors attend to a different peer set than top-k does, so this is a cost comparison and must be labelled as one. |

---

## Tier 4 — declined, with reasons

### E9. Energy and power measurement

**DO NOT RUN.** The GPUs in this host are shared with other tenants; both cards
carried between 59 and 66 GiB of another process's memory and non-zero
utilisation throughout this study. `nvidia-smi` reports board-level power for
the whole device, so any energy figure would attribute another tenant's work to
our kernels. `pynvml` is not installed and per-process energy accounting is not
available on this driver to an unprivileged account. A software proxy would
satisfy the reviewer's request while being wrong. The manuscript states that no
energy claim is made and why.

### E10. A second GPU generation

**DO NOT RUN — unavailable.** The host has two cards of the same model. The
older hardware that produced this project's earlier runs is no longer
accessible. The limitation is stated rather than papered over with a
same-model second card presented as architectural replication.

### E11. Alternative CPU allocators (jemalloc, tcmalloc, large pages)

**OPTIONAL, low value.** The manuscript already isolates the allocator effect by
pinning the glibc heap and counting minor faults inside the timed region, and
reports both the pinned and un-pinned runs. Swapping allocators would show that
a different allocator has a different page-return policy, which is not a fact
about attention. Declined in favour of E5 and E8.

### E12. A parameter-matched sham-communication arm in the training study

**DO NOT RUN.** It would sharpen the positive control, which is a precondition
of the secondary study rather than a result of the paper. The manuscript already
concedes that the controls establish that the communicating architecture helps
rather than isolating the channel. Spending the budget here rather than on E5
and E8 would improve the weakest part of the paper at the expense of the
strongest.

---

## What was run, and what it cost

| | Status | Artifact |
|---|---|---|
| E1 small-k, absolute | run | `results/attention_core_d16.json` |
| E2 backward | run | `results/attention_core_d16.json`, `results/microbenchmark_comm.json` |
| E3 stage accounting | run | `results/stage_accounting.json` |
| E4 training step | run | `results/training_step_timing.json` |
| E5 fused kernel | run | `results/attention_core_d16.json`, `src/kernels/fused_topk_attention.py` |
| E6 head width to 512 | run | `results/attention_core_dsweep.json` |
| E7 operator tuning | not run | superseded by E5 |
| E8 pre-score selector | run | `results/prescore.json`, `src/kernels/prescore_selection.py` |
| E9 energy | declined | shared device |
| E10 second GPU generation | declined | unavailable |
| E11 allocators | declined | not a fact about attention |
| E12 sham-communication arm | declined | budget allocated to E5 and E8 |
