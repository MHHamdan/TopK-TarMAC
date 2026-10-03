# Recent related-work audit (to 2026-10-03)

Manuscript: *Score-Then-Select Sparse Attention in Multi-Agent Communication: Operation-Count Limits and Measured Latency* (`paper_submission_revision/`). No manuscript files were edited.

## Method

Read `sections/00_abstract.tex`, `01_intro.tex`, `02_related.tex`, `03_analysis.tex`, the residency caveat in `04_measurement.tex:147-150` and `12_app_measurement.tex:176-186`, the selector description at `04_measurement.tex:205-218`, and `references.bib`, which has 70 entries including `gupta2021topk`, `zhao2019explicit`, `yuan2025nsa`, `lai2026msa`, `dong2025flexattention` and `dehghani2022misnomer`.

**Queries run** (WebSearch, then WebFetch on the arXiv, venue or DOI page for every candidate kept):
1. DeepSeek Sparse Attention lightning indexer top-k DeepSeek-V3.2
2. Loki low-rank keys sparse attention; SparQ attention; Quest query-aware sparsity
3. "top-k attention" still requires full QK^T / no FLOP reduction
4. "top-k" attention upper bound speedup 2x, QK^T must be computed; "at most half" FLOPs post-hoc top-k
5. fused exact top-k attention kernel, streaming running top-k without materialising scores
6. Parallel top-k on GPU (AIR top-K); RadiK; Blackwell top-k for sparse attention
7. SeerAttention; MoBA; SpargeAttn; Twilight; HashAttention; MagicPIG; The Sparse Frontier
8. Energon; DOTA; SpAtten (hardware dynamic sparse attention)
9. Block-Sparse FlashAttention; Sparse Flash Attention (Pagliardini); FLOP-vs-wall-clock sparse attention
10. CommFormer; ExpoComm; scalable MARL communication 2025-2026; sparse MARL communication with wall-clock time
11. Long Range Arena; Efficient Transformers survey; computational complexity of self-attention (SETH)
12. ZETA (Z-order top-k attention); MoSA; SPARSEK

Full PDFs were extracted with `pypdf` where the abstract did not settle the question: DeepSeek-V3.2 §2.1 and §2.3, the DFSS Proposition 4.3 and its proof (App. A.3), the CommFormer and ExpoComm experimental sections, and Gupta et al. (Limitations). For BSFA and ZETA the arXiv HTML was read.

**Exclusions.** Search snippets surfaced several candidates that were not opened or not checked closely enough to cite: SCOPE, Sol-Attn, LongCat Sparse Attention, FlashMemory-DeepSeek-V4, Block Sparse Attention with Log-Linear Complexity, UNIQUE, ClusterAttention and AttnFuse. They are excluded.

## Table

Decision key: **YES** means the paper must be engaged, because it bears on a claim or the novelty framing. **PARTIAL** means cite it in one clause. **NO** means do not add it.

| # | Paper | Year / venue | URL | What it shows | Changes our claim? | Recommended action / suggested sentence |
|---|---|---|---|---|---|---|
| 1 | Chen, Qu, Quan, Liu, Ding, Xie, *Dynamic N:M Fine-Grained Structured Sparse Attention Mechanism* (DFSS) | PPoPP 2023, pp. 369-379 | https://doi.org/10.1145/3572848.3577500 (arXiv 2203.00091) | Prop. 4.3 gives a memory-access upper bound on the speedup of top-K attention at density s, even with an **oracle mask at zero selection cost**: Speedup < (4d+3T)/(2d+T+(d+2T+dT)s). At d=64 and T=128 this needs s<4.5% to exceed 1. The text says the bound is not strict because selection overhead is excluded. It also notes that Explicit Sparse Transformer throughput is lower than dense, and that the gathered aggregation loses 2-D tiling. | **YES, novelty.** This is a prior statement that top-k attention, which keeps the full QK^T, has a constant-factor speedup ceiling. It is stated in bytes and tiles, not MACs, and it excludes the selector. Proposition 1 is therefore not the first such observation. What remains ours: the closed-form MAC ceiling 2N/(N+k), the backward bound 6N/(N+5k), the selector-cost taxonomy, the per-pair byte criterion kd≤N, and the measurements. | In §2 "Efficient and sparse attention", after the Gupta sentence: *"That top-$k$ attention cannot approach $N/k$ has been noted before: \citet{chen2023dfss} bound its speedup by a constant in a memory-access model even when the mask is supplied at no cost, and observe that the gathered aggregation forfeits two-dimensional tiling. Proposition~\ref{prop:ceiling} states the same limit as a closed-form operation count, extends it to the backward pass, and places it as one end of a selector-cost axis."* Also soften the intro contribution bullet from "a cost model ... any score-then-select rule removes at most half" to "formalises, as an operation count, ..." |
| 2 | Ohayon, Lamprecht, Hubara, Cohen, Soudry, Elata, *Block Sparse Flash Attention* (BSFA) | arXiv 2512.07011 (v2 29 Sep 2026: "Accepted to NeurIPS 2026") | https://arxiv.org/abs/2512.07011 | Computes exact QK, then skips value blocks whose block-max is below a calibrated threshold, inside FlashAttention-2's online softmax. States that "the skipped PV multiplication and value loads constitute approximately 50% of attention FLOPs and 50% of HBM bandwidth". Measured 1.10-1.24x end to end and up to 1.38x kernel-only (A100/H100, 4K-128K tokens). | **YES, novelty and corroboration.** A contemporaneous sequence-model paper states the half-of-FLOPs observation for exact-score selection and measures speedups well inside a 2x envelope. It also corroborates our threshold kernel, which is fused, streams and keeps the dense aggregation. It does not state a bound as a proposition, and it has no backward pass and no selector taxonomy. | §2, same paragraph: *"Block-Sparse FlashAttention computes every score and skips only value blocks, noting that these are about half of the attention FLOPs, and measures attention speedups of at most $1.38\times$ at 64K tokens \citep{ohayon2026bsfa}; Proposition~\ref{prop:ceiling} is the per-pair form of that limit."* In §4 (threshold kernel), one clause: "a block-level analogue of our threshold kernel \citep{ohayon2026bsfa}". |
| 3 | DeepSeek-AI, *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models* (DSA, lightning indexer) | arXiv 2512.02556, Dec 2025 | https://arxiv.org/abs/2512.02556 | Lightning indexer I_{t,s}=Σ_j w·ReLU(q^I·k^I), few heads, FP8, followed by top-2048 per query. §2.3: "Although the lightning indexer still has a complexity of O(L²), it requires much less computation compared with MLA". | **YES.** DSA is the canonical production instance of **case (b)**: an all-pairs selector in a cheap feature space, where S → 2d/(γp) is a constant. DSA's own text concedes that the indexer stays quadratic. This is direct support for Corollary 1(b) and should be the flagship example. It does not challenge the claim: its gains come from k=2048 ≪ L at 128K tokens plus FP8 and few-head indexing (small γp/d). | In §3.3 after Corollary 1 and in §2: *"DeepSeek Sparse Attention is case~(b) at scale: its lightning indexer scores every query--key pair with a few FP8 heads before selecting 2048 keys, and its authors note that the indexer remains $O(L^2)$ \citep{deepseek2025v32}; the saving is bounded by the ratio of attention width to indexer width, not by $N/k$."* |
| 4 | Zeng, Huang, Lu, Xu, Chen, Ling, Wang, *ZETA: Leveraging Z-order Curves for Efficient Top-k Attention* | ICLR 2025 | https://arxiv.org/abs/2501.14577 | Projects queries and keys to d_k=3, maps them to 1-D by Z-order, radix-sorts, and takes a window of K keys around each query's insertion point. Exact scores only for candidates; O(N log N); Triton kernel faster than FlashAttention at 8K+ tokens (608 ms vs 7044 ms fwd+bwd at 64K). | **YES.** Our pre-score Z-order selector (Morton order, k adjacent in curve order, `04_measurement.tex:212-213`) is the same mechanism, applied to 2-D agent positions instead of projected keys. ZETA is prior art for the selector, and it is also an independent case-(c) instance that beats dense at long N. The novelty framing must not imply the Z-order window is ours. | `04_measurement.tex:212`: *"\emph{Z-order} orders agents along a Morton curve and keeps the $k$ adjacent in that order, as ZETA does for low-dimensional projections of queries and keys \citep{zeng2025zeta}; its sort is $O(N\log N)$, case~(c)."* Add ZETA to the case-(c) examples in §3.3. |
| 5 | Jaber, Jaber, *StreamIndex: Memory-Bounded Compressed Sparse Attention via Streaming Top-k* | arXiv 2605.02568, May 2026 | https://arxiv.org/abs/2605.02568 | Triton kernels for the DeepSeek-V4 indexer: a fused tile score kernel plus a chunked partition-merge driver that keeps a **running top-k buffer per query** and never materialises the score tensor. Still computes all indexer scores. Extends the feasible length from 65,536 to 1,048,576 tokens on one H200 (6.21 GB peak). | **YES, for the residency caveat.** `04_measurement.tex:147-150` and `12_app_measurement.tex:182-186` say "a streaming exact top-k ... might do better / would relax it". A streaming exact top-k with a running candidate set now exists. It was evaluated for memory at 10^5-10^6 tokens, not for latency at N≤512. The caveat remains correct but should cite it. It also confirms that streaming does not remove the full-score cost, which supports Prop. 1. | Replace "a streaming exact top-$k$ that keeps a running candidate set would relax it" with: *"a streaming exact top-$k$ that keeps a running candidate set relaxes it, as StreamIndex does for long-context indexers \citep{jaber2026streamindex}, still computing every score; whether it is faster than dense attention at $N\le512$ we did not test."* |
| 6 | Gupta, Dar, Goodman, Ciprut, Berant, *Memory-efficient Transformers via Top-k Attention* (already cited as `gupta2021topk`) | SustaiNLP 2021 | https://arxiv.org/abs/2106.06899 | Limitations: "As our method requires computing inner products of all queries and keys, it has a quadratic compute requirement"; "top-k attention targets memory consumption but not runtime". | **PARTIAL (rewording only).** This is the earliest explicit statement in our bib that score-then-select keeps the quadratic score term. The manuscript currently cites it only for "memory efficiency as the stated benefit". | Change to: *"...compute each query's exact top-$k$ over all keys in chunks and note that the compute therefore remains quadratic \citep{gupta2021topk}."* |
| 7 | Singhania, Singh, He, Feizi, Bhatele, *Loki: Low-rank Keys for Efficient Sparse Attention* | NeurIPS 2024 | https://arxiv.org/abs/2406.02542 | Approximate scores for all keys in a PCA subspace (25-50% of d), then exact attention over the top-k (12.5-25%). | **PARTIAL.** A textbook case (b) with p = d/4..d/2, so 2d/(γp) is at most about 4-8 before the top-k cost. | One clause in §3.3 alongside DSA: "...or in a low-rank key subspace \citep{singhania2024loki}". |
| 8 | Ribar, Chelombiev, Hudlass-Galley, Blake, Luschi, Orr, *SparQ Attention: Bandwidth-Efficient LLM Inference* | ICML 2024 | https://arxiv.org/abs/2312.04985 | Approximate scores over all positions using the r largest query components, top-k, then exact attention. Explicit transfer model M = S·r + 2k·d_h + 4d_h vs 2S·d_h; up to 8x transfer reduction. | **PARTIAL.** A case-(b) selector, and prior art for **byte accounting** of selection. Our kd≤N criterion is a different statement (per-pair compaction of all N queries vs a score row), but a reviewer will ask about the relation. | In §3.4 after the byte criterion: *"SparQ Attention applies a transfer count of this kind to single-query decoding, where it favours selection \citep{ribar2024sparq}; with all $N$ queries active, the compacted operand is what decides."* |
| 9 | Tang, Zhao, Zhu, Xiao, Kasikci, Han, *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference* | ICML 2024 | https://arxiv.org/abs/2406.10774 | Per-page min/max key metadata gives an upper-bound score per page; top-K pages are loaded. 2.23x self-attention speedup. | **PARTIAL.** A block-granular pre-score selector whose cost is N/B pages per query, i.e. block-amortised. It is one more instance of the NSA argument already made in §2. | Add to the NSA/MSA citation: "...amortises the selector over blocks \citep{yuan2025nsa,lai2026msa,tang2024quest}". |
| 10 | Zhang, Xiang, Huang, Wei, Xi, Zhu, Chen, *SpargeAttention: Accurate and Training-free Sparse Attention Accelerating Any Model Inference* | ICML 2025 | https://arxiv.org/abs/2502.18137 | Stage 1 is a pre-score block predictor (mean-pooled Q/K blocks, a compressed QK^T over all block pairs, a self-similarity guard). Stage 2 skips P̃V **after** QK is computed, when the row-max gap is below λ. 1.83x on Mochi and 1.73x end to end on Llama-3.1 at 128K. | **PARTIAL.** Splits cleanly along our axis: stage 2 is score-then-select and capped by Prop. 1; stage 1 is case (b) at block granularity. A good one-clause illustration that the taxonomy applies to deployed systems. | §2: *"SpargeAttn combines both kinds, a block predictor that runs before scoring and a filter that skips only the value product after it \citep{zhang2025spargeattn}."* |
| 11 | Gao, Zeng, Du, Cao, Zhou, Qi, Lai, So, Cao, Yang, Yang, *SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs* | arXiv 2410.13276, Oct 2024 | https://arxiv.org/abs/2410.13276 | Learned gate on pooled Q/K blocks predicts block sparsity, followed by block-sparse FlashAttention. | **PARTIAL (optional).** Block-level case (b). Redundant with SpargeAttn and MoBA; cite only if the paragraph lists examples. | Optional: add to the block-amortised list in §2. |
| 12 | Lu, Jiang, Liu, Du, ... Qiu, *MoBA: Mixture of Block Attention for Long-Context LLMs* | arXiv 2502.13189, Feb 2025 | https://arxiv.org/abs/2502.13189 | MoE-style gate routes each query to the top-k KV blocks (block 4096, top-12, up to 95.31% sparsity). Deployed in Kimi. | **PARTIAL (optional).** Same class as NSA: block-amortised pre-score selection. | Optional: add to the NSA/MSA citation list. |
| 13 | Zhang, Naruse, Li, Wang, *Parallel Top-K Algorithms on GPU: A Comprehensive Study and New Methods* (AIR top-K, GridSelect) | SC 2023 | https://doi.org/10.1145/3581784.3607062 | Iteration-fused adaptive radix top-K and a shared-queue GridSelect; large speedups over earlier GPU top-K. | **PARTIAL.** Our "bisection on the score's bit pattern" (`04_measurement.tex:144`, App. selection) is a radix-select variant. A systems reviewer will expect the standard GPU top-k references. It does not change the result: these are standalone operators over a materialised array, not row-local selection inside an attention kernel. | App. selection: *"Bisection on the bit pattern is a row-local radix select; standalone GPU radix selects \citep{zhang2023airtopk,li2024radik} operate on materialised arrays and are not drop-in inside a fused attention kernel."* |
| 14 | Li, Zhou, Zhang, Wei, Li, Chen, *RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection* | ICS 2024 | https://doi.org/10.1145/3650200.3656596 (arXiv 2501.14336) | Radix top-k that scales with k, length and batch; up to 2.5x / 4.8x over the previous state of the art. | **PARTIAL.** Same role as #13. | Cite together with #13. |
| 15 | Cheng, Zhao, Liu, Li, Qiao, Duan, Chen, Chen, Darvish Rouhani, Yang, *Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell via Temporal Correlation* | arXiv 2604.22312, Apr 2026 | https://arxiv.org/abs/2604.22312 | On Blackwell, top-K selection is "a meaningful latency bottleneck" in DSA-style decoding even after the indexer and attention are optimised. Bit-exact; 1.88x over the production radix-select kernel. | **PARTIAL.** Independent Blackwell evidence for our finding that ranking costs more than the aggregation it saves (§4 Q3). | One clause in §4 or the selection appendix: *"Selection is likewise a measurable share of sparse-attention decode latency on the same GPU generation \citep{cheng2026gvr}."* |
| 16 | Liu, Xue, Tang, Shao, Cheng, Ren, Jiang, Ming, Ding, Lin, Wei, Liu, Liu, Zhu, *Sample-Guided Exact Top-K Selection for Long-Context Sparse Attention* | arXiv 2609.08450, Sep 2026 | https://arxiv.org/abs/2609.08450 | Strided sampling proposes a threshold; a **mandatory complete-row pass** certifies it, then exact refinement. 1.55x geomean over baselines (Tencent HPC-Ops). | **NO (optional with #15).** Confirms that exact selection must read the whole row; adds nothing beyond #15. | Do not add, or append it to the #15 citation. |
| 17 | Hu, Shen, Zhang, Tao, *Learning Multi-Agent Communication from Graph Modeling Perspective* (CommFormer) | ICLR 2024 | https://arxiv.org/abs/2405.08550 | Learns an adjacency α by bi-level optimisation; α is applied as a **mask on attention scores** (plus edge embeddings), with sparsity S=0.4 in all tasks. Motivated by cost ("imposing significant computational complexity"). | **PARTIAL.** The most-cited 2024 MARL learned-graph method. In our taxonomy its selector is free at inference (learned, fixed), but it is executed as **masked** attention, so no arithmetic is removed. This fits the §2 argument exactly and closes a 2024 gap. | §2 MARL paragraph: *"CommFormer learns a sparse communication graph and applies it as a mask on the attention scores \citep{hu2024commformer}, a free selector executed as masked attention, so the removed interactions are still computed."* |
| 18 | Li, Wang, Bai, Zhang, *Exponential Topology-enabled Scalable Communication in Multi-agent Reinforcement Learning* (ExpoComm) | ICLR 2025 | https://arxiv.org/abs/2502.19717 | Fixed static or one-peer exponential graphs (K=⌈log2 N⌉ or 1 neighbours) with attention or RNN aggregation. MAgent up to 64 agents, IMP up to 100. Compares against a DGN+TarMAC top-K nearest-neighbour baseline. Excludes runs above 40 GB GPU memory. | **PARTIAL.** The current "scalable communication" reference point. It is a case-(d) free selector with k=log2 N. Its agent counts (≤100) support our N≤512 scope. | §2: add to the free-selector sentence: *"...as are fixed exponential topologies that connect each agent to $\lceil\log_2 N\rceil$ peers \citep{li2025expocomm}."* Consider updating "two to roughly thirty agents, DGN reaching the low hundreds" to mention ExpoComm's 64-100. |
| 19 | Duman Keles, Wijewardena, Hegde, *On the Computational Complexity of Self-Attention* | ALT 2023, PMLR 201 | https://proceedings.mlr.press/v201/duman-keles23a.html (arXiv 2209.04881) | Self-attention is necessarily quadratic unless SETH is false, even approximately for several variants. | **PARTIAL.** Supports the point that sub-quadratic pre-score selectors (cases c and d) must compute a different function from exact top-k. The manuscript already says this for its own selectors ("attend to different peers than top-k"); one clause gives it a formal basis. | §3.3 after Prop. 2: *"Such selectors compute a different function from exact top-$k$; exact attention admits no generally sub-quadratic algorithm under SETH \citep{keles2023complexity}."* |
| 20 | Qu, Liu, Tu, Chen, Ding, Xie, *DOTA: Detect and Omit Weak Attentions for Scalable Transformer Acceleration* | ASPLOS 2022, pp. 14-26 | https://doi.org/10.1145/3503222.3507738 | Jointly trained lightweight low-rank detector predicts weak connections before full attention; accelerator with 152.6x / 4.5x speedup over GPU / custom hardware. | **PARTIAL (optional).** A hardware case-(b) selector that predates DSA. Useful if §2 wants to show that the case-(b) idea is not new and the constant is what matters. | Optional: "...low-rank or low-precision detectors in accelerators \citep{qu2022dota}". |
| 21 | Zhou, Liu, Gu, Sun, *Energon: Toward Efficient Acceleration of Transformers Using Dynamic Sparse Attention* | IEEE TCAD 42(1):136-149, 2023 (online 2022) | https://arxiv.org/abs/2110.09310 | Mixed-precision multi-round filtering over all query-key pairs at low bit-width, then full-precision attention on survivors. | **NO (optional, same role as #20).** | Cite only together with DOTA if that clause is added. |
| 22 | Pagliardini, Paliotta, Jaggi, Fleuret, *Fast Attention Over Long Sequences With Dynamic Sparse Flash Attention* | NeurIPS 2023 | https://arxiv.org/abs/2306.01160 | Dynamic sparse patterns (QK-dropping, hashing) "often result in runtimes significantly slower than computing full attention" unless fused into FlashAttention; fused, 2.0x/3.3x training speedup at 8K/16K. | **PARTIAL.** Prior evidence of exactly our composed-vs-fused finding in sequence models. | §2 "Hardware-aware execution": *"Dynamic sparsity composed from general operators is slower than FlashAttention unless fused into it \citep{pagliardini2023sparseflash}."* |
| 23 | Desai, Yang, Cuadron, Zaharia, Gonzalez, Stoica, *HashAttention: Semantic Sparsity for Faster Inference* | ICML 2025 | https://arxiv.org/abs/2412.14468 | Learned Hamming-space codes for all keys, then top-k; up to 4.3x attention latency reduction at 32x sparsity (A100). | **NO.** Another case-(b) instance in decode; DSA and Loki already cover it. | Don't add. |
| 24 | Lin, Tang, Yang, Wang, Tang, Tian, Stoica, Han, Gao, *Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning* | NeurIPS 2025 | https://arxiv.org/abs/2502.02770 | Top-p budget refinement on top of a base selector; 15.4x self-attention speedup in long-context decode. | **NO.** Changes the budget rule, not the cost structure; decode regime. | Don't add. |
| 25 | Chen, Sadhukhan, Ye, Zhou, Zhang, Nolte, Tian, Douze, Bottou, Jia, Chen, *MagicPIG: LSH Sampling for Efficient LLM Generation* | ICLR 2025 | https://arxiv.org/abs/2410.16179 | Argues top-k is a biased estimator and uses LSH sampling with CPU offload. | **NO.** About quality and heterogeneous systems; the hashing case is already covered by Reformer. | Don't add. |
| 26 | Nawrot et al., *The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs* | ACL Findings 2026 | https://arxiv.org/abs/2504.17768 | IsoFLOPS study: at very long sequences, larger and highly sparse models beat smaller dense ones. | **NO.** Accuracy and FLOP trade-off at LLM scale; no kernel latency claims bearing on ours. | Don't add. |
| 27 | Tay, Dehghani, Abnar, Shen, Bahri, Pham, Rao, Yang, Ruder, Metzler, *Long Range Arena* / Tay, Dehghani, Bahri, Metzler, *Efficient Transformers: A Survey* | ICLR 2021 / ACM CSUR 55(6), 2022 | https://arxiv.org/abs/2011.04006 ; https://doi.org/10.1145/3530811 | Benchmark and taxonomy of efficient transformers. | **NO.** The FLOPs-vs-time point is already carried by `dehghani2022misnomer`, `ivanov2021datamovement` and `dao2022flashattention`. | Don't add. |
| 28 | Xiu, Tang, Rong, Yan, Wang, Lu, Cai, *A Preliminary Study on the Promises and Challenges of Native Top-k Sparse Attention* | arXiv 2512.03494, Dec 2025 | https://arxiv.org/abs/2512.03494 | Exact top-k decoding matches full attention in quality; notes "the high computational complexity of exact Top-k Attention". | **NO.** Quality study; no bound. | Don't add. |
| 29 | Yao et al., *Recall Before You Rank (ReTopK)* | arXiv 2607.27692, Jul 2026 | https://arxiv.org/abs/2607.27692 | States that exact top-k "still requires scoring the current query against the full KV cache ... leaving selector cost linear in context length"; reuses past supports. | **NO.** Restates the premise in the decode setting; no bound. | Don't add. |
| 30 | Tan et al., *DSV* (ASPLOS 2026) / Piękos, Csordás, Schmidhuber, *MoSA* (arXiv 2505.00315) / Lou, Jia, Zheng, Tu, *SPARSEK* (arXiv 2406.16747) / Guo, Shi, Fan, *Email mechanism* (IJCAI 2023) / Vora et al., *SCoUT* (arXiv 2603.04833) | various | https://arxiv.org/abs/2502.07590 ; https://arxiv.org/abs/2505.00315 ; https://arxiv.org/abs/2406.16747 ; https://arxiv.org/abs/2301.01919 ; https://arxiv.org/abs/2603.04833 | Low-rank selection for video DiT; expert-choice token routing per head (case d); scoring network plus differentiable top-k (case d); MARL message chains; Gumbel group-based recipient selection. | **NO.** Each fits an existing case and adds no new constraint. | Don't add. |

Already cited and verified to exist: `lai2026msa` (arXiv 2606.13392, MiniMax Sparse Attention: Index Branch scores KV blocks, then top-k) and `yuan2025nsa`. No change is needed for either.

## Verdict on the novelty framing

1. **The 2x ceiling is not new as an observation.**
   - DFSS (PPoPP 2023, Prop. 4.3) already bounds the speedup of top-K attention by a constant that holds with an oracle mask, in a memory-access model, and explicitly excludes selection overhead.
   - BSFA (NeurIPS 2026) states that the value side is about 50% of attention FLOPs when all scores are computed.
   - Gupta et al. (2021) state that compute stays quadratic.

   The current intro wording ("any score-then-select rule removes at most half"), presented as a contribution, will read as overclaiming to a reviewer who knows DFSS. Reframe Proposition 1 as a **formalisation** in MAC terms that is exact in N and k, with the backward-pass extension 6N/(N+5k), and as the anchor (case a) of the selector-cost axis. The novelty is the **axis and classification** (Prop. 2 and Cor. 1), the **per-pair byte criterion**, and the **measured envelope at N≤512 with narrow heads**. None of the 30 papers checked gives the four-class selector-cost corollary or a kd≤N compaction criterion.

2. **Case (b) has a flagship modern instance.** DeepSeek Sparse Attention's lightning indexer is an all-pairs, cheap-feature-space selector whose authors concede O(L²). Citing it strengthens Corollary 1(b) and shows the taxonomy describes production systems. It does not contradict any claim.

3. **The Z-order selector is prior art.** ZETA (ICLR 2025) uses the same Morton-order window for top-k attention, including a Triton kernel faster than FlashAttention at long N. It must be cited where the selector is introduced. The manuscript's claim concerns cost at N≤512 on agent positions, which remains ours.

4. **The "residency is ours" caveat needs a citation, not a retraction.** StreamIndex (2026) implements the streaming exact top-k with a running buffer that the caveat anticipates. It was evaluated for memory at 10^5-10^6 tokens and still computes every score, which is consistent with Prop. 1. The N≤512 latency question remains untested.

5. **No checked paper contradicts a measured claim.** BSFA (≤1.38x), GVR (top-K is a bottleneck on Blackwell) and Pagliardini et al. (composed dynamic sparsity is slower than FlashAttention) are independent corroboration.

---

## Appendix: BibTeX for recommended additions (YES and PARTIAL)

```bibtex
@inproceedings{chen2023dfss,
  title     = {Dynamic {N:M} Fine-Grained Structured Sparse Attention Mechanism},
  author    = {Zhaodong Chen and Zheng Qu and Yuying Quan and Liu Liu and
               Yufei Ding and Yuan Xie},
  booktitle = {Proceedings of the 28th ACM SIGPLAN Annual Symposium on
               Principles and Practice of Parallel Programming (PPoPP)},
  pages     = {369--379},
  year      = {2023},
  doi       = {10.1145/3572848.3577500},
  url       = {https://doi.org/10.1145/3572848.3577500}
}

@inproceedings{ohayon2026bsfa,
  title     = {Block Sparse Flash Attention},
  author    = {Daniel Ohayon and Itay Lamprecht and Itay Hubara and Israel Cohen and
               Daniel Soudry and Noam Elata},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  note      = {To appear; arXiv:2512.07011},
  url       = {https://arxiv.org/abs/2512.07011}
}

@article{deepseek2025v32,
  title         = {{DeepSeek-V3.2}: Pushing the Frontier of Open Large Language Models},
  author        = {{DeepSeek-AI}},
  journal       = {arXiv preprint arXiv:2512.02556},
  year          = {2025},
  eprint        = {2512.02556},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2512.02556}
}

@inproceedings{zeng2025zeta,
  title     = {{ZETA}: Leveraging {Z}-order Curves for Efficient Top-$k$ Attention},
  author    = {Qiuhao Zeng and Jerry Huang and Peng Lu and Gezheng Xu and
               Boxing Chen and Charles Ling and Boyu Wang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2501.14577}
}

@article{jaber2026streamindex,
  title         = {{StreamIndex}: Memory-Bounded Compressed Sparse Attention via
                   Streaming Top-k},
  author        = {Jaber Jaber and Osama Jaber},
  journal       = {arXiv preprint arXiv:2605.02568},
  year          = {2026},
  eprint        = {2605.02568},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2605.02568}
}

@inproceedings{singhania2024loki,
  title     = {Loki: Low-rank Keys for Efficient Sparse Attention},
  author    = {Prajwal Singhania and Siddharth Singh and Shwai He and
               Soheil Feizi and Abhinav Bhatele},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {37},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.02542}
}

@inproceedings{ribar2024sparq,
  title     = {{SparQ} Attention: Bandwidth-Efficient {LLM} Inference},
  author    = {Luka Ribar and Ivan Chelombiev and Luke Hudlass-Galley and
               Charlie Blake and Carlo Luschi and Douglas Orr},
  booktitle = {Proceedings of the 41st International Conference on Machine
               Learning (ICML)},
  series    = {PMLR},
  volume    = {235},
  year      = {2024},
  url       = {https://arxiv.org/abs/2312.04985}
}

@inproceedings{tang2024quest,
  title     = {Quest: Query-Aware Sparsity for Efficient Long-Context {LLM} Inference},
  author    = {Jiaming Tang and Yilong Zhao and Kan Zhu and Guangxuan Xiao and
               Baris Kasikci and Song Han},
  booktitle = {Proceedings of the 41st International Conference on Machine
               Learning (ICML)},
  series    = {PMLR},
  volume    = {235},
  year      = {2024},
  url       = {https://arxiv.org/abs/2406.10774}
}

@inproceedings{zhang2025spargeattn,
  title     = {{SpargeAttention}: Accurate and Training-free Sparse Attention
               Accelerating Any Model Inference},
  author    = {Jintao Zhang and Chendong Xiang and Haofeng Huang and Jia Wei and
               Haocheng Xi and Jun Zhu and Jianfei Chen},
  booktitle = {Proceedings of the 42nd International Conference on Machine
               Learning (ICML)},
  series    = {PMLR},
  volume    = {267},
  year      = {2025},
  url       = {https://arxiv.org/abs/2502.18137}
}

@article{gao2024seerattention,
  title         = {{SeerAttention}: Learning Intrinsic Sparse Attention in Your {LLMs}},
  author        = {Yizhao Gao and Zhichen Zeng and Dayou Du and Shijie Cao and
                   Peiyuan Zhou and Jiaxing Qi and Junjie Lai and Hayden Kwok-Hay So and
                   Ting Cao and Fan Yang and Mao Yang},
  journal       = {arXiv preprint arXiv:2410.13276},
  year          = {2024},
  url           = {https://arxiv.org/abs/2410.13276}
}

@article{lu2025moba,
  title         = {{MoBA}: Mixture of Block Attention for Long-Context {LLMs}},
  author        = {Enzhe Lu and Zhejun Jiang and Jingyuan Liu and Yulun Du and
                   Tao Jiang and Chao Hong and Shaowei Liu and Weiran He and
                   Enming Yuan and Yuzhi Wang and Zhiqi Huang and Huan Yuan and
                   Suting Xu and Xinran Xu and Guokun Lai and Yanru Chen and
                   Huabin Zheng and Junjie Yan and Jianlin Su and Yuxin Wu and
                   Neo Y. Zhang and Zhilin Yang and Xinyu Zhou and Mingxing Zhang and
                   Jiezhong Qiu},
  journal       = {arXiv preprint arXiv:2502.13189},
  year          = {2025},
  url           = {https://arxiv.org/abs/2502.13189}
}

@inproceedings{zhang2023airtopk,
  title     = {Parallel Top-{K} Algorithms on {GPU}: A Comprehensive Study and
               New Methods},
  author    = {Jingrong Zhang and Akira Naruse and Xipeng Li and Yong Wang},
  booktitle = {Proceedings of the International Conference for High Performance
               Computing, Networking, Storage and Analysis (SC)},
  year      = {2023},
  doi       = {10.1145/3581784.3607062},
  url       = {https://doi.org/10.1145/3581784.3607062}
}

@inproceedings{li2024radik,
  title     = {{RadiK}: Scalable and Optimized {GPU}-Parallel Radix Top-{K} Selection},
  author    = {Yifei Li and Bole Zhou and Jiejing Zhang and Xuechao Wei and
               Yinghan Li and Yingda Chen},
  booktitle = {Proceedings of the 38th ACM International Conference on
               Supercomputing (ICS)},
  year      = {2024},
  doi       = {10.1145/3650200.3656596},
  url       = {https://doi.org/10.1145/3650200.3656596}
}

@article{cheng2026gvr,
  title         = {Guess-Verify-Refine: Data-Aware Top-{K} for Sparse-Attention
                   Decoding on {Blackwell} via Temporal Correlation},
  author        = {Long Cheng and Ritchie Zhao and Timmy Liu and Mindy Li and
                   Xianjie Qiao and Kefeng Duan and Yu-Jung Chen and Xiaoming Chen and
                   Bita Darvish Rouhani and June Yang},
  journal       = {arXiv preprint arXiv:2604.22312},
  year          = {2026},
  url           = {https://arxiv.org/abs/2604.22312}
}

@inproceedings{hu2024commformer,
  title     = {Learning Multi-Agent Communication from Graph Modeling Perspective},
  author    = {Shengchao Hu and Li Shen and Ya Zhang and Dacheng Tao},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2405.08550}
}

@inproceedings{li2025expocomm,
  title     = {Exponential Topology-enabled Scalable Communication in Multi-agent
               Reinforcement Learning},
  author    = {Xinran Li and Xiaolu Wang and Chenjia Bai and Jun Zhang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2502.19717}
}

@inproceedings{keles2023complexity,
  title     = {On the Computational Complexity of Self-Attention},
  author    = {Feyza Duman Keles and Pruthuvi Mahesakya Wijewardena and
               Chinmay Hegde},
  booktitle = {Proceedings of the 34th International Conference on Algorithmic
               Learning Theory (ALT)},
  series    = {PMLR},
  volume    = {201},
  pages     = {597--619},
  year      = {2023},
  url       = {https://proceedings.mlr.press/v201/duman-keles23a.html}
}

@inproceedings{qu2022dota,
  title     = {{DOTA}: Detect and Omit Weak Attentions for Scalable Transformer
               Acceleration},
  author    = {Zheng Qu and Liu Liu and Fengbin Tu and Zhaodong Chen and
               Yufei Ding and Yuan Xie},
  booktitle = {Proceedings of the 27th ACM International Conference on
               Architectural Support for Programming Languages and Operating
               Systems (ASPLOS)},
  pages     = {14--26},
  year      = {2022},
  doi       = {10.1145/3503222.3507738},
  url       = {https://doi.org/10.1145/3503222.3507738}
}

@inproceedings{pagliardini2023sparseflash,
  title     = {Fast Attention Over Long Sequences With Dynamic Sparse Flash Attention},
  author    = {Matteo Pagliardini and Daniele Paliotta and Martin Jaggi and
               Fran{\c{c}}ois Fleuret},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {36},
  year      = {2023},
  url       = {https://arxiv.org/abs/2306.01160}
}
```

Check before pasting:
- `keles2023complexity`: the page range 597-619 comes from memory and was not shown on a fetched page. Confirm it on the PMLR page or drop `pages`.
- `ohayon2026bsfa`: the NeurIPS 2026 acceptance comes from the arXiv v2 comment ("Accepted to NeurIPS 2026"). Switch to the proceedings entry once it is published.
