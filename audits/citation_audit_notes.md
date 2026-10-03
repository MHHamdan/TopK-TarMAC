# Citation audit notes: paper_submission_revision

Audit date: 2026-10-03. Scope: `references.bib` has 67 entries. 43 are cited in `main.tex`, `sections/*.tex` and `tables/audit.tex` (included through `\input{tables/audit}` in `sections/10_appendix.tex`). `main.bbl` has 43 `\bibitem`s, which matches. The other 24 entries are uncited.

How each entry was checked:
- Metadata: arXiv API for every eprint (title, author count, author order), ACL Anthology `.bib` (NSA, entmax, Routing, Gupta), Crossref (Roofline, IROS, ICRA, AAAI, SC15, MAPL), proceedings and venue pages (MLSys, ICLR, NeurIPS, OpenReview through search).
- Claims: I read the full PDF text of 31 cited papers and pulled the quotes below from it.

## 1. Claims that are not, or only partly, supported

### 1a. Agent range at 02_related.tex:25-26 (NO as written)
> "These methods operate with two to roughly thirty agents, DGN reaching the low hundreds."

The papers cited in that paragraph contradict this:
- **ATOC** (jiang2018atoc): "We trained ATOC and the baselines with the settings of N = 50 and L = 50 ... we directly use the trained models under the setting of N = 50 and L = 50 to the scenario of N = 100 and L = 100." Another task has "50 agents".
- **Mean-field MARL** (yang2018meanfield): "three different numbers of agents (N = 100, 500, 1000)". There is also an "Ising model with 20 × 20 grid", and in the battle game "each army consists of 64 homogeneous agents" ("64 v.s. 64").
- **CommNet**: in the lever game, "m = 5 and N = 500" (5 active agents drawn from 500).
- **DGN** supports "low hundreds": "DGN continuously outperforms Floyd with BL up to N = 140".
- The others fit "two to roughly thirty": TarMAC Nmax=20; IC3Net n≤20 (n=10 on 20×20 grid); G2ANet 5/10/20; MAGIC ≤20; Sun et al. nA=20/30; MAT up to 27m_vs_30m.

Fix: "Most of these methods are evaluated with two to roughly thirty agents; DGN and ATOC reach about one hundred, and mean-field MARL, which replaces communication with a population average, reaches one thousand."

### 1b. FlashAttention and Rabe & Staats at 02_related.tex:58-60 (PARTIAL)
> "FlashAttention and its successors reduce memory traffic without changing the operation count"

- FlashAttention-1: "Even with the increased FLOPs due to recomputation, our algorithm both runs faster ..." The backward pass changes the count.
- FlashAttention-2 aims at "reduc[ing] the number of non-matmul FLOPs".
- Rabe & Staats: "requires O(√n) memory ... While the time complexity is still O(n^2)". This reduces memory *footprint*, which is not the same as memory *traffic*.

Fix: "reduce memory traffic or footprint without reducing the operation count."

### 1c. CPO (achiam2017cpo) at 05_training.tex:12-13 (PARTIAL)
> "a communication budget enforced by a Lagrangian \citep{achiam2017cpo,stooke2020pidlagrangian}"

CPO: "Our method has similar policy updates to primal-dual methods ... but crucially, we differ in computing the dual variables (the Lagrange multipliers for the constraints)."

CPO is a trust-region method for constrained MDPs, not a Lagrangian method. Fix: "a communication budget (a constrained-MDP formulation \citep{achiam2017cpo}) enforced by a Lagrangian \citep{stooke2020pidlagrangian}."

### 1d. Hoefler & Belli at 02_related.tex:70-72 (PARTIAL)
> "Our benchmarking follows established practice for reporting performance on shared hardware"

The paper gives general reporting rules and does not cover shared hardware specifically. It mentions noise from "other applications causing interference". Its Rule 4 says "Avoid summarizing ratios; summarize the costs or rates that the ratios base on instead". The manuscript reports ratios against the best dense kernel and takes minima, so a referee could object.

Fix: drop "on shared hardware", or say which rules are followed (for example Rule 5, confidence intervals and nondeterminism). Also report the absolute times that the ratios are built from.

### 1e. IC3Net quote in the audit table, tables/audit.tex:10 (PARTIAL)
The table gives IC3Net's efficiency claim as "continuous communication allows efficient ...". In the abstract that sentence describes prior work: "Recent works show that continuous communication allows efficient training with back-propagation". IC3Net's own claim is that it "has better training efficiency than simple continuous communication model". Replace the quote.

### 1f. Minor note on the DGN quote
DGN's sentence is "receiving a large amount of information requires high bandwidth and incurs high computational complexity". ATOC (NeurIPS 2018, same first author) contains it almost verbatim, with "long delay" added: "receiving a large amount of information requires high bandwidth and incurs long delay and high computational complexity". The attribution to DGN is accurate. Crediting ATOC as well is optional.

## 2. Supporting quotes for claims marked YES (key ones)

| Key | Manuscript claim | Quote from source |
|---|---|---|
| jiang2018dgn | quoted phrase (02_related:8-9) | "it may be costly and less helpful to take all other agents into consideration, because receiving a large amount of information requires high bandwidth and incurs high computational complexity" |
| jiang2018dgn | neighbourhood from structure | "each node i has a set of neighbors, B_i, which is determined by distance or other metrics ... and varies over time" |
| jiang2018atoc | communication field / proximity | "it only considers the agents in its observable field"; "chooses collaborators ... all based on proximity"; "each initiator can select at most m collaborators" |
| yang2018meanfield | structural interaction | "interactions within the population ... approximated by those between a single agent and the average effect from the overall population or neighboring agents" |
| singh2018ic3net | scalar gate per agent | "communication binary action g_t ... act as a gating function when calculating the communication vector" |
| kim2019schednet | rank by per-agent scalar | "Top(k). Selecting top k agents in terms of their weight values." |
| liu2020g2anet | pairwise hard attention | "we represent all agents as a complete graph ... hard-attention is used to cut the unrelated edges"; "hard weight W_h^{i,j}, which determines whether there is interaction relationship between agent i and j" (Bi-LSTM) |
| niu2021magic | pairwise graph construction | "After concatenating the node feature vectors pairwise ... we obtain a matrix E^t ∈ R^{N×N×2D}" (Scheduler before the GAT Message Processor) |
| sun2020sparsecomm | sparsemax on all-pair logits, complexity motive | "the softmax activation function operated on the raw attention logits"; "we generalize sparsemax [16]"; "This sparsity mechanism can reduce the complexity of both the forward pass and the backpropagation" |
| liu2020who2com | query matched against every candidate | "the other agents compute a learned matching score between their own visual observations and the received request"; "after returning all scores back ... connects to one of the normal agents with the highest matching score" |
| gupta2021topk | chunked exact top-k, memory benefit | "We process the queries in chunks, and for each query, compute the top-k scores with respect to the keys. Our approach offers several advantages: (a) its memory usage is linear in the input size ..." |
| zhao2019explicit | k largest scores per query | "explicit selection method based on top-k selection"; "the top-k largest attention scores" |
| child2019sparse | fixed patterns | "strided attention. Formally, A(1)_i = {t,...,i} ... A(2)_i = {j : (i−j) mod l = 0}" |
| kitaev2020reformer | LSH candidates | "the full attention pattern can be approximated by only allowing attention within each bucket" |
| roy2021routing | online k-means | "a sparse routing module based on online k-means" |
| martins2016sparsemax | exact zeros | "it may assign exactly zero probability to some of its output variables" |
| shazeer2017moe | score-then-select gating | "Noisy Top-K Gating ... keep only the top k values, setting the rest to −∞" |
| dehghani2022misnomer | counts ≠ runtime | "it is commonly assumed that these cost indicators are correlated ... we show that this might not necessarily be the case" |
| ivanov2021datamovement | data movement dominates | "We find that data movement is the key bottleneck when training ... training has now become memory-bound." |
| dong2025flexattention | score mods + block masks | "compiler-driven programming model"; "employs a pre-computed BlockMask ... tracks block-level sparsity" |
| yuan2025nsa | blockwise, tensor cores, 64k | "Optimize blockwise sparse attention for Tensor Core utilization"; block importance scores come from "the attention computation of compression tokens"; "substantial speedups ... on 64k-length sequences" |
| lai2026msa | blockwise, up to 1M | "a blockwise sparse attention built upon Grouped Query Attention (GQA). A lightweight Index Branch scores key-value blocks and independently selects a Top-k subset"; "improve tensor-core utilization under block-granular access"; "14.2× prefill and 7.6× decoding wall-clock speedups" at 1M context |
| wen2022mat | sequence model; measured time = yes | "map agents' observation sequences to agents' optimal action sequences"; appendix figures titled "Relative Wall Time" (6h_vs_8z, 27m_vs_30m) |
| zaheer2020bigbird | measured time = no | No wall-clock, throughput or latency figure found in the full arXiv text (incl. appendices). Only hit: "sparse multiplications cannot be efficiently ..." |

### lai2026msa: does it exist?
Yes. arXiv:2606.13392v2 is "MiniMax Sparse Attention". It was submitted on 2026-06-11 and runs to 30 pages with 14 figures. It has 17 authors, from Xunhao Lai to Pengyu Zhao, and the bib lists the same names in the same order. Affiliations are MiniMax, Peking University, NVIDIA, Zhejiang University and HUST. Code is at github.com/MiniMax-AI/MSA. It is not fabricated, but it is an unrefereed preprint.

## 3. Ready-to-paste BibTeX corrections

```bibtex
@inproceedings{yuan2025nsa,
  title     = {Native Sparse Attention: Hardware-Aligned and Natively Trainable
               Sparse Attention},
  author    = {Jingyang Yuan and Huazuo Gao and Damai Dai and Junyu Luo and
               Liang Zhao and Zhengyan Zhang and Zhenda Xie and Yuxing Wei and
               Lean Wang and Zhiping Xiao and Yuqing Wang and Chong Ruan and
               Ming Zhang and Wenfeng Liang and Wangding Zeng},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for
               Computational Linguistics (Volume 1: Long Papers)},
  pages     = {23078--23097},
  year      = {2025},
  address   = {Vienna, Austria},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2025.acl-long.1126},
  eprint    = {2502.11089},
  url       = {https://aclanthology.org/2025.acl-long.1126/}
}

@inproceedings{hoefler2015benchmarking,
  title     = {Scientific Benchmarking of Parallel Computing Systems: Twelve
               Ways to Tell the Masses when Reporting Performance Results},
  author    = {Hoefler, Torsten and Belli, Roberto},
  booktitle = {Proceedings of the International Conference for High Performance
               Computing, Networking, Storage and Analysis ({SC})},
  pages     = {1--12},
  year      = {2015},
  publisher = {ACM},
  doi       = {10.1145/2807591.2807644}
}

@inproceedings{tillet2019triton,
  title     = {Triton: An Intermediate Language and Compiler for Tiled Neural
               Network Computations},
  author    = {Tillet, Philippe and Kung, H. T. and Cox, David},
  booktitle = {Proceedings of the 3rd ACM SIGPLAN International Workshop on
               Machine Learning and Programming Languages ({MAPL})},
  pages     = {10--19},
  year      = {2019},
  publisher = {ACM},
  doi       = {10.1145/3315508.3329973}
}
```

Optional URL additions, for completeness. None of these is an error.
- `dehghani2022misnomer`: `url = {https://openreview.net/forum?id=iulEMLYh1uR}`
- `dao2024flashattention2`: `url = {https://openreview.net/forum?id=mZn2Xyh9Ec}`
- `dong2025flexattention`: `url = {https://proceedings.mlsys.org/paper_files/paper/2025/file/61a9278dfef5f871b5e472389f8d6fa1-Paper-Conference.pdf}`
- `ivanov2021datamovement`: `url = {https://proceedings.mlsys.org/paper_files/paper/2021/file/bc86e95606a6392f51f95a8de106728d-Paper.pdf}`
- `liu2020g2anet`: `url = {https://doi.org/10.1609/aaai.v34i05.6211}`

A style issue across the file: about 30 conference papers are typed `@article` with the venue in `journal`. Examples are TarMAC, DGN, ATOC, IC3Net, CommNet, Reformer, Performer, MoE, MFRL, sparsemax, entmax, CPO, rliable and EPyMARL. With `tmlr.bst` these print as journal articles. Converting them to `@inproceedings` and `booktitle` is recommended, but it does not change any fact.

## 4. Uncited entries (24)

rashid2018qmix, foerster2017coma, lowe2017maddpg, sunehag2017vdn, foerster2016dial, dewitt2020ippo, kuba2021happo, samvelyan2019smac, ellis2022smacv2, rashid2020wqmix, son2019qtran, yang2020qatten, hu2023marllib, schulman2017ppo, espeholt2020seedrl, horgan2018apex, liang2017rllib, hoffman2020acme, maddison2017concrete, louizos2018l0, henderson2018matters, rutherford2023jaxmarl, christianos2020seac, hamdan2026marliotp.

BibTeX leaves these out of the output, so they do no harm. If any are cited later, fix these first:
- Abbreviated or truncated titles: QMIX, SMACv2, Weighted QMIX (missing "for Deep Multi-Agent Reinforcement Learning"), QTRAN and Qatten.
- Acme: the author list is cut off with "others"; arXiv lists 39 authors.
- hamdan2026marliotp is a self-citation whose venue I could not verify.

The file header still says "49 entries"; the file has 67.
