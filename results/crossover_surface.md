# Crossover surface — where does gathering win?

46 measured cells, 2 skipped for memory. For each (d, batch, N) the *best* k for the gather path is used, i.e. the most favourable case sparsity can construct.

- Cells where gathering beats **unfused dense**: **0/46**
- Cells where gathering beats **fused SDPA**: **1/46**


## Full surface (ratio of best-k gathered to unfused dense)

| d | batch | N=12 | N=48 | N=192 | N=512 |
|---|---|---|---|---|---|
| 32 | 32 | 1.17 | 1.18 | 1.18 | 2.19 |
| 32 | 256 | 1.18 | 1.19 | 1.83 | 2.40 |
| 32 | 4096 | 1.35 | 1.97 | 1.48 | 2.33 |
| 64 | 32 | 1.17 | 1.19 | 1.22 | 2.29 |
| 64 | 256 | 1.19 | 1.23 | 1.75 | 2.44 |
| 64 | 4096 | 1.49 | 1.84 | 1.47 | 2.34 |
| 128 | 32 | 1.17 | 1.19 | 1.23 | 1.81 |
| 128 | 256 | 1.19 | 1.24 | 1.51 | 2.10 |
| 128 | 4096 | 1.45 | 1.91 | 1.49 | skip |
| 256 | 32 | 1.18 | 1.18 | 1.18 | 1.44 |
| 256 | 256 | 1.17 | 1.25 | 1.37 | 1.70 |
| 256 | 4096 | 1.26 | 1.57 | 1.45 | skip |
