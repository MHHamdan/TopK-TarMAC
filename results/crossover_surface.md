# Crossover surface — where does gathering win?

46 measured cells, 2 skipped for memory. For each (d, batch, N) the *best* k for the gather path is used, i.e. the most favourable case sparsity can construct.

- Cells where gathering beats **unfused dense**: **0/46**
- Cells where gathering beats **fused SDPA**: **1/46**


## Full surface (ratio of best-k gathered to unfused dense)

| d | batch | N=12 | N=48 | N=192 | N=512 |
|---|---|---|---|---|---|
| 32 | 32 | 1.22 | 1.20 | 1.19 | 2.29 |
| 32 | 256 | 1.21 | 1.21 | 1.83 | 2.43 |
| 32 | 4096 | 1.35 | 1.98 | 1.48 | 2.36 |
| 64 | 32 | 1.22 | 1.22 | 1.22 | 2.35 |
| 64 | 256 | 1.23 | 1.22 | 1.76 | 2.46 |
| 64 | 4096 | 1.50 | 1.85 | 1.47 | 2.36 |
| 128 | 32 | 1.21 | 1.22 | 1.24 | 1.87 |
| 128 | 256 | 1.23 | 1.27 | 1.52 | 2.12 |
| 128 | 4096 | 1.45 | 1.91 | 1.50 | skip |
| 256 | 32 | 1.23 | 1.20 | 1.16 | 1.47 |
| 256 | 256 | 1.19 | 1.31 | 1.37 | 1.71 |
| 256 | 4096 | 1.28 | 1.61 | 1.45 | skip |
