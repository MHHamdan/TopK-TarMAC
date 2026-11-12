| quantity | blackwell | a100 | h100 | h200 |
|---|---|---|---|---|
| composed gathered ÷ best dense, min over N≥192, all k (fwd) | 9.01 | 6.57 | 6.75 | 7.65 |
| composed gathered ÷ best dense, min (bwd) | 1.53 | 2.60 | 2.03 | 2.10 |
| resident fused k=1 ÷ best dense (graph, N≥192) | 0.67--0.91 | 0.69--0.87 | 0.69--0.89 | 0.69--0.89 |
| streaming fp32 k=1 ÷ best dense | 0.46--0.63 | 0.56--0.64 | 0.56--0.68 | 0.56--0.68 |
| streaming fp32 k=2 ÷ best dense | 0.56--0.84 | 0.77--0.96 | 0.76--1.13 | 0.76--1.13 |
| streaming fp32 k=4 ÷ best dense | 0.78--1.15 | 1.03--1.50 | 1.05--1.45 | 1.05--1.47 |
| streaming fp32 k=8 ÷ best dense | 0.92--1.76 | 1.28--2.96 | 1.25--2.04 | 1.25--2.06 |
| share of fused time spent finding the set (N≥384) | 90--95% | 91--95% | 91--96% | 93--97% |
| streaming kernel: share of time finding the set (N≥384) | 85--91% | 88--94% | 88--94% | 88--95% |
| small N: rank correlation with kernel count; with FLOPs | 0.95; 0.03 | 0.91; 0.10 | 0.94; 0.02 | 0.93; 0.06 |
| streaming TF32 k=1 ÷ best TF32 dense | 0.62--0.82 | 0.83--1.05 | 0.83--1.03 | 0.88--1.33 |
| streaming bf16 k=1 ÷ best bf16 dense | 0.85--1.06 | 1.03--1.30 | 1.10--1.32 | 1.11--1.52 |
| small N: best dense at N=12 (µs); composed ÷ dense at N≤96 | 4.0; 2.7--9.8 | 21.0; 1.7--9.1 | 13.3; 1.6--9.4 | 9.5; 2.0--9.0 |
| pre-score Z-order, N=512 k=4: attend; select+attend ÷ best dense | 0.09; 0.85 | 0.22; 1.22 | 0.24; 1.45 | 0.16; 1.12 |
| measured matmul TFLOP/s (roofline probe); DRAM copy GB/s | 77; 1461 | 19; 1363 | 52; 2987 | 51; 4205 |
