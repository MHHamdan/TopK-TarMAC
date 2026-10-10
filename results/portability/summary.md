| quantity | blackwell | a100 | h100 |
|---|---|---|---|
| composed gathered ÷ best dense, min over N≥192, all k (fwd) | 9.01 | 6.57 | 6.75 |
| composed gathered ÷ best dense, min (bwd) | 1.53 | 2.60 | 2.03 |
| resident fused k=1 ÷ best dense (graph, N≥192) | 0.67--0.91 | 0.69--0.87 | 0.69--0.89 |
| streaming fp32 k=1 ÷ best dense | 0.46--0.63 | 0.56--0.64 | 0.56--0.68 |
| streaming fp32 k=2 ÷ best dense | 0.56--0.84 | 0.77--0.96 | 0.76--1.13 |
| streaming fp32 k=4 ÷ best dense | 0.78--1.15 | 1.03--1.50 | 1.05--1.45 |
| streaming fp32 k=8 ÷ best dense | 0.92--1.76 | 1.28--2.96 | 1.25--2.04 |
| share of fused time spent finding the set (N≥384) | 90--95% | 91--95% | 91--96% |
| streaming TF32 k=1 ÷ best TF32 dense | 0.62--0.82 | 0.83--1.05 | 0.83--1.03 |
| streaming bf16 k=1 ÷ best bf16 dense | 0.85--1.06 | 1.03--1.30 | 1.10--1.32 |
| small N: best dense at N=12 (µs); composed ÷ dense at N≤96 | 4.0; 2.7--9.8 | 21.0; 1.7--9.1 | 13.3; 1.6--9.4 |
| pre-score Z-order, N=512 k=2: attend; select+attend ÷ best dense | 0.06; 0.83 | 0.20; 1.19 | 0.23; 1.44 |
| measured matmul TFLOP/s (roofline probe); DRAM copy GB/s | 77; 1461 | 19; 1363 | 52; 2987 |
