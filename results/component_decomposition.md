# Where the top-k gather path's time actually goes

batch 256, msg_dim 16, k = 25% of peers, NVIDIA RTX PRO 6000 Blackwell Server Edition. Latency is `min_ms` over interleaved samples.

`saving` is what the sparse contraction buys (`agg_dense - agg_sparse`); `overhead` is what selection costs (`topk_select + gather_v + softmax_k - softmax_N`). A positive `net` means selection overhead exceeds the aggregation saving.

| N | k | topk_select | gather_v | softmax_N | agg_dense | agg_sparse | saving | overhead | net | overhead/saving |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 2 | 0.0141 | 0.0151 | 0.0116 | 0.0167 | 0.0194 | -0.0027 | +0.0286 | +0.0313 | — |
| 48 | 11 | 0.0281 | 0.0146 | 0.0116 | 0.0173 | 0.0222 | -0.0049 | +0.0415 | +0.0464 | — |
| 96 | 23 | 0.0688 | 0.0277 | 0.0119 | 0.0248 | 0.0559 | -0.0311 | +0.0948 | +0.1259 | — |
| 192 | 47 | 0.2576 | 0.1476 | 0.0355 | 0.0590 | 0.1455 | -0.0866 | +0.3832 | +0.4698 | — |
| 384 | 95 | 1.0784 | 0.5606 | 0.2047 | 0.1780 | 0.4956 | -0.3177 | +1.4800 | +1.7976 | — |
| 512 | 127 | 3.4258 | 0.9989 | 0.3646 | 0.2492 | 0.8444 | -0.5952 | +4.1361 | +4.7313 | — |
