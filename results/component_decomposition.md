# Where the top-k gather path's time actually goes

batch 256, msg_dim 16, k = 25% of peers, NVIDIA RTX PRO 6000 Blackwell Server Edition. Latency is `min_ms` over interleaved samples.

`saving` is what the sparse contraction buys (`agg_dense - agg_sparse`); `overhead` is what selection costs (`topk_select + gather_v + softmax_k - softmax_N`). A positive `net` means selection overhead exceeds the aggregation saving.

| N | k | topk_select | gather_v | softmax_N | agg_dense | agg_sparse | saving | overhead | net | overhead/saving |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 2 | 0.0164 | 0.0126 | 0.0095 | 0.0138 | 0.0163 | -0.0026 | +0.0284 | +0.0310 | — |
| 48 | 11 | 0.0365 | 0.0122 | 0.0094 | 0.0149 | 0.0211 | -0.0062 | +0.0474 | +0.0537 | — |
| 96 | 23 | 0.0866 | 0.0270 | 0.0096 | 0.0222 | 0.0551 | -0.0328 | +0.1135 | +0.1463 | — |
| 192 | 47 | 0.4338 | 0.1456 | 0.0356 | 0.0564 | 0.1444 | -0.0880 | +0.5544 | +0.6424 | — |
| 384 | 95 | 1.4274 | 0.5543 | 0.2034 | 0.1735 | 0.4945 | -0.3210 | +1.8239 | +2.1449 | — |
| 512 | 127 | 3.8965 | 0.9862 | 0.3652 | 0.2470 | 0.8453 | -0.5983 | +4.5954 | +5.1938 | — |
