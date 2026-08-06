# Roofline: why fewer FLOPs does not buy less time

## Measured machine balance

| Device | Achievable fp32 GEMM | Achievable bandwidth | Ridge point (FLOP/byte) |
|---|---|---|---|
| `cuda:1` NVIDIA RTX PRO 6000 Blackwell Server Edition | 76.84 TFLOP/s | 1449 GB/s | **53.0** |
| `cpu` x86_64 CPU | 1.66 TFLOP/s | 451 GB/s | **3.7** |

## Path arithmetic intensity (batch 256, d=16, k=25% of peers)

| N | k | dense GFLOP | dense GB | dense AI | gather GFLOP | gather GB | gather AI | AI ratio (gather/dense) |
|---|---|---|---|---|---|---|---|---|
| 12 | 2 | 0.003 | 0.001 | **1.85** | 0.001 | 0.002 | **0.64** | 0.34x |
| 48 | 11 | 0.041 | 0.013 | **3.23** | 0.024 | 0.036 | **0.66** | 0.20x |
| 192 | 47 | 0.651 | 0.164 | **3.98** | 0.387 | 0.584 | **0.66** | 0.17x |
| 512 | 127 | 4.631 | 1.107 | **4.18** | 2.763 | 4.158 | **0.66** | 0.16x |

## Stage breakdown at N=512, k=127

| Path | Stage | GFLOP | GB moved | AI |
|---|---|---|---|---|
| dense | scores  q@k^T | 2.147 | 0.285 | 7.53 |
| dense | softmax over N | 0.336 | 0.537 | 0.62 |
| dense | aggregate (B,N,N)x(B,N,d) | 2.147 | 0.285 | 7.53 |
| gather | scores  q@k^T | 2.147 | 0.285 | 7.53 |
| gather | top-k select | 0.000 | 0.402 | 0.00 |
| gather | softmax over k | 0.083 | 0.133 | 0.62 |
| gather | gather v -> (B,N,k,d) | 0.000 | 2.197 | 0.00 |
| gather | aggregate (B*N,1,k)x(B*N,k,d) | 0.533 | 1.140 | 0.47 |

## Reading

The two zero-FLOP stages the gather path adds -- the top-k selection and the gather itself -- move substantial memory for no arithmetic, so they have arithmetic intensity 0 and pull the whole path below the dense path's intensity. Both paths sit far beneath every ridge point measured above, i.e. both are memory-bound, and in that regime latency tracks bytes moved. Removing multiply-accumulates from a memory-bound kernel does not make it faster; adding traffic to one makes it slower.

This is why the result reproduces on CPU despite a machine balance an order of magnitude different from the GPU's: the conclusion follows from the ratio of the two paths' intensities, which is a property of the algorithm, not of the device.

