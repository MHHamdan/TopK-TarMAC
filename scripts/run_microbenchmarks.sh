#!/usr/bin/env bash
# Full microbenchmark campaign for the efficiency claim.
#
# Runs sequentially, never in parallel: the arms are compared to each other,
# so a second job on the same device would corrupt exactly the comparison
# the campaign exists to make.
#
# Usage: scripts/run_microbenchmarks.sh [DEVICE]   (default cuda:0)
set -euo pipefail
cd "$(dirname "$0")/.."
DEV="${1:-cuda:0}"
PY=.venv/bin/python
R=results

echo "=== 1/6 headline sweep: fp32, batch 256, N up to 512, all arms ==="
$PY scripts/microbenchmark_comm.py --device "$DEV" --dtype fp32 --batch 256 \
  --n-list 6 12 24 48 96 192 384 512 --k-frac 0.25 \
  --reps 200 --warmup 30 --runs 7 --profile-kernels \
  --out $R/microbenchmark_comm.json

echo "=== 2/6 sparsity sweep: does ANY k make the gather path win? ==="
$PY scripts/microbenchmark_comm.py --device "$DEV" --dtype fp32 --batch 256 \
  --n-list 48 96 192 384 512 --k-frac 0.01 0.03 0.0625 0.125 0.25 0.5 \
  --arms dense dense_sdpa topk_masked topk_gather \
  --reps 150 --warmup 30 --runs 5 \
  --out $R/microbenchmark_ksweep.json

echo "=== 3/6 FLOP-bound regime: batch 4096 ==="
$PY scripts/microbenchmark_comm.py --device "$DEV" --dtype fp32 --batch 4096 \
  --n-list 6 12 24 48 96 192 --k-frac 0.25 \
  --reps 100 --warmup 20 --runs 5 \
  --out $R/microbenchmark_comm_batch4096.json

echo "=== 4/6 execution modes: eager vs torch.compile vs CUDA graphs ==="
$PY scripts/microbenchmark_comm.py --device "$DEV" --dtype fp32 --batch 256 \
  --n-list 12 48 192 512 --k-frac 0.25 \
  --arms dense dense_sdpa topk_masked topk_gather \
  --exec-modes eager compile cudagraph \
  --reps 100 --warmup 20 --runs 5 \
  --out $R/microbenchmark_execmodes.json

echo "=== 5/6 bf16: the dtype where flash attention is actually reachable ==="
$PY scripts/microbenchmark_comm.py --device "$DEV" --dtype bf16 --batch 256 \
  --n-list 6 12 24 48 96 192 384 512 --k-frac 0.25 \
  --reps 200 --warmup 30 --runs 5 \
  --out $R/microbenchmark_bf16.json

echo "=== 6/6 CPU: is the conclusion a GPU-kernel artifact or arithmetic? ==="
$PY scripts/microbenchmark_comm.py --device cpu --dtype fp32 --batch 256 \
  --n-list 6 12 24 48 96 192 --k-frac 0.25 \
  --reps 30 --warmup 10 --runs 5 \
  --out $R/microbenchmark_cpu.json

echo "=== campaign complete ==="
