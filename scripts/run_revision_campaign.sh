#!/usr/bin/env bash
# Every measurement added for the revision, in one place and in one order.
#
# The sweeps are serialised deliberately: the device is shared with other
# tenants, and running two of our own sweeps at once would put our measurements
# in each other's way on top of everything else already there.
set -eu

DEV="${DEV:-cuda:0}"
PY="${PY:-.venv/bin/python}"
LOG="${LOG:-/tmp/revision_campaign}"
mkdir -p "$LOG"
cd "$(dirname "$0")/.."

run() {
    name="$1"; shift
    echo "=== $name  $(date -Is)"
    "$PY" "$@" > "$LOG/$name.log" 2>&1
    tail -3 "$LOG/$name.log"
}

# Sparsity, agent count, and forward against backward, at the head width the
# manuscript's headline configuration uses.
run core_d16 scripts/bench_attention_core.py --device "$DEV" \
    --n-list 12 24 48 96 192 384 512 --k-list 1 2 4 8 16 32 64 128 \
    --d-list 16 --batch-list 256 --backward --reps 60 --runs 3 \
    --out results/attention_core_d16.json

# Head width, to 512, which is where the manuscript previously stopped short.
run core_dsweep scripts/bench_attention_core.py --device "$DEV" \
    --n-list 96 192 512 --k-list 1 2 4 8 16 32 \
    --d-list 16 32 64 128 256 512 --batch-list 256 --reps 40 --runs 3 \
    --out results/attention_core_dsweep.json

# Batch, over two orders of magnitude.
run core_batch scripts/bench_attention_core.py --device "$DEV" \
    --n-list 96 192 512 --k-list 1 2 4 8 16 --d-list 16 \
    --batch-list 32 1024 4096 --reps 40 --runs 3 \
    --out results/attention_core_batch.json

# Per-stage arithmetic, traffic, intensity and attainment.
run stages scripts/stage_accounting.py --device "$DEV" \
    --n-list 48 96 192 384 512 --reps 50 --runs 3 \
    --out results/stage_accounting.json

# Selection before scoring: the escape case of Proposition 2.
run prescore scripts/bench_prescore.py --device "$DEV" \
    --n-list 48 96 192 384 512 --k-list 2 4 8 16 --reps 50 --runs 3 \
    --out results/prescore.json

# What a whole training iteration spends its time on.
run training_step scripts/bench_training_step.py --device "$DEV" \
    --n-agents 6 12 24 --iterations 5 \
    --out results/training_step_timing.json

# Machine balance, with the bandwidth probe reading a buffer that has been
# written rather than an untouched allocation.
run roofline scripts/roofline.py --devices "$DEV" cpu \
    --out results/roofline.json

echo "=== campaign complete $(date -Is)"
