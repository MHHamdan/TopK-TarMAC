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

# The tiling search runs first: every later sweep times the dense configuration
# it finds, so that no ratio anywhere in the manuscript is quoted against a
# dense kernel weaker than the best one the manuscript itself reports.
run autotune_small scripts/autotune_kernels.py --device "$DEV" \
    --n-list 48 96 --k-list 1 2 4 8 --d-list 16 --reps 40 \
    --out results/autotune_small.json

run autotune scripts/autotune_kernels.py --device "$DEV" \
    --n-list 192 384 512 --k-list 1 2 4 8 16 --d-list 16 32 64 128 \
    --reps 30 --out results/autotune.json

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
    --batch-list 32 1024 \
    --impls dense_unfused dense_fused dense_triton dense_tuned topk_gather \
            topk_fused_gather \
    --reps 40 --runs 3 --out results/attention_core_batch.json

# The three fused selection strategies over the same grid, which is what
# decides whether sparse arithmetic and selection cost independent of the
# retained count can be had at the same time.
run core_sort scripts/bench_attention_core.py --device "$DEV" \
    --n-list 12 24 48 96 192 384 512 --k-list 1 2 4 8 16 32 64 128 \
    --d-list 16 --batch-list 256 --reps 60 --runs 3 \
    --impls dense_fused dense_triton dense_tuned topk_gather \
            topk_fused_gather topk_fused_masked topk_fused_sort \
    --out results/attention_core_sort.json

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
    --n-agents 6 12 24 --iterations 5 --warmup 2 \
    --out results/training_step_timing.json

# What a single kernel call costs before it computes anything, which is what
# decides whether a small-N ratio is about the algorithm.
run launch_overhead scripts/launch_overhead.py --device "$DEV" \
    --out results/launch_overhead.json

# Machine balance, with the bandwidth probe reading a buffer that has been
# written and device work timed with device events.
run roofline scripts/roofline.py --devices "$DEV" cpu \
    --out results/roofline.json

# The pre-score sweep again, for the same reason.
run prescore_repeat scripts/bench_prescore.py --device "$DEV" \
    --n-list 48 96 192 384 512 --k-list 2 4 8 16 --reps 50 --runs 3 \
    --out results/prescore_repeat.json

# The headline sweep again, hours later, to establish which agent counts
# reproduce. No claim in the paper is read off one that does not.
run core_repeat scripts/bench_attention_core.py --device "$DEV" \
    --n-list 96 192 384 512 --k-list 1 2 4 8 16 32 64 128 \
    --d-list 16 --batch-list 256 --reps 60 --runs 3 \
    --out results/attention_core_d16_repeat.json

# Every implementation against the reference at every reported configuration.
run equivalence scripts/verify_all_configurations.py --device "$DEV" \
    --out results/equivalence.json

echo "=== campaign complete $(date -Is)"
echo "regenerate the manuscript's tables and figures with:"
echo "  $PY scripts/make_revision_tables.py"
echo "  $PY scripts/make_revision_figures.py"
echo "  $PY scripts/check_revision_numbers.py"
