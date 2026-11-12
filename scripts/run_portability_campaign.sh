#!/usr/bin/env bash
# The headline measurements again on a different GPU, for the portability
# question: do the crossovers measured on the RTX PRO 6000 Blackwell (sm_120)
# hold on other architectures?
#
# Run from a disposable copy of the repository, never from the main checkout:
# every tiling the benchmarks read is searched again on the device at hand, and
# the searches write to results/autotune*.json and results/precision.json, which
# the kernels and sweeps read back from there. The Blackwell searches are
# removed first so that no row of them can merge into the new ones.
#
# Outputs land in results/portability/$TAG/, with the device description and
# the searched tilings beside the sweeps that used them.
set -eu

TAG="${TAG:?set TAG, e.g. h100}"
DEV="${DEV:-cuda:0}"
PY="${PY:-.venv/bin/python}"
STEPS="${STEPS:-all}"
cd "$(dirname "$0")/.."
OUT="results/portability/$TAG"
LOG="$OUT/logs"
mkdir -p "$LOG"

want() { [ "$STEPS" = all ] || [[ " $STEPS " == *" $1 "* ]]; }

run() {
    name="$1"; shift
    want "$name" || return 0
    echo "=== $name  $(date -Is)"
    "$PY" "$@" > "$LOG/$name.log" 2>&1 || { echo "FAILED $name"; tail -20 "$LOG/$name.log"; return 0; }
    tail -2 "$LOG/$name.log"
}

if want describe; then
    nvidia-smi -q > "$OUT/nvidia_smi_q.txt"
    nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,power.limit,driver_version \
        --format=csv > "$OUT/device.csv"
    "$PY" - > "$OUT/device.json" <<'EOF'
import json, torch, triton
p = torch.cuda.get_device_properties(0)
print(json.dumps({"name": p.name, "capability": f"{p.major}.{p.minor}",
                  "sm_count": p.multi_processor_count,
                  "total_memory_gib": p.total_memory / 2**30,
                  "l2_cache_bytes": getattr(p, "L2_cache_size", None),
                  "torch": torch.__version__, "cuda": torch.version.cuda,
                  "triton": triton.__version__}, indent=1))
EOF
    rm -f results/autotune.json results/autotune_small.json results/autotune_repeat.json \
          results/autotune_streaming.json results/autotune_streaming_precision.json \
          results/autotune_streaming_precision_unadmitted.json results/precision.json
fi

# Tiling searches: every later sweep reads them.
run autotune_small scripts/autotune_kernels.py --device "$DEV" \
    --n-list 48 96 --k-list 1 2 4 8 --d-list 16 --reps 40 \
    --out results/autotune_small.json
run autotune scripts/autotune_kernels.py --device "$DEV" \
    --n-list 192 384 512 --k-list 1 2 4 8 16 --d-list 16 32 \
    --reps 30 --out results/autotune.json
run autotune_streaming scripts/autotune_streaming.py --device "$DEV" \
    --n-list 192 384 512 1024 --k-list 1 2 4 8 --d-list 16 32

# Composed paths against dense, forward and backward (Table: crossover).
run core_d16 scripts/bench_attention_core.py --device "$DEV" \
    --n-list 12 24 48 96 192 384 512 --k-list 1 2 4 8 16 32 \
    --d-list 16 --batch-list 256 --backward --reps 60 --runs 3 \
    --out "$OUT/attention_core_d16.json"

# Fused score-then-select against the same kernel given the set (Table: fixed set).
run fixed_set scripts/bench_fixed_set.py --device "$DEV" \
    --n-list 96 192 384 512 --k-list 1 2 4 8 16 32 --d-list 16 32 \
    --out "$OUT/fixed_set.json"

# Streaming exact top-k in fp32 (Table: streaming).
run streaming_topk scripts/bench_streaming_topk.py --device "$DEV" \
    --n-list 192 384 512 1024 --k-list 1 2 4 8 --d-list 16 32 --runs 3 \
    --out "$OUT/streaming_topk.json"

# Small N under CUDA-graph replay (Table: small N).
run small_n scripts/bench_small_n.py --device "$DEV" \
    --out "$OUT/small_n.json"
run small_n_analyse scripts/bench_small_n.py --analyse "$OUT/small_n.json"

# Selection before scoring (Table: prescore).
run prescore scripts/bench_prescore.py --device "$DEV" \
    --n-list 48 96 192 384 512 --k-list 2 4 8 16 --reps 50 --runs 3 \
    --out "$OUT/prescore.json"

# Formats: the resident kernel and dense baselines per format, then the
# streaming kernel searched and timed per format.
run precision scripts/bench_precision.py --device "$DEV" \
    --n-list 192 384 512 --k-list 1 2 4 8 --d-list 16 32 \
    --out results/precision.json
for m in tf32 bf16; do
    run "autotune_streaming_$m" scripts/autotune_streaming.py --device "$DEV" \
        --mode "$m" --n-list 192 384 512 --k-list 1 2 4 --d-list 16 32
    run "autotune_streaming_${m}_unadmitted" scripts/autotune_streaming.py --device "$DEV" \
        --mode "$m" --ignore-admission --n-list 192 384 512 --k-list 1 2 4 --d-list 16 32 \
        --out results/autotune_streaming_precision_unadmitted.json
done
run streaming_precision scripts/bench_streaming_precision.py --device "$DEV" \
    --modes tf32 bf16 --out "$OUT/streaming_precision.json"

# Machine balance and per-call overhead on this device.
run roofline scripts/roofline.py --devices "$DEV" \
    --out "$OUT/roofline.json" --md "$OUT/roofline.md"
run launch_overhead scripts/launch_overhead.py --device "$DEV" \
    --out "$OUT/launch_overhead.json"
run stages scripts/stage_accounting.py --device "$DEV" \
    --n-list 48 96 192 384 512 --reps 50 --runs 3 \
    --out "$OUT/stage_accounting.json"

# Batch sensitivity of the fp32 streaming crossover. Heads are independent, so
# h heads of width d over a batch b are the same launch as batch b*h: this sweep
# is also the multi-head packing sweep. Both kernels are searched per batch.
for b in 32 1024 4096; do
    run "autotune_streaming_b$b" scripts/autotune_streaming.py --device "$DEV" \
        --n-list 192 384 512 --k-list 1 2 4 8 --d-list 16 --batch "$b" --dense-above 0
    run "streaming_topk_b$b" scripts/bench_streaming_topk.py --device "$DEV" \
        --n-list 192 384 512 --k-list 1 2 4 8 --d-list 16 --batch "$b" --runs 3 \
        --extra-seeds 1 --out "$OUT/streaming_topk_b$b.json"
done

# Hardware counters per kernel (needs Nsight Compute and counter permission).
if [ -n "${NCU:-}" ]; then
    run ncu scripts/ncu_counters.py --ncu "$NCU" --out "$OUT/ncu_counters.json"
fi

for f in autotune autotune_small autotune_streaming autotune_streaming_precision \
         autotune_streaming_precision_unadmitted precision; do
    [ -f "results/$f.json" ] && cp "results/$f.json" "$OUT/$f.json"
done
echo "=== portability campaign $TAG complete $(date -Is)"
