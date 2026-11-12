#!/usr/bin/env bash
# Launch the Phase-C reduced design, one priority tier at a time.
#
# The tiers are a gate, not a convenience. Tier 1 asks whether the
# communication channel does anything at all on a task where it must; if it
# does not, tiers 2-4 measure the sparsification of a channel that carries
# nothing and none of their results are interpretable (D-011, D-027). So this
# script deliberately runs ONE tier per invocation and stops.
#
#   scripts/run_reduced_design.sh 1        # positive control
#   scripts/run_reduced_design.sh 2        # random-k control
#   scripts/run_reduced_design.sh 3        # fixed-k grid
#   scripts/run_reduced_design.sh 4        # Lagrangian + entmax
#
# Env: SEEDS (default 5), JOBS (default 8), STEPS (override total steps).
set -uo pipefail
cd "$(dirname "$0")/.."
TIER="${1:?usage: run_reduced_design.sh <tier 1-4>}"
SEEDS="${SEEDS:-5}"
JOBS="${JOBS:-8}"
PY=.venv/bin/python

mapfile -t ARMS < <(
  $PY - "$TIER" <<'EOF'
import json, sys
m = json.load(open("configs/reduced/MANIFEST.json"))
for a in m["arms"]:
    if a["tier"] == int(sys.argv[1]):
        print(a["run_name"])
EOF
)

if [ "${#ARMS[@]}" -eq 0 ]; then
  echo "no arms in tier $TIER" >&2; exit 1
fi

echo "tier $TIER: ${#ARMS[@]} arms x $SEEDS seeds = $(( ${#ARMS[@]} * SEEDS )) runs, ${JOBS} at a time"
mkdir -p results/logs

launch() {
  local arm="$1" seed="$2"
  local log="results/logs/${arm}_seed${seed}.log"
  if [ -f "results/${arm}/seed_${seed}/final_eval.json" ]; then
    echo "  skip ${arm} seed ${seed} (already complete)"; return 0
  fi
  $PY scripts/train.py --config "configs/reduced/${arm}.yaml" --seed "$seed" \
      ${STEPS:+--total-steps $STEPS} > "$log" 2>&1 \
      && echo "  done ${arm} seed ${seed}" \
      || echo "  FAILED ${arm} seed ${seed} (see $log)"
}

for arm in "${ARMS[@]}"; do
  for seed in $(seq 0 $((SEEDS - 1))); do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    launch "$arm" "$seed" &
  done
done
wait
echo "tier $TIER complete. Inspect before launching the next tier:"
echo "  .venv/bin/python scripts/summarise_reduced_design.py"
