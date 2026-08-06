#!/usr/bin/env bash
# Regenerate every paper figure and table from the tracked code + configs.
#
# Usage:
#   bash scripts/reproduce.sh              # full run (trains anything missing)
#   bash scripts/reproduce.sh --dry-run    # print the plan, touch nothing
#   bash scripts/reproduce.sh --no-train   # rebuild artefacts from existing runs
#
# Step counts come from each YAML's `total_steps`. They are asserted against
# the `total_steps` recorded in every completed run's config.json, so the
# published numbers and the reproduction recipe cannot silently diverge --
# they did before (see AUDIT.md D-15: the YAMLs said 1.0-1.5M while every run
# on disk was trained for 800k).

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

DRY_RUN=0
NO_TRAIN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --no-train) NO_TRAIN=1 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

run() {
  if [ "$DRY_RUN" = "1" ]; then echo "  [dry-run] $*"; else "$@"; fi
}

SEEDS=(0 1 2 3 4)

declare -a CONFIGS=(
  "configs/mappo_mpe_n3_baseline.yaml       mappo_mpe_n3_baseline"
  "configs/mappo_mpe_n3_dense.yaml          mappo_mpe_n3_dense"
  "configs/mappo_mpe_n3_topk.yaml           mappo_mpe_n3_topk"
  "configs/mappo_mpe_n6_baseline.yaml       mappo_mpe_n6_baseline"
  "configs/mappo_mpe_n6_dense.yaml          mappo_mpe_n6_dense"
  "configs/mappo_mpe_n6_topk.yaml           mappo_mpe_n6_topk"
  "configs/mappo_mpe_n6_topk_fixed_k2.yaml  mappo_mpe_n6_topk_fixed_k2"
  "configs/mappo_mpe_n6_topk_fixed_k4.yaml  mappo_mpe_n6_topk_fixed_k4"
  "configs/mappo_mpe_n12_baseline.yaml      mappo_mpe_n12_baseline"
  "configs/mappo_mpe_n12_dense.yaml         mappo_mpe_n12_dense"
  "configs/mappo_mpe_n12_topk.yaml          mappo_mpe_n12_topk"
)

echo "== 0. Config/run step-count consistency check =="
$PYTHON scripts/check_config_drift.py || {
  echo "ABORT: configs disagree with completed runs; reproduction would not" >&2
  echo "       regenerate the published numbers. Fix the YAMLs first." >&2
  exit 1
}

echo "== 1. Sanity smoke tests =="
run $PYTHON -m pytest -q tests/

echo "== 2. Literature artefacts =="
run $PYTHON scripts/build_literature.py

echo "== 3. Training (skips any run that already has final_eval.json) =="
for spec in "${CONFIGS[@]}"; do
  cfg=$(echo "$spec" | awk '{print $1}')
  name=$(echo "$spec" | awk '{print $2}')
  steps=$(grep -E '^\s*total_steps:' "$cfg" | head -1 | awk '{print $2}' | tr -d "_,'\"")
  for seed in "${SEEDS[@]}"; do
    out="results/$name/seed_$seed/final_eval.json"
    if [ -f "$out" ]; then continue; fi
    if [ "$NO_TRAIN" = "1" ]; then
      echo "  [no-train] would train $name seed=$seed for $steps steps"
      continue
    fi
    echo "training $name seed=$seed steps=$steps"
    run $PYTHON scripts/train.py --config "$cfg" --seed "$seed" \
        --run-name "$name" --total-steps "$steps"
  done
done

echo "== 4. Measurements =="
run $PYTHON scripts/measure_random.py
run $PYTHON scripts/measure_flops.py \
  --runs mappo_mpe_n3_baseline mappo_mpe_n3_dense mappo_mpe_n3_topk \
         mappo_mpe_n6_baseline mappo_mpe_n6_dense mappo_mpe_n6_topk \
         mappo_mpe_n12_baseline mappo_mpe_n12_dense mappo_mpe_n12_topk

echo "== 5. Provenance =="
run $PYTHON scripts/collect_provenance.py

echo "== 6. Paper artefacts =="
run $PYTHON scripts/make_tables.py --n-list 3 6 12
run $PYTHON scripts/make_figures.py --n 6 --n-list 3 6 12

echo "== 7. Bibliography check =="
run $PYTHON scripts/check_bib.py

echo "== 8. Compile =="
if [ "$DRY_RUN" = "1" ]; then
  echo "  [dry-run] would compile paper/main.tex"
else
  cd paper
  if command -v tectonic >/dev/null 2>&1; then
    tectonic --keep-logs --synctex main.tex
  elif command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode main.tex
  else
    pdflatex -interaction=nonstopmode main.tex
    bibtex main || true
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex
  fi
  cd ..
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY RUN COMPLETE — nothing was executed."
else
  echo "DONE — paper/main.pdf"
fi
