#!/usr/bin/env bash
# Regenerate every paper figure and table from the tracked code + configs.
# Assumes the venv is activated and PYTHONPATH includes the repo root.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# 1. Sanity smoke tests
$PYTHON -m pytest -q tests/

# 2. Build literature artefacts from the structured data source
$PYTHON scripts/build_literature.py

# 3. Train all conditions (only if results are missing) — this is the heavy
#    step; comment out if you only want to rebuild figures from existing logs.
declare -a CONFIGS=(
  "configs/mappo_mpe_n3_baseline.yaml mappo_mpe_n3_baseline"
  "configs/mappo_mpe_n3_dense.yaml    mappo_mpe_n3_dense"
  "configs/mappo_mpe_n3_topk.yaml     mappo_mpe_n3_topk"
  "configs/mappo_mpe_n6_baseline.yaml mappo_mpe_n6_baseline"
  "configs/mappo_mpe_n6_dense.yaml    mappo_mpe_n6_dense"
  "configs/mappo_mpe_n6_topk.yaml     mappo_mpe_n6_topk"
  "configs/mappo_mpe_n12_baseline.yaml mappo_mpe_n12_baseline"
  "configs/mappo_mpe_n12_dense.yaml    mappo_mpe_n12_dense"
  "configs/mappo_mpe_n12_topk.yaml     mappo_mpe_n12_topk"
)

for spec in "${CONFIGS[@]}"; do
  cfg=$(echo "$spec" | awk '{print $1}')
  run=$(echo "$spec" | awk '{print $2}')
  steps=$(grep -E '^\s*total_steps:' "$cfg" | head -1 | awk '{print $2}' | tr -d "_,'\"")
  for seed in 0 1 2 3 4; do
    out="results/$run/seed_$seed/final_eval.json"
    if [ -f "$out" ]; then continue; fi
    echo "training $run seed=$seed"
    $PYTHON scripts/train.py --config "$cfg" --seed "$seed" \
        --run-name "$run" --total-steps "$steps"
  done
done

# 4. Render the FLOPs summary + paper artefacts
$PYTHON scripts/measure_flops.py \
  --runs mappo_mpe_n3_baseline mappo_mpe_n3_dense mappo_mpe_n3_topk \
         mappo_mpe_n6_baseline mappo_mpe_n6_dense mappo_mpe_n6_topk \
         mappo_mpe_n12_baseline mappo_mpe_n12_dense mappo_mpe_n12_topk
$PYTHON scripts/make_tables.py --n-list 3 6 12
$PYTHON scripts/make_figures.py --n 6 --n-list 3 6 12

# 5. Compile the paper. Tectonic is self-contained and resolves packages on
# the fly; falls back to pdflatex+bibtex if tectonic is not installed.
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
echo "DONE — paper/main.pdf"
