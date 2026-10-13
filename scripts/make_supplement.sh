#!/usr/bin/env bash
# Build the anonymised supplementary archive for double-blind review.
#
# Copies only what reproduces the paper -- kernels, harness, analysis scripts,
# measurement artifacts, manuscript sources -- strips build products and
# caches, rewrites home-directory paths, and then refuses to finish if any
# identifying string survives. Git history is never included.
#
#   bash scripts/make_supplement.sh            -> dist/supplement.zip
#   IDENT_PATTERNS='name|handle' bash ...      -> extra strings to forbid
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=dist/supplement
rm -rf "$OUT" dist/supplement.zip
mkdir -p "$OUT"

# What reproduces the paper. README.md, LICENSE, decision logs, revision
# reports and audits are left out: they name the authors or the repository.
rsync -a \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '*.egg-info' \
  --exclude '.ipynb_checkpoints' \
  src scripts tests configs pyproject.toml Makefile "$OUT"/
mkdir -p "$OUT/results"
rsync -a --exclude 'logs' --exclude 'provenance.json' \
  --exclude 'MASTER_LOG.csv' results/ "$OUT/results/"
mkdir -p "$OUT/paper_submission_revision"
rsync -a --exclude 'main.log' --exclude 'main.aux' --exclude 'main.blg' \
  --exclude 'main.out' --exclude '*.synctex*' \
  paper_submission_revision/ "$OUT/paper_submission_revision/"
rm -f "$OUT/scripts/make_supplement.sh"
# Set only in the accepted build, and it names the funding grant.
rm -f "$OUT/paper_submission_revision/sections/acknowledgment.tex"

# Home-directory paths, e.g. an editable install recorded in the lockfile.
find "$OUT" -type f \( -name '*.lock' -o -name '*.json' -o -name '*.yaml' \
  -o -name '*.yml' -o -name '*.py' -o -name '*.md' -o -name '*.sh' \
  -o -name '*.csv' -o -name '*.toml' \) -print0 |
  xargs -0 sed -i -E -e 's#/home/[A-Za-z0-9_.-]+#/home/anonymous#g' \
    -e 's#-home-[A-Za-z0-9_.]+-#-home-anonymous-#g'

cat > "$OUT/README.md" <<'EOF2'
# Supplementary material

Code and measurement artifacts for "SCALE: When Does Sparse Multi-Agent
Attention Pay for Its Selector?" (SCALE: Selector-Cost Accounting for Latency
Evaluation).

    make check      # tests, bibliography, every number in the manuscript
    make tables     # regenerate paper_submission_revision/tables from results/
    make figures    # regenerate paper_submission_revision/figures from results/
    make paper      # compile the manuscript (needs tectonic)

GPU entry points: `scripts/run_revision_campaign.sh` (every sweep),
`scripts/verify_all_configurations.py` (equivalence),
`scripts/bench_fixed_set.py` (fixed-set sweep),
`scripts/run_portability_campaign.sh` (the headline sweeps on another GPU, into
`results/portability/<gpu>/`), `scripts/ncu_counters.py` (Nsight Compute
counters) and `scripts/summarise_portability.py` (the other-GPU tables). The environment, timing protocol
and measurement parameters are in Appendix I of the manuscript; the package set
is `configs/environment.lock`.
EOF2

# Refuse to ship anything that identifies the authors.
PAT='mhamdan|Hamdan|mh2022ets|github\.com/[A-Za-z0-9_-]+/TopK|hamdan2026|RGPIN|NSERC'
if [[ -n "${IDENT_PATTERNS:-}" ]]; then PAT="$PAT|$IDENT_PATTERNS"; fi
# Two deliberate exceptions: a third-person citation of prior work (the cited
# paper's author list) and the anonymised code URL (github.com/XXX/...).
ALLOW='Cheriet|github\.com/XXX/'
if grep -rIHnE "$PAT" "$OUT" | grep -vE "$ALLOW" | cut -d: -f1 | sort -u | grep .; then
  echo "identifying strings remain in the files above; archive not built" >&2
  exit 1
fi
# PDFs are binary: check their metadata separately.
for f in $(find "$OUT" -name '*.pdf'); do
  if strings "$f" | grep -qE "$PAT"; then
    echo "identifying string in $f" >&2; exit 1
  fi
done

python3 -c "import shutil; shutil.make_archive('dist/supplement', 'zip', 'dist', 'supplement')"
echo "wrote dist/supplement.zip ($(du -h dist/supplement.zip | cut -f1))"
