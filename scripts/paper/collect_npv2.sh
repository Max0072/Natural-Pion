#!/bin/bash
# Copy the logs of the per-head Section 5 runs (scripts/paper/v2_perhead.sbatch) into runs-record/npv2, without
# checkpoints, then refresh every number of the preprint. Runs not finished yet stay [missing] in the render.
#   bash scripts/paper/collect_npv2.sh
set -e
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
cd "$REPO"
rsync -a --include='*/' --include='log.jsonl' --include='diagnostics.jsonl' --include='manifest.json' --exclude='*' \
    "$DATA_p330/runs/npv2/" runs-record/npv2/
FIG=""; ls runs-record/npv2/s73242/*/log.jsonl >/dev/null 2>&1 && grep -q '"step": 73241' runs-record/npv2/s73242/*/log.jsonl \
    && FIG="python scripts/analysis/figures.py --out docs/paper/figures/preprint;"
apptainer exec --env PYTHONPATH="$REPO" --pwd "$REPO" "$SIF" sh -c \
    "$FIG python scripts/analysis/budget_fit.py; python scripts/analysis/paper_numbers.py; python scripts/analysis/render_paper.py"
