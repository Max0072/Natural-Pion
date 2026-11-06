#!/bin/bash
# Copy the logs of the second-version runs (scripts/paper/v2.sbatch) into runs-record/npv2, without
# checkpoints, then refresh every number of the preprint. Runs not finished yet stay [missing] in the render.
#   bash scripts/paper/collect_npv2.sh
set -e
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
cd "$REPO"
# rows first submitted from v2_rest.sbatch wrote under runs/npv2x; the same <tag>/<run> layout, merged here
[ -d "$DATA_p330/runs/timing" ] && rsync -a --include='*/' --include='log.jsonl' --include='manifest.json' --exclude='*' \
    "$DATA_p330/runs/timing/" runs-record/timing/
for SRC in npv2 npv2x; do
    [ -d "$DATA_p330/runs/$SRC" ] && rsync -a --include='*/' --include='log.jsonl' --include='diagnostics.jsonl' \
        --include='manifest.json' --include='profile.json' --exclude='*' "$DATA_p330/runs/$SRC/" runs-record/npv2/
done
FIG=""; ls runs-record/npv2/rms73242/*/log.jsonl >/dev/null 2>&1 && grep -q '"step": 73241' runs-record/npv2/rms73242/*/log.jsonl \
    && FIG="python scripts/analysis/figures.py --out docs/paper/figures/preprint;"
apptainer exec --bind "$DATA_p330:$DATA_p330" --env PYTHONPATH="$REPO" --pwd "$REPO" "$SIF" sh -c \
    "$FIG python scripts/probes/retraction_drift.py; python scripts/analysis/budget_fit.py; python scripts/analysis/paper_numbers.py; python scripts/analysis/render_paper.py"
# On a GPU, once the runs exist (they measure checkpoints and short training runs of the current code):
#   sbatch scripts/probes/paper_checks.sbatch        K-FAC vs exact curvature, fp32 drift (the method's s3000 checkpoint)
#   RETS="cayley ns ns_delta" sbatch scripts/probes/ns_diag.sbatch     per-step ||W||_F^2 change of each retraction
#   sbatch scripts/probes/optimizer_step_bench.sbatch                 the optimizer step by part (Table 2b)

