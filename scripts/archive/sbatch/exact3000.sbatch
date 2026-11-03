#!/bin/bash
#SBATCH -A p330
#SBATCH -J exact3000
#SBATCH --array=0-5%6
#SBATCH -w rtx6003
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# The real grid for candidate C, option A: `trust="exact"`, the per-layer real
# trust region (curv_exact from a backward hook, log-smoothed via exact_beta --
# see scripts/sbatch/exactcal.sbatch for the full background and the journal,
# 2026-08-28, "the exact trust region: it works as a mechanism").
#
# CALIBRATED, not guessed: exactcal.sbatch (job 333701/333797) read the median
# rotation angle at steps 100-300 for six eta values and log-interpolated to
# the quad_curv reference (rot=6e-3, median angle ~1.15-1.23 at the same
# window): eta 0.1 gives median 0.516, eta 0.2 gives 1.730, interpolating to
# **eta ~ 0.16** for a matching angle. This sits just above the 8-25x band
# (4.8e-2 to 1.5e-1) predicted from the pre-fix 2026-08-28 measurement, so the
# old estimate held up across the eta-convention change and the AdamW retuning.
#
# THE GRID. Six points around 0.16 with margin on both sides, since the
# 300-step calibration might not have found the true 3000-step optimum (recall
# the project's own repeated finding that short horizons do not fix an eta):
# 0.08, 0.12, 0.16, 0.2, 0.3, 0.45. adamw fixed at 8e-3 (today's tuned value),
# exact_beta=0.9, seed 0, 3000 steps -- the same protocol as every other
# B=512 head-to-head cell this session.
#
# COMPARE AGAINST: ngd-pion-s (quad_curv) at its own optimum, 3.7954
# (rot=6e-3, adamw=8e-3, seed 0); pion, 3.8412 (rot=1e-3, adamw=8e-3, seed 0).
# Neither rerun here -- both already on disk.
#
#   sbatch -p rtx scripts/sbatch/exact3000.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/exact3000"

ETA=( 0.08 0.12 0.16 0.2 0.3 0.45 )
L="${ETA[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task $SLURM_ARRAY_TASK_ID  ngd-pion-s trust=exact  eta=$L  adamw=8e-3  exact_beta=0.9  batch=512  steps=3000  seed=0"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 \
        --ngd-trust exact --ngd-exact-beta 0.9 --ngd-exact-tokens 4096 \
        --batch-sequences 512 --micro-batch 256 \
        --train-steps 3000 \
        --ngd-t-fac 25 --precision bf16 --seed 0 \
        --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
