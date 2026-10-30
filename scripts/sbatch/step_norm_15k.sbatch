#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepnl15
#SBATCH --array=0-2
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 14:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Does step_norm's lead survive to 15000 steps? (journal 2026-10-30; H4 of docs/PLAN.md)
#
# step_norm (t_fac 1, alpha off, cosine), flags of ladder15k.sbatch otherwise: B 512, adamw 8e-3,
# seed 0. References at 15000: pion 3.5254, old method (t_fac 25, alpha) 3.5302. About 7.5 h per task.
# Angles 0.1, 0.2, 0.4: on the old method the best rate halved from 3000 to 15000 steps.
#
# THE RULE, H4's, fixed before the numbers. gap15 = 3.5254 - best val:
#     >= +0.020   the lead persists
#     +0.010 to +0.020   decaying
#     <  +0.010   gone
# A best cell on an edge means extend first. One seed.
# PREDICTION: persists.
#
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_15k.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"

ROT=( 0.1 0.2 0.4 )
L="${ROT[$SLURM_ARRAY_TASK_ID]}"
OUT="$DATA_p330/runs/stepnorm15k"
mkdir -p "$OUT"

echo "node $(hostname)  task $SLURM_ARRAY_TASK_ID  step_norm  angle=$L  t_fac=1  adamw=8e-3  batch=512  steps=15000  seed=0"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --train-steps 15000 \
        --ngd-t-fac 1 --precision bf16 --seed 0 --device cuda \
        --log-every 23 --eval-every 1000 \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
