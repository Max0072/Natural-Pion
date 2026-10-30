#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepnorm
#SBATCH --array=0-5
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Is alpha's benefit the control of the step's length? (journal 2026-10-30.)
#
# THE ONE VARIABLE, against tfac1 (t_fac 1, alpha off): the step is divided by its own
# sigma, scripts/probes/step_norm.py. `--lr` is now the nominal angle c*sigma at the peak,
# so the grid is new, bracketing the arm's real medians (about 1 at the peak).
# Side effect, small: AdamW's floor lr_min*(adamw/lr) changes with --lr, as it does
# across every rate grid; it matters only in the last steps.
#
# WHY. In tfac1 the nominal angle of whole attention layers runs away and stays there
# (c*sigma up to 1e5; wq, wk, wv of 4 of 8 blocks at rot 2e-3; 1 of 8 at 1e-3), and alpha
# reduces it. Here a runaway is impossible.
#
# THE RULE, fixed before the numbers. Vn = best val here, Va = best of tfac1_alpha
# (3.7843, or lower once 8e-3 and 1.6e-2 are in). N = (3.8419 - Vn) / (3.8419 - Va):
#     N >= 0.8       length control is what alpha gives; quad/curv is not needed
#     0.2 <= N < 0.8 partly
#     N < 0.2        no
# A best cell on an edge means extend first. N > 1 would mean length control beats alpha.
# One seed; the noise is about 0.005.
#
# PREDICTION. N >= 0.8. Equal angle for all layers also removes alpha's between-layer
# variation (3.9x), which is the way this could fall short.
#
#   sbatch -x rtx6004 scripts/sbatch/step_norm.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/stepnorm"

ROT=( 0.05 0.1 0.2 0.4 0.8 1.6 )
TFAC=( 1 1 1 1 1 1 )
L="${ROT[$SLURM_ARRAY_TASK_ID]}"
T="${TFAC[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task $SLURM_ARRAY_TASK_ID  step_norm trust=none  t_fac=$T  rot=$L  adamw=8e-3  batch=512  steps=3000  seed=0"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --train-steps 3000 \
        --ngd-t-fac "$T" --precision bf16 --seed 0 --device cuda \
        --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
