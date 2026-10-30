#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepnrms
#SBATCH --array=1-3%1
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Is step_norm's 3.7399 (seed 0, angle 0.4) real? (journal 2026-10-30.)
#
# THE ONE VARIABLE. Flags of step_norm.sbatch task 3; only `--seed` changes, 1, 2, 3.
# `%1`: one seed at a time, concurrent runs on one node share a seed or starve for I/O
# (memory, ADR 0010 was a one-off waiver).
#
# THE RULE, fixed before the numbers, as H1 in docs/PLAN.md. Paired with `pion`, same seed:
#   seed 0  3.8412   seed 1  3.8377   seed 2  3.8369   seed 3  3.8417
# The four gaps pion - step_norm must all be positive with mean / standard error > 2.35
# (one-sided 5%, 3 degrees of freedom). Same test against the with-alpha arm
#   seed 0  3.7954   seed 1  3.8015   seed 2  3.8040   seed 3  3.8052
# which is the claim that this beats the method as it stood.
# Angle 0.4 was picked on seed 0, so seed 0's gap leans favourable; seeds 1-3 are not.
#
#   sbatch -x rtx6004 scripts/sbatch/step_norm_seeds.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/stepnormseeds"

L=0.4
T=1
SEED="$SLURM_ARRAY_TASK_ID"

echo "node $(hostname)  task $SLURM_ARRAY_TASK_ID  step_norm trust=none  t_fac=$T  rot=$L  adamw=8e-3  batch=512  steps=3000  seed=$SEED"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --train-steps 3000 \
        --ngd-t-fac "$T" --precision bf16 --seed "$SEED" --device cuda \
        --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
