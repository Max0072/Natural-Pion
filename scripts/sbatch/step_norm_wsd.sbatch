#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepwsd
#SBATCH --array=0-1
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# WSD for the angle: constant at the peak, then linear to the floor over the last fraction. Batched t_fac 25 build, seed 0, 3000 steps, peak 0.2.
#   0  decay over the last 20%      1  decay over the last 10%
# d = val - 3.7242 (same build, cosine). |d| < 0.010: WSD is as good, and long horizons can be tuned by branching decays off one
# constant run. d >= +0.010 worse, <= -0.010 better.
# PREDICTION: 20% |d| < 0.010, medium confidence; 10% worse than 20%.
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_wsd.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
NAME=( decay0.2 decay0.1 ); DEC=( 0.2 0.1 ); N="${NAME[$T]}"
OUT="$DATA_p330/runs/stepnormwsd/$N"; mkdir -p "$OUT"; true
echo "node $(hostname) task $T $N"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env STEP_NORM_A_LAST=6 --env STEP_NORM_D_LAST=6 --env STEP_NORM_FAST=1 --env STEP_NORM_BATCH=1 \
    --env STEP_NORM_DECAY="${DEC[$T]}" \
    --env STEP_NORM_SCHEDULE=wsd --env STEP_NORM_PEAK=0.2 --env STEP_NORM_STEPS=3000 \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py --optimizer ngd-pion-m --ngd-momentum lie --ngd-beta1 0.9 --lr 0.2 --adamw-lr 8e-3 --ngd-trust none \
        --ngd-beta 0.5 --ngd-beta-backward 0.5 --train-steps 3000 --ngd-t-fac 25 \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 --precision bf16 --seed 0 --device cuda --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
