#!/bin/bash
#SBATCH -A p330
#SBATCH -J steprp
#SBATCH --array=0-1
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# RProp's rule on the angle (Riedmiller and Braun 1993): one multiplier m on every layer, from a global vote of
# sign(<G_t, G_t-1>) on raw generators; > 0: m *= 1.2, < 0: m *= 0.5, m in [1e-3, 4]. m rises only while more than
# 79% of votes agree, so it should fall by itself once noise dominates. Batched t_fac 25 build, seed 0, 3000 steps, peak 0.2.
#   0  cosine x m      does adaptation on top of the schedule help?      d = val - 3.7242 (same build without RProp)
#   1  constant x m    can RProp make the decay itself? constant alone gave 3.8313 (t_fac 1, no momentum)
# d <= -0.010 better, |d| < 0.010 no change, >= +0.010 worse. Read m's trajectory in rprop.jsonl.
# PREDICTION: task 0 |d| < 0.010; task 1 worse than 0 but far better than constant alone; m falls in the second half. Low confidence.
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_rprop.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
NAME=( cosine constant ); N="${NAME[$T]}"
OUT="$DATA_p330/runs/stepnormrprop/$N"; mkdir -p "$OUT"; rm -f "$OUT/rprop.jsonl"
echo "node $(hostname) task $T $N"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env STEP_NORM_A_LAST=6 --env STEP_NORM_D_LAST=6 --env STEP_NORM_FAST=1 --env STEP_NORM_BATCH=1 \
    --env STEP_NORM_RPROP=1 --env STEP_NORM_RPROP_LOG="$OUT/rprop.jsonl" \
    --env STEP_NORM_SCHEDULE="$N" --env STEP_NORM_PEAK=0.2 --env STEP_NORM_STEPS=3000 \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py --optimizer ngd-pion-m --ngd-momentum lie --ngd-beta1 0.9 --lr 0.2 --adamw-lr 8e-3 --ngd-trust none \
        --ngd-beta 0.5 --ngd-beta-backward 0.5 --train-steps 3000 --ngd-t-fac 25 \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 --precision bf16 --seed 0 --device cuda --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
