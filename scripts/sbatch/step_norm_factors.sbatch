#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepfac
#SBATCH --array=0-5%4
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# How A and D are collected. Against t_fac 25, Lie momentum, angle 0.2, 3000 steps, seed 0: 3.7227, 0.844 s a step.
#   0  D last 6 steps, beta_D 0.5            speed: 98.4% of D's weight, saves about 0.06 s
#   1  beta_D 0.9, every step                quality: D now rests on 1-2 batches
#   2  beta_D 0.95, every step               quality
#   3  A every 2nd step, beta_A 0.9025       speed: same time constant
#   4  A last 6 steps, beta_A 0.5            speed: does A need a long window?
#   5  A and D last 6 steps, both beta 0.5   speed: largest saving, 0.11 s
# d = val - 3.7227. Seed spread with momentum is about 0.006.
#     d <= -0.020   better
#     |d| < 0.010   no change
#     d >= +0.020   worse
#     else          one seed cannot say
# One angle: a change of beta could move the best angle, so a loss here is not final. Step cost read from the log's window.
# PREDICTION: 0 no change; 1, 2 no change or better; 3 no change; 4, 5 worse.
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_factors.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
NAME=( dlast6 betad09 betad095 aevery2 alast6 adlast6 ); N="${NAME[$T]}"
BA=( 0.95 0.95 0.95 0.9025 0.5 0.5 ); BD=( 0.5 0.9 0.95 0.5 0.5 0.5 )
AL=( 0 0 0 0 6 6 ); DL=( 6 0 0 0 0 6 ); AE=( 1 1 1 2 1 1 )
OUT="$DATA_p330/runs/stepnormfactors/$N"; mkdir -p "$OUT"
echo "node $(hostname) task $T $N"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env STEP_NORM_A_LAST="${AL[$T]}" --env STEP_NORM_D_LAST="${DL[$T]}" --env STEP_NORM_A_EVERY="${AE[$T]}" \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py --optimizer ngd-pion-m --ngd-momentum lie --ngd-beta1 0.9 --lr 0.2 --adamw-lr 8e-3 --ngd-trust none \
        --ngd-beta "${BA[$T]}" --ngd-beta-backward "${BD[$T]}" \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 --train-steps 3000 --ngd-t-fac 25 \
        --precision bf16 --seed 0 --device cuda --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
