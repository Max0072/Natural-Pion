#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepinit
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# The spectrum is frozen at initialisation, so the initial spectrum is a property of the trained model.
# Our build (batched, t_fac 25, A and D last 6, momentum 0.9, cosine), angle 0.2, 3000 steps, seed 0.
# Base, normal(0, 0.02): 3.7242. Only the 56 rotated matrices change; embedding and head keep normal(0, 0.02).
#   0-2  power-law tail on the top 20% of a normal bulk, alpha 2 / 3 / 5 (top sv x7.6 / x2.4 / smaller; more energy)
#   3    flat spectrum at sqrt(fan_out/fan_in) (init_pl_alpha inf)
#   4    orthogonal, every sv 1
# Rule: better if val <= 3.7142 (base - 0.010); the best then goes to 15000 steps, and Pion gets the same init.
# PREDICTION (low confidence): alpha 2-3 better by 0.005-0.02; flat and orthogonal worse.
#   sbatch -x rtx6003,rtx6004 -t 01:30:00 --array=0-4 scripts/sbatch/step_norm_init.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
NAME=( pl2 pl3 pl5 flat orth )
ARGS=( "--init-pl-alpha 2" "--init-pl-alpha 3" "--init-pl-alpha 5" "--init-pl-alpha inf" "--init orthogonal --init-gain 1" )
N="${NAME[$T]}"; OUT="$DATA_p330/runs/stepnorminit/$N"; mkdir -p "$OUT"
echo "node $(hostname) task $T $N ${ARGS[$T]}"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env STEP_NORM_A_LAST=6 --env STEP_NORM_D_LAST=6 --env STEP_NORM_FAST=1 --env STEP_NORM_BATCH=1 \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py --optimizer ngd-pion-m --ngd-momentum lie --ngd-beta1 0.9 --lr 0.2 --ngd-trust none \
      --ngd-beta 0.5 --ngd-beta-backward 0.5 --ngd-t-fac 25 ${ARGS[$T]} \
      --batch-sequences 512 --micro-batch 256 --rho-micro 128 --precision bf16 --seed 0 --device cuda --log-every 23 \
      --eval-every 500 --train-steps 3000 --adamw-lr 8e-3 --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
