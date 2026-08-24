#!/bin/bash
#SBATCH -A p330
#SBATCH -J ngd-sweep
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
# 64 CPUs against 8 GPUs on these nodes, so 8 per job is the fair share.
# Without this SLURM hands out one core and the data loader becomes the
# bottleneck long before the GPU does.
#SBATCH --cpus-per-task=8
#SBATCH -p rtx
#SBATCH --time=08:00:00
# Sweeps live on rtx: 32 GPUs against b200's 16, and a 1.2B run fits the 24h
# cap several times over.
# A learning-rate sweep as a job array. Every optimizer has its own optimum --
# on the toy problem they spanned 4.6e-3 to 2.2e+1 -- so comparing at one
# shared rate measures which is better at that rate, not which is better. This
# is the most common way an optimizer comparison goes wrong.
#
#   sbatch --array=0-11%8 scripts/sbatch/sweep.sbatch
#
# `%4` caps concurrency; raise it to whatever the queue allows.

# NOTE: SLURM opens the -o/-e paths before this script runs, so `logs/` has to
# exist in the submission directory already. `mkdir -p logs` once, before the
# first sbatch, or the job dies with no output explaining why.
set -euo pipefail

REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
RUNS="${RUNS:-$DATA_p330/runs}"

OPTIMIZERS=(ngd pion_ablated pion)
LRS=(1e-4 3e-4 1e-3 3e-3)

O=$(( SLURM_ARRAY_TASK_ID / ${#LRS[@]} ))
L=$(( SLURM_ARRAY_TASK_ID % ${#LRS[@]} ))
OPT="${OPTIMIZERS[$O]}"
LR="${LRS[$L]}"

# Sweeps run at 1x Chinchilla -- 1.2B tokens for a 60M model, 9155 steps --
# which is an eighth of the published budget and enough to rank configurations.
# The full 9.6B is spent only on the final table.
STEPS=9155

echo "task ${SLURM_ARRAY_TASK_ID}: ${OPT} lr=${LR} steps=${STEPS}"

apptainer exec --env PYTHONUNBUFFERED=1 --nv \
    --bind "$REPO:$REPO" --bind "$DATA_p330:$DATA_p330" \
    --pwd "$REPO" \
    "$SIF" \
    python -m harness.run \
        --optimizer "$OPT" --lr "$LR" --train-steps "$STEPS" \
        --data-path "$DATA/c4_train.bin" \
        --val-path "$DATA/c4_val.bin" \
        --out-dir "$RUNS"
