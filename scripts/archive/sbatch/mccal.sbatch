#!/bin/bash
#SBATCH -A p330
#SBATCH -J mccal
#SBATCH --array=0-1%2
#SBATCH -w rtx6002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# How much does the MC Fisher change the scale of F? Calibration, not a result.
#
# `mc_every = 1` builds `D` from labels drawn from the model instead of from the
# data, which is the true Fisher rather than the empirical one. Those deltas
# have a different magnitude -- early in training the model half-agrees with its
# own samples while the data surprises it -- so `F` changes scale and the
# natural step length changes with it.
#
# Comparing the two at a shared `eta` would therefore be exactly the error this
# project has a standing rule against. It also does not need guessing:
#
#     quad = <G, F^-1 G>   scales as 1/k when F scales as k
#     pred_drop = eta * alpha * quad, and alpha is ~1 on a fresh basis
#
# so a hundred steps of each at one `eta` reads the factor straight off
# `pred_drop`, and the sweep's grid gets centred on the measurement instead of
# on what looks reasonable. Seven grid-edge errors in a single day is why.
#
# 100 steps is enough for a *scale* and is not enough to rank anything. Nothing
# about which arm trains better may be read from this job.
#
#   sbatch -p rtx scripts/sbatch/mccal.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/mccal"

MC=("0" "1")
M="${MC[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  mc_every=$M  eta=2e-2  100 steps  (calibration only)"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr 2e-2 \
        --ngd-fisher-mc-every "$M" \
        --rho-every 5 --rho-micro 128 \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --max-steps 100 --log-every 5 --eval-every 1000 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
