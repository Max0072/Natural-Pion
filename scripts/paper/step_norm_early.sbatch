#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepearly
#SBATCH --array=0-1
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:30:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Early decay (the user's hypothesis): at 15000 a 0.2 peak leads to step ~1000, is caught by 2000 and loses after
# (0.2 / 0.1 / 0.05: 4.1129 / 4.1130 / 4.1525 at 999, 3.9094 / 3.8779 / 3.8735 at 2999, final 3.4940 / 3.4777 / 3.4720).
# Cosine still holds 85% of the peak at a quarter of the run. Schedule: angle = peak (1 - t/T) / sqrt(1 + t/1000), zero at the end.
# Batched t_fac 25 build, seed 0, 15000 steps, peak 0.2 / 0.1. Control: cosine 0.2 3.4940, best cosine 0.05 3.4720.
#   val <= 3.4620  early decay beats the best cosine; |val - 3.4720| < 0.010 only the scale matters; worse otherwise.
# PREDICTION: peak 0.2 beats cosine 0.2 by >= 0.010 (about even odds); beating 3.4720, low confidence.
#   sbatch -x rtx6003,rtx6004 scripts/paper/step_norm_early.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
PK=( 0.2 0.1 ); L="${PK[$T]}"
OUT="$DATA_p330/runs/stepnormearly/isqrt$L"; mkdir -p "$OUT"
echo "node $(hostname) task $T peak $L"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env STEP_NORM_A_LAST=6 --env STEP_NORM_D_LAST=6 --env STEP_NORM_FAST=1 --env STEP_NORM_BATCH=1 \
    --env STEP_NORM_SCHEDULE=isqrt --env STEP_NORM_PEAK="$L" --env STEP_NORM_STEPS=15000 --env STEP_NORM_T0=1000 \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py --optimizer ngd-pion-m --ngd-momentum lie --ngd-beta1 0.9 --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --ngd-beta 0.5 --ngd-beta-backward 0.5 --train-steps 15000 --ngd-t-fac 25 \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 --precision bf16 --seed 0 --device cuda --log-every 23 --eval-every 1000 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
