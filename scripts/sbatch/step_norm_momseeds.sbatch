#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepmsd
#SBATCH --array=1-3%1
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Seeds 1-3 for the best cell: F^-1, Lie momentum, angle 0.1, t_fac 1, 3000 steps (seed 0 3.7032).
# Paired gaps over seeds 0-3, against Pion 3.8412, 3.8377, 3.8369, 3.8417 and against no momentum 3.7399, 3.7436, 3.7419, 3.7416:
#     all four positive and mean/SE > 2.35 (t, df 3, one-sided 0.05)   holds
# PREDICTION: holds against both.
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_momseeds.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"

T=$SLURM_ARRAY_TASK_ID; P=fisher; L=0.1
OUT="$DATA_p330/runs/stepnormmomseeds"
mkdir -p "$OUT"

echo "node $(hostname)  task $T  precond=$P  momentum=lie  angle=$L  t_fac=1  adamw=8e-3  batch=512  steps=3000  seed=$T"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env STEP_NORM_PRECOND="$P" \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py \
        --optimizer ngd-pion-m --ngd-momentum lie --ngd-beta1 0.9 \
        --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --train-steps 3000 \
        --ngd-t-fac 1 --precision bf16 --seed "$T" --device cuda \
        --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
