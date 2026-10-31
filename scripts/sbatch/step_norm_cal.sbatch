#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepcal
#SBATCH --array=0
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Calibrates the grids for the fisher and rms norms: sqrt(curv)/sigma and the RMS base/sigma per layer, along the angle-0.4 trajectory.
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_cal.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
L=0.4
OUT="$DATA_p330/runs/stepnormcal"; mkdir -p "$OUT"; rm -f "$OUT/cal.jsonl"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env STEP_NORM_CAL="$OUT/cal.jsonl" \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --precision bf16 --seed 0 --device cuda \
        --train-steps 3000 --max-steps 700 --ngd-t-fac 1 --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
