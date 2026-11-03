#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepm15
#SBATCH --array=0
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 14:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Best configuration by steps at 15000: t_fac 1, Lie momentum, angle 0.1, seed 0.
# gap15 = 3.5254 - val: >= +0.020 persists, +0.010..+0.020 decaying, < +0.010 gone. Momentum's effect = 3.4710 - val: >= +0.010 it still pays at this horizon.
# PREDICTION: persists, momentum's effect >= +0.010. One angle: the optimum moved with horizon before, so an edge cannot be judged here.
#   sbatch -x rtx6003,rtx6004 scripts/paper/step_norm_mom15k.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"

L=0.1
OUT="$DATA_p330/runs/stepnormmom15k"; mkdir -p "$OUT"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py --optimizer ngd-pion-m --ngd-momentum lie --ngd-beta1 0.9 --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 --train-steps 15000 --ngd-t-fac 1 \
        --precision bf16 --seed 0 --device cuda --log-every 23 --eval-every 1000 \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
