#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepatt
#SBATCH --array=0-3%4
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Is an equal angle between layers optimal? One variable against job 393882 at angle 0.4: attention's angle x m, FFN stays. No momentum, t_fac 1, cosine, seed 0, 3000 steps.
# M = 3.7399 - best val.
#     M >= +0.020   unequal angles are better: report which
#     M <  +0.010   equal is as good as anything here
#     between       one seed cannot say
# Best at m = 1/4 or 4 means extend. PREDICTION: M < +0.010.
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_attn.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
MS=( 0.25 0.5 2 4 ); M="${MS[$T]}"
OUT="$DATA_p330/runs/stepnormattn/m$M"; mkdir -p "$OUT"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env STEP_NORM_ATTN_MULT="$M" \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py --optimizer ngd-pion-s --lr 0.4 --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 --train-steps 3000 --ngd-t-fac 1 \
        --precision bf16 --seed 0 --device cuda --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
