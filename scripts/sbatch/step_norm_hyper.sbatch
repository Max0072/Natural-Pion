#!/bin/bash
#SBATCH -A p330
#SBATCH -J stephyp
#SBATCH --array=0-2
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 14:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Does a fast decay beat the cosine at 15000? At 15000 the 0.4 cosine trails the 0.1 cosine from step 1999 until the tail.
# angle = peak / (1 + t / t0), rotational angle only, AdamW stays on its cosine. No momentum, t_fac 1, seed 0.
# E = 3.4862 - best val.
#     E >= 0.020   the schedule pays
#     E <  0.010   it does not
#     between      one seed cannot say
# PREDICTION: E >= 0.020.
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_hyper.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
PK=( 0.4 0.4 0.8 ); T0=( 500 2000 500 ); L="${PK[$T]}"; Z="${T0[$T]}"
OUT="$DATA_p330/runs/stepnormhyper/t0$Z"; mkdir -p "$OUT"
echo "node $(hostname) task $T hyper peak=$L t0=$Z"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env STEP_NORM_SCHEDULE=hyper --env STEP_NORM_PEAK="$L" --env STEP_NORM_T0="$Z" --env STEP_NORM_STEPS=15000 \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --precision bf16 --seed 0 --device cuda \
        --train-steps 15000 --ngd-t-fac 1 --log-every 23 --eval-every 1000 \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
