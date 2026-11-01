#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepprf
#SBATCH --array=0-1
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Where a step's time goes. Lie momentum, angle 0.2, 400 steps, no evaluation in the window read. Task 0 t_fac 25, task 1 t_fac 1.
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_prof.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"

TF=( 25 1 ); T=$SLURM_ARRAY_TASK_ID; F="${TF[$T]}"; L=0.2
OUT="$DATA_p330/runs/stepnormprof/tfac$F"; mkdir -p "$OUT"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env STEP_NORM_PROFILE="$OUT/profile.json" --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py --optimizer ngd-pion-m --ngd-momentum lie --ngd-beta1 0.9 --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 --train-steps 3000 --max-steps 400 --ngd-t-fac "$F" \
        --precision bf16 --seed 0 --device cuda --log-every 23 --eval-every 100000 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
