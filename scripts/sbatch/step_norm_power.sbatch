#!/bin/bash
#SBATCH -A p330
#SBATCH -J steppow
#SBATCH --array=0-7%4
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Flatten the step's spectrum? X -> U (S / sigma)^p V^T. One variable against F^-1 with momentum, t_fac 1, 3000 steps, seed 0: p = 1, best 3.7032 at angle 0.1.
# p = 0 turns every plane by the same angle, p = 0.5 is between. Angles lower than at p = 1, since the total rotation is larger.
# Q = best val - 3.7032, per p.
#     Q <= -0.020   flatter is better
#     |Q| <  0.010  the spectrum's shape does not matter
#     Q >= +0.020   keeping the spectrum is better
#     else          one seed cannot say
# Edge means extend. PREDICTION: Q >= -0.010 for both, low confidence. The user expects different speeds in different directions to win.
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_power.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
PS=( 0 0 0 0 0.5 0.5 0.5 0.5 ); RS=( 0.003 0.01 0.03 0.1 0.003 0.01 0.03 0.1 ); PW="${PS[$T]}"; L="${RS[$T]}"
OUT="$DATA_p330/runs/stepnormpower/p$PW"; mkdir -p "$OUT"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env STEP_NORM_POWER="$PW" \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py --optimizer ngd-pion-m --ngd-momentum lie --ngd-beta1 0.9 --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 --train-steps 3000 --ngd-t-fac 1 \
        --precision bf16 --seed 0 --device cuda --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
