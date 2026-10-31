#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepnrm
#SBATCH --array=0-9%4
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Another norm after F^-1. One variable against job 393882: what the angle fixes. No momentum, t_fac 1, cosine, seed 0, 3000 steps.
#     fisher  c = rot_t / sqrt(X F X)                       fixes the predicted KL of the step
#     rms     c = rot_t * sqrt(m n) / ||W X_in + X_out W||_F  Pion's RMS on F^-1 G
# Calibration 394427: both base/sigma have median about 3e-3, so rot 3e-4 ... 4.8e-3 spans median angles 0.1 ... 1.6.
# N = best val - 3.7399 per norm.
#     N >= +0.020   the spectral norm is better
#     |N| < 0.010   it does not matter which
#     N <= -0.020   this norm is better
#     else          one seed cannot say
# Edge means extend. PREDICTION: rms |N| < 0.010; fisher N >= +0.020, because its angle floats.
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_norm.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
NM=( fisher fisher fisher fisher fisher  rms rms rms rms rms ); N="${NM[$T]}"
ROT=( 3e-4 6e-4 1.2e-3 2.4e-3 4.8e-3  3e-4 6e-4 1.2e-3 2.4e-3 4.8e-3 ); L="${ROT[$T]}"
OUT="$DATA_p330/runs/stepnormnorm/$N"; mkdir -p "$OUT"
echo "node $(hostname) task $T norm=$N rot=$L"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env STEP_NORM_NORM="$N" \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --precision bf16 --seed 0 --device cuda \
        --train-steps 3000 --ngd-t-fac 1 --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
