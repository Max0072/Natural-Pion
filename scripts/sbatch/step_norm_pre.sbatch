#!/bin/bash
#SBATCH -A p330
#SBATCH -J steppre
#SBATCH --array=0-8%4
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Normalise before F^-1, not after. One variable against job 393882: c = rot_t / sigma(G), X = F^-1 G unchanged.
# No momentum, t_fac 1, cosine, seed 0, 3000 steps. Sent before calibration 394427 ran, at the user's word, so the grid is
# wide, a decade apart: --lr is no angle here, the logged `angle` is the real c * sigma. Refine around the best afterwards.
# Calibration 394427: sigma(X)/sigma(G) has median 3e4, so rot about 1.3e-5 gives a median angle 0.4. Tasks 1-4 were far too large
# (angle_min 16 and up) and were cancelled; tasks 5-8 replace them.
# P = best val - 3.7399.
#     P >= +0.020   after is better
#     |P| < 0.010   the order does not matter
#     P <= -0.020   before is better
#     else          one seed cannot say
# Edge means extend, and so does a decade grid's best cell: refine before concluding. PREDICTION: P >= +0.020.
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_pre.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
ROT=( 1e-4 1e-3 1e-2 1e-1 1  1e-6 3e-6 1e-5 3e-5 ); L="${ROT[$T]}"
OUT="$DATA_p330/runs/stepnormpre"; mkdir -p "$OUT"
echo "node $(hostname) task $T norm=pre rot=$L"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env STEP_NORM_NORM=pre \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --precision bf16 --seed 0 --device cuda \
        --train-steps 3000 --ngd-t-fac 1 --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
