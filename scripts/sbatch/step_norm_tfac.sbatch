#!/bin/bash
#SBATCH -A p330
#SBATCH -J steptf
#SBATCH --array=0-4%4
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Is the fresh basis needed once the step is normalised? One variable against job 393882: t_fac 25, not 1.
# No momentum, cosine, seed 0, 3000 steps. A t_fac 1 step costs 1.75 s against Pion's 0.48-0.67 s.
# D = best val - 3.7399.
#     D <  0.010   the fresh basis is not needed
#     D >= 0.020   it is
#     between      one seed cannot say
# Edge means extend. PREDICTION: D >= 0.020.
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_tfac.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
ROT=( 0.1 0.2 0.4 0.8 1.6 ); L="${ROT[$T]}"
OUT="$DATA_p330/runs/stepnormtfac"; mkdir -p "$OUT"
echo "node $(hostname) task $T t_fac=25 angle=$L"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --precision bf16 --seed 0 --device cuda \
        --train-steps 3000 --ngd-t-fac 25 --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
