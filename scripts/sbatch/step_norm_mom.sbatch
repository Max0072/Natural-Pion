#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepmom
#SBATCH --array=0-9%4
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# The missing half of a 2x2: {F^-1, G} x {no momentum, Lie momentum beta1=0.9, as Pion}.
# The no-momentum half is jobs 393882 and 393994/394098. The G-against-Pion gap there
# confounded normalisation with momentum and retraction; F^-1 against G did not.
# Optimizer is ngd-pion-m: under ngd-pion-s the harness forces momentum to "none".
#
# Fm, Gm = best val with momentum, each over its own grid. Fisher gain with momentum = Gm - Fm.
#     gain >= 0.020   F^-1 matters beyond what momentum gives
#     gain <  0.010   momentum replaces F^-1: its gain was denoising
#     between         one seed cannot say; run seeds
# Momentum's effect = (best without) - (best with), per column, same thresholds.
# A best cell on an edge means extend before concluding.
# PREDICTION: G gains at least 0.10 from momentum, F^-1 gains less, gain with momentum >= 0.020.
#
#   sbatch -x rtx6003,rtx6004 scripts/sbatch/step_norm_mom.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"

PRE=( fisher fisher fisher fisher fisher  none none none none none )
ROT=( 0.1 0.2 0.4 0.8 1.6  0.05 0.1 0.2 0.4 0.8 )
T=$SLURM_ARRAY_TASK_ID; P="${PRE[$T]}"; L="${ROT[$T]}"
OUT="$DATA_p330/runs/stepnormmom/$P"
mkdir -p "$OUT"

echo "node $(hostname)  task $T  precond=$P  momentum=lie  angle=$L  t_fac=1  adamw=8e-3  batch=512  steps=3000  seed=0"
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
        --ngd-t-fac 1 --precision bf16 --seed 0 --device cuda \
        --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
