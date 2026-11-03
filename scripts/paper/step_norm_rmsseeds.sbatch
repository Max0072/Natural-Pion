#!/bin/bash
#SBATCH -A p330
#SBATCH -J steprsd
#SBATCH --array=1-3%1
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# RMS on F^-1 G against the spectral norm, paired over seeds 0-3. RMS rot 2.4e-3, seed 0 3.7540. Spectral at 0.4, no momentum: 3.7399, 3.7436, 3.7419, 3.7416.
# gap = RMS - spectral per seed.
#     all four positive, mean/SE > 2.35, mean >= 0.010   spectral is better
#     mean < 0.010                                       the norm does not matter
#     else                                               unresolved
# PREDICTION: spectral better, mean about 0.014.
#   sbatch -x rtx6003,rtx6004 scripts/paper/step_norm_rmsseeds.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
OUT="$DATA_p330/runs/stepnormrmsseeds"; mkdir -p "$OUT"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --env STEP_NORM_NORM=rms \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/step_norm.py --optimizer ngd-pion-s --lr 2.4e-3 --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 --train-steps 3000 --ngd-t-fac 1 \
        --precision bf16 --seed "$T" --device cuda --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
