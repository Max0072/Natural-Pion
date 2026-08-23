#!/bin/bash
#SBATCH -A p330
#SBATCH -J ngd-pion
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# --partition is deliberately absent: fill it from `sinfo` and pass it on the
# command line, so the same script serves both the RTX and B200 pools.
#
#   sbatch -p <partition> scripts/sbatch/train.sbatch --optimizer ngd --lr 1e-3
#   sbatch -p <partition> scripts/sbatch/train.sbatch --anchor bilateral
#
# One run, one GPU, no distributed anything. A 60M model fits on a single card,
# so N cards are worth far more as N concurrent experiments than as one faster
# experiment, and the sweeps this project needs are wide rather than deep.

set -euo pipefail
mkdir -p logs

REPO="${REPO:-$HOME/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
RUNS="${RUNS:-$DATA_p330/runs}"

echo "node        $(hostname)"
echo "gpu         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo unknown)"
echo "job         ${SLURM_JOB_ID:-none} on ${SLURM_JOB_PARTITION:-unknown}"
echo "args        $*"

apptainer exec --nv \
    --bind "$REPO:$REPO" --bind "$DATA_p330:$DATA_p330" \
    --pwd "$REPO" \
    "$SIF" \
    python -m harness.run \
        --data-path "$DATA/c4_train.bin" \
        --val-path "$DATA/c4_val.bin" \
        --out-dir "$RUNS" \
        "$@"
