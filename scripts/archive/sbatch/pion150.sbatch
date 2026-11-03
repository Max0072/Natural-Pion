#!/bin/bash
#SBATCH -A p330
#SBATCH -J pion150
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Where does the method that actually wins sit on the 150-step board?
#
# Every `pion` run on disk is full length (73241 steps, eight of them, val
# 3.3719 to 3.4432). The published configuration has **never been run at 150
# steps**, so the 150-step board contains only `pion_ablated` -- which is the
# worst arm measured, 0.40 below the do-nothing line.
#
# This is a sharp control. `pion` is demonstrably the best optimizer in this
# harness at full length. If at 150 steps it lands BELOW the inert control's
# 5.7071, then the 150-step proxy is not merely noisy but **anti-correlated**
# with the quantity the project cares about, and no ranking taken at that
# horizon -- which is every sweep in this repository -- can be trusted.
#
# Three `eta`, not one. `1e-3` is their published value and the one that
# produces 3.37 at length; the neighbours are there so the conclusion cannot be
# answered with "its 150-step optimum is elsewhere".
#
# `--adamw-lr 1e-3` pins AdamW as it is pinned for every other arm on the
# board. Their published design ties the two together, but a tied rate would
# make this incomparable with everything it is being placed next to.
#
#   sbatch -p rtx -w rtx6002,rtx6003 scripts/sbatch/pion150.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/control"

ETAS=(3e-4 1e-3 3e-3)
ETA="${ETAS[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID:-?}  pion eta=$ETA"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer pion --lr "$ETA" \
        --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --max-steps 150 --log-every 30 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
