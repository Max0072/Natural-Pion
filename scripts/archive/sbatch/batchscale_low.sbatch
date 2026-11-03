#!/bin/bash
#SBATCH -A p330
#SBATCH -J bslow
#SBATCH --array=0-5%6
#SBATCH -w rtx6002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Extending the B=2048 grids downward, because job 300530 put both of them on
# their lower edge and edge results are not optima.
#
#     ngd-pion-s  B=2048   0.006 -> 4.2215   <- best, and the lowest tested
#     pion        B=2048   0.001 -> 4.0781   <- best, and the lowest tested
#
# Both B=512 grids came out interior and stand as measured: `ngd-pion-s` 3.7149
# at 0.006 against `pion` 3.7386 at 0.001, so the smaller batch already has its
# clean one-variable comparison.
#
# The grids were shifted *up* for the larger batch on the reasoning that less
# noise permits a longer step. The data says the opposite, and choosing a range
# by argument rather than by measurement is the error this project has made
# repeatedly. Here it is corrected the only way that works: extend until the
# optimum is interior, and if these land on an edge again, extend again.
#
# Same token budget as the rest -- 1500 steps at 2048 sequences is 786M tokens,
# identical to 6000 at 512 -- and the same everything else, so the arms remain
# comparable to those already on disk.
#
#   sbatch -p rtx scripts/sbatch/batchscale_low.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/batchscale"

OPT=( ngd-pion-s ngd-pion-s ngd-pion-s  pion   pion    pion   )
LR=(  2e-4       6e-4       2e-3        3e-5   1e-4    3e-4   )

O="${OPT[$SLURM_ARRAY_TASK_ID]}"
L="${LR[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID}  $O  batch=2048  lr=$L  steps=1500"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer "$O" --lr "$L" \
        --batch-sequences 2048 --micro-batch 256 \
        --train-steps 1500 \
        --rho-micro 128 \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --log-every 23 --eval-every 125 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
