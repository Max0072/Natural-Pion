#!/bin/bash
#SBATCH -A p330
#SBATCH -J a2dhead
#SBATCH --array=0-8%7
#SBATCH -w rtx6002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:30:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# The head-to-head at B=512 with BOTH optimizers tuned. The number the project
# actually needs, and the one cell that was cut for speed.
#
# WHERE THE GRID STANDS after jobs 304581 and 304973, at 393M tokens:
#
#     ngd-pion-s B=512   3.7954  rot 6e-3, adamw 8e-3    adamw still on the edge
#     ngd-pion-s B=2048  4.0514  rot 6e-4, adamw 8e-3    INTERIOR, resolved
#     pion       B=2048  4.1845  rot 1e-3, adamw 1.6e-2  adamw still on the edge
#     pion       B=512   never run
#
# Two things are already settled by that. The small batch still wins for us --
# 3.7954 against 4.0514, and the small-batch cell has not even reached its own
# optimum -- so tuning AdamW did not overturn it and the batch story is closed.
# And the rotational optimum at 2048 really does sit ten times below the one at
# 512: with AdamW now eight times higher than the pinned value the cell is fully
# interior, `2e-4` and `2e-3` both worse. That is a property of the method, not
# an artefact of an undertrained embedding, and it is unexplained.
#
# What is missing is the comparison the paper turns on: `ngd-pion-s` against
# `pion` at the batch anyone would actually use, with each given its own two
# rates. Tuning AdamW is worth 0.079 to us at this batch and 0.241 at the larger
# one; Pion gets whatever it is worth to Pion, measured rather than assumed.
#
# AdamW owns 56.5% of the model and every run this project has made pinned it at
# 1e-3. So this is not a refinement -- until it exists, no head-to-head number
# here rests on a tuned baseline.
#
# The last two arms close the two remaining edges, so that every quoted optimum
# is interior.
#
#   sbatch -p rtx scripts/sbatch/adamw2d_head.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/adamw2d"

#      ------------------- pion, B=512 -------------------  ngd512  pion2048
OPT=(  pion  pion  pion   pion  pion  pion   pion   ngd-pion-s pion   )
BS=(   512   512   512    512   512   512    512    512        2048   )
ROT=(  1e-3  1e-3  1e-3   3e-3  3e-3  3e-3   3e-4   6e-3       1e-3   )
ADW=(  2e-3  8e-3  1.6e-2 2e-3  8e-3  1.6e-2 8e-3   1.6e-2     3.2e-2 )

T=$SLURM_ARRAY_TASK_ID
O="${OPT[$T]}"; B="${BS[$T]}"; L="${ROT[$T]}"; A="${ADW[$T]}"
if [ "$B" = "512" ]; then STEPS=3000; EVERY=500; else STEPS=750; EVERY=125; fi

echo "node $(hostname)  task $T  $O  batch=$B  rot=$L  adamw=$A  steps=$STEPS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer "$O" --lr "$L" --adamw-lr "$A" \
        --batch-sequences "$B" --micro-batch 256 --rho-micro 128 \
        --train-steps "$STEPS" \
        --ngd-t-fac 25 --precision bf16 --seed 0 \
        --log-every 23 --eval-every "$EVERY" --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
