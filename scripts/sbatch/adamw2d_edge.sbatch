#!/bin/bash
#SBATCH -A p330
#SBATCH -J a2dedge
#SBATCH --array=0-13%7
#SBATCH -w rtx6002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:30:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Extending job 304581, whose every cell put its AdamW optimum on the top edge.
#
#     ngd-pion-s B=512   best 3.8276 at rot 6e-3, adamw 2e-3   <- adamw edge
#     ngd-pion-s B=2048  best 4.0852 at rot 6e-4, adamw 4e-3   <- both edges
#     pion       B=2048  best 4.3847 at rot 3e-4, adamw 4e-3   <- adamw edge
#
# AdamW owns 32 879 104 parameters, **56.5% of the model**, and `1e-3` -- the
# value pinned in every run this project has ever made -- is beaten by 2e-3 even
# at the standard batch, by 0.047. At 2048 sequences, where a fixed token budget
# gives it a quarter as many updates, tuning it is worth **0.208**. That is four
# times as much, which is exactly the ratio predicted if the pinning was what
# made the larger batch look bad.
#
# It moved the headline number it was invoked to explain:
#
#     pinned adamw = 1e-3       3.8744 vs 4.2928    gap 0.418
#     each at its own adamw     3.8276 vs 4.0852    gap 0.258
#
# So tuning closed 0.16 of 0.42 and the small batch still leads. But nothing has
# reached its optimum yet, and the larger batch is moving four times faster, so
# the question is open rather than answered. Grids that stop on an edge have
# wasted more of this project's time than any other single mistake.
#
# The rotational rate at 2048 is also on its lower edge at 6e-4, so it is
# extended down as well; if AdamW being wrong is what shortened it, the two
# should move back together.
#
# High AdamW rates may simply diverge. That is a useful outcome -- it bounds the
# grid from above, which is what an edge cannot do.
#
#   sbatch -p rtx scripts/sbatch/adamw2d_edge.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/adamw2d"

#      --------- ngd B=512 ---------  ------------- ngd B=2048 -------------  ----- pion B=2048 -----
OPT=(  ngd-pion-s ngd-pion-s ngd-pion-s ngd-pion-s ngd-pion-s ngd-pion-s ngd-pion-s ngd-pion-s ngd-pion-s ngd-pion-s pion pion pion pion )
BS=(   512        512        512        512        2048       2048       2048       2048       2048       2048       2048 2048 2048 2048 )
ROT=(  2e-3       2e-3       6e-3       6e-3       6e-4       6e-4       2e-3       2e-3       2e-4       2e-4       1e-3 1e-3 3e-3 3e-3 )
ADW=(  4e-3       8e-3       4e-3       8e-3       8e-3       1.6e-2     8e-3       1.6e-2     4e-3       8e-3       8e-3 1.6e-2 8e-3 1.6e-2 )

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
