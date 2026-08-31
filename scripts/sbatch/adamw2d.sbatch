#!/bin/bash
#SBATCH -A p330
#SBATCH -J adamw2d
#SBATCH --array=0-35%7
#SBATCH -w rtx6002
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# AdamW owns 56.5% of the model and has never been tuned. Both rates, together.
#
# THE DEFECT THIS REPAIRS. `model.parameter_split` gives the rotational
# optimizer the 2-D weights that are neither embedding nor head: 25 296 896
# parameters. AdamW gets the other **32 879 104 -- 56.5% of the model**. Job
# 300530 held `adamw_lr = 1e-3` fixed across both batch sizes, so at 2048
# sequences more than half the network took a quarter as many updates at an
# unchanged per-step rate. The comparison between batches therefore varied two
# things, and the conclusion drawn from it -- that a small batch dominates at a
# fixed token budget -- is not established.
#
# THE TELL WAS ALREADY IN THE DATA. A larger batch should permit a *longer*
# step. The measured optimum moved the other way, from 6e-3 at 512 sequences to
# 6e-4 at 2048, a tenfold fall. The natural reading is that the rotation was
# being shortened while an undertrained embedding caught up -- which is a
# statement about AdamW, not about the batch.
#
# THE DESIGN. Both rates move together, because if the rotational optimum really
# is compensating for AdamW then neither can be found while the other is held
# wrong. Four cells, 3x3 each:
#
#     ngd-pion-s  B=512    rot {2e-3, 6e-3, 2e-2}   adamw {5e-4, 1e-3, 2e-3}
#     ngd-pion-s  B=2048   rot {6e-4, 2e-3, 6e-3}   adamw {1e-3, 2e-3, 4e-3}
#     pion        B=512    rot {3e-4, 1e-3, 3e-3}   adamw {5e-4, 1e-3, 2e-3}
#     pion        B=2048   rot {3e-4, 1e-3, 3e-3}   adamw {1e-3, 2e-3, 4e-3}
#
# The AdamW grids bracket both scaling rules: 4x the batch is 2e-3 under the
# square root and 4e-3 under the linear one, with 1e-3 kept as the unscaled
# control. **The B=512 cells are swept too**, or a tuned large batch would be
# compared against an untuned small one, which is the same error in the mirror.
#
# 786M tokens either way -- 6000 steps at 512 sequences, 1500 at 2048 -- full
# cosine in both, evaluations at matched token counts.
#
# Any cell whose optimum lands on an edge is not an answer and gets extended;
# that has been the day's most reliable way to waste a grid.
#
# rtx6002: the only node running arrays reliably, see docs/CLUSTER.md.
#
#   sbatch -p rtx scripts/sbatch/adamw2d.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/adamw2d"

T=$SLURM_ARRAY_TASK_ID
CELL=$(( T / 9 )); W=$(( T % 9 )); RI=$(( W / 3 )); AI=$(( W % 3 ))

case $CELL in
  0) OPT=ngd-pion-s; B=512;  ROT=(2e-3 6e-3 2e-2); ADW=(5e-4 1e-3 2e-3) ;;
  1) OPT=ngd-pion-s; B=2048; ROT=(6e-4 2e-3 6e-3); ADW=(1e-3 2e-3 4e-3) ;;
  2) OPT=pion;       B=512;  ROT=(3e-4 1e-3 3e-3); ADW=(5e-4 1e-3 2e-3) ;;
  3) OPT=pion;       B=2048; ROT=(3e-4 1e-3 3e-3); ADW=(1e-3 2e-3 4e-3) ;;
esac
L="${ROT[$RI]}"; A="${ADW[$AI]}"
if [ "$B" = "512" ]; then STEPS=6000; EVERY=500; else STEPS=1500; EVERY=125; fi

echo "node $(hostname)  task $T  cell $CELL  $OPT  batch=$B  rot=$L  adamw=$A  steps=$STEPS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer "$OPT" --lr "$L" --adamw-lr "$A" \
        --batch-sequences "$B" --micro-batch 256 --rho-micro 128 \
        --train-steps "$STEPS" \
        --ngd-t-fac 25 --precision bf16 --seed 0 \
        --log-every 23 --eval-every "$EVERY" --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
