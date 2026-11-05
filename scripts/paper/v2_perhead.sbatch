#!/bin/bash
#SBATCH -A p330
#SBATCH -J npv2
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Section 5 redone (user, 2026-11-05) with the query projection rotated per head, as Pion does, and the final
# Natural Pion Fast (one A per input, Newton-Schulz in Z - I, batched sigma, bf16 statistics). Pion runs unchanged.
# Each task is a row of the tables below: TASK STEPS ANGLE SEED EVAL. Tasks are added as the budgets are fixed
# (the equal-time ones from the step time of tasks 0-3).
#   sbatch -x rtx6003,rtx6004 -t 01:00:00 --array=0-3 scripts/paper/v2_perhead.sbatch
REPO="${REPO:-$DATA_p330/Natural-Pion}"; SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"; DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
# 0-3: 3000 steps, seeds 0-3          4-7: 15000 steps, seeds 0-3          8-10: budget curve 9000 / 30000 / 45000
# 11: full length, 73 242 steps (Pion's budget). Equal-time comparisons are read off the budget curve (user: no
# dedicated equal-time runs).
#   sbatch ... -t 01:00:00 --array=0-3 ; -t 04:00:00 --array=4-7 ; -t 03:00:00 --array=8 ; -t 07:00:00 --array=9 ;
#   -t 10:00:00 --array=10 ; -t 16:00:00 --array=11
#        0    1    2    3    4     5     6     7     8     9     10    11
STEPS=(3000 3000 3000 3000 15000 15000 15000 15000 9000  30000 45000 73242)
LR=(   0.2  0.2  0.2  0.2  0.05  0.05  0.05  0.05  0.05  0.05  0.05  0.05 )
SEED=( 0    1    2    3    0     1     2     3     0     0     0     0    )
EV=(   500  500  500  500  1000  1000  1000  1000  1000  2000  2000  2000 )
OUT="$DATA_p330/runs/npv2/s${STEPS[$T]}"; mkdir -p "$OUT"
apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run --optimizer natural-pion-fast --np-retraction ns_delta --np-stats-dtype bf16 \
      --np-split-q-per-head true --lr "${LR[$T]}" --ngd-beta1 0.9 --train-steps "${STEPS[$T]}" --batch-sequences 512 \
      --micro-batch 256 --rho-micro 128 --precision bf16 --seed "${SEED[$T]}" --device cuda --log-every 23 \
      --eval-every "${EV[$T]}" --adamw-lr 8e-3 --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT"
