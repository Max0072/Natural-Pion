#!/bin/bash
#SBATCH -A p330
#SBATCH -J steplong
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# What the claim "faster than Pion, by steps and by clock" still needs. Our build = batched t_fac 25 (FAST, BATCH, A and D
# last 6, momentum 0.9, cosine). Pion = published, lr 1e-3, AdamW 8e-3 (our cell; the old full Pion runs used AdamW 1e-3).
#
# Seeds, tasks 0-8 (seed-major, so concurrent tasks share a seed; submit %3), -t 04:30:00:
#   pion3840 / pion19200 / ours15k at angle 0.05, seeds 1-3. Seed 0: Pion 3.7804 / 3.4960, ours 3.7242 (3000) / 3.4720.
#   Clock gap per seed = Pion(our wall time) - ours; ahead if the mean is >= +0.010 and every seed is positive.
# Pion lr, tasks 9-10, -t 01:30:00: lr 5e-4 and 2e-3 at 3840 steps, seed 0 (1e-3: 3.7804). A Pion lr beating 1e-3 by
#   >= 0.010 replaces it as the baseline for the clock claim; a win at the edge extends.
# Task 15, -t 01:30:00: Pion lr 4e-3 at 3840. 2e-3 gave 3.7668, 1e-3 3.7804: 2e-3 wins by 0.014 at the edge, so the grid extends.
# Tasks 16-17, -t 04:30:00: Pion lr 2e-3 / 4e-3 at 19200, seed 0 (1e-3: 3.4960; ours 15000 @0.05: 3.4720). Baseline at
#   19200 = best of the three. PREDICTION: the higher-lr gain is smaller than at 3840; our gap shrinks to ~0.015, stays >= 0.010.
#   sbatch -x rtx6003,rtx6004 -t 04:30:00 --array=16-17 scripts/paper/step_norm_long.sbatch
# Long horizon, tasks 11-14, -t 16:00:00:
#   11 Pion 73242 steps (9.6B tokens; the corpus holds 10B, so Pion cannot run past it), about 9.5 h at 0.466 s.
#   12, 13 ours 56900 steps (73242 x 0.466 / 0.60: Pion's full wall time), angle 0.025 / 0.05. Clock claim at full length.
#   14 ours 73242 steps, angle 0.05: the step claim at full length.
# PREDICTION: seeds keep the sign (ahead at both horizons on every seed); Pion lr 1e-3 stays best within 0.010;
# at full length by clock ahead by less than at 15000 (0.024), possibly level. Low confidence on the last.
#   sbatch -x rtx6003,rtx6004 -t 04:30:00 --array=0-8%3 scripts/paper/step_norm_long.sbatch
#   sbatch -x rtx6003,rtx6004 -t 01:30:00 --array=9-10 scripts/paper/step_norm_long.sbatch
#   sbatch -x rtx6003,rtx6004 -t 16:00:00 --array=11-14 scripts/paper/step_norm_long.sbatch
# Budget curve, tasks 18-22 (user, 2026-11-04), seed 0: points that let scripts/analysis/budget_fit.py interpolate at
#   every Pion target. Ours at 9000 steps, angle 0.1 / 0.05 (best taken, as at every horizon), ours 30000 at 0.05;
#   Pion 30000 and 45000 (round budgets, user); task 23 ours 45000 at 0.05, the pair to Pion's 45000. PREDICTION: power-law and interpolation savings agree within 0.05 once these are in.
#   sbatch -x rtx6003,rtx6004 -t 02:30:00 --array=18-19 scripts/paper/step_norm_long.sbatch
#   sbatch -x rtx6003,rtx6004 -t 07:00:00 --array=20-21 scripts/paper/step_norm_long.sbatch
#   sbatch -x rtx6003,rtx6004 -t 08:00:00 --array=22 scripts/paper/step_norm_long.sbatch
#   sbatch -x rtx6003,rtx6004 -t 10:00:00 --array=23 scripts/paper/step_norm_long.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
#      0        1         2       3        4         5       6        7         8       9           10          11        12         13        14         15     16     17
KIND=( pion     pion      ours    pion     pion      ours    pion     pion      ours    pion        pion        pion      ours       ours      ours       pion  pion  pion  ours ours ours  pion  pion  ours )
STEPS=( 3840    19200     15000   3840     19200     15000   3840     19200     15000   3840        3840        73242     56900      56900     73242      3840  19200 19200 9000 9000 30000 30000 45000 45000 )
LR=(   1e-3     1e-3      0.05    1e-3     1e-3      0.05    1e-3     1e-3      0.05    5e-4        2e-3        1e-3      0.025      0.05      0.05       4e-3  2e-3  4e-3  0.1  0.05 0.05  1e-3  1e-3  0.05 )
SEED=( 1        1         1       2        2         2       3        3         3       0           0           0         0          0         0          0     0     0     0    0    0     0     0     0 )
EV=(   500      1000      1000    500      1000      1000    500      1000      1000    500         500         2000      2000       2000      2000       500   1000  1000  1000 1000 2000  2000  2000  2000 )
K="${KIND[$T]}"; ST="${STEPS[$T]}"; L="${LR[$T]}"; S="${SEED[$T]}"
N="$K$ST-lr$L"; OUT="$DATA_p330/runs/stepnormlong/$N"; mkdir -p "$OUT"
echo "node $(hostname) task $T $N seed $S"
COMMON=( --batch-sequences 512 --micro-batch 256 --rho-micro 128 --precision bf16 --seed "$S" --device cuda --log-every 23
         --eval-every "${EV[$T]}" --train-steps "$ST" --adamw-lr 8e-3 --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT" )
if [ "$K" = pion ]; then
  apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
      python -m harness.run --optimizer pion --lr "$L" "${COMMON[@]}"
else
  apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --env STEP_NORM_A_LAST=6 --env STEP_NORM_D_LAST=6 --env STEP_NORM_FAST=1 --env STEP_NORM_BATCH=1 \
      --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
      python scripts/probes/step_norm.py --optimizer ngd-pion-m --ngd-momentum lie --ngd-beta1 0.9 --lr "$L" --ngd-trust none \
        --ngd-beta 0.5 --ngd-beta-backward 0.5 --ngd-t-fac 25 "${COMMON[@]}"
fi
