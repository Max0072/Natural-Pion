#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepinitL
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Power-law tail init (alpha 3 beat normal by 0.022 at 3000, journal 2026-11-03 12:20): does it hold with seeds, at length,
# and does it help Pion as much? Our build as in step_norm_long; Pion published, lr 1e-3. Same init for both arms.
#   0-2  ours 3000 @0.2, alpha 3, seeds 1-3 (normal: 3.7176 / 3.7241 / 3.7205)       -t 01:30
#   3-4  ours 3000 @0.2, alpha 2.5 / 4, seed 0 (alpha 3: 3.7020)                      -t 01:30
#   5-6  Pion 3000 / 3840, alpha 3, seed 0 (normal: 3.8412 / 3.7804)                   -t 01:30
#   7    ours 15000 @0.05, alpha 3 (normal 3.4720)                                     -t 04:30
#   8-9  Pion 15000 / 19200, alpha 3 (normal 3.5254 / 3.4960)                          -t 04:30
#   10   ours 56900 @0.05, alpha 3 (normal 3.3468)                                     -t 16:00
#   11   Pion 73242, alpha 3 (normal 3.3706)                                           -t 16:00
# Rule: init gain = val(normal) - val(alpha 3) per arm. It is ours if our gain exceeds Pion's by >= 0.010 at equal steps.
# PREDICTION (low confidence): seeds keep >= 0.010; gain shrinks with length (about 0.01 at 15000); Pion gains about as much.
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
#      0     1     2     3     4     5     6     7      8      9      10     11
KIND=( ours  ours  ours  ours  ours  pion  pion  ours   pion   pion   ours   pion )
STEPS=(3000  3000  3000  3000  3000  3000  3840  15000  15000  19200  56900  73242 )
LR=(   0.2   0.2   0.2   0.2   0.2   1e-3  1e-3  0.05   1e-3   1e-3   0.05   1e-3 )
SEED=( 1     2     3     0     0     0     0     0      0      0      0      0 )
PL=(   3     3     3     2.5   4     3     3     3      3      3      3      3 )
EV=(   500   500   500   500   500   500   500   1000   1000   1000   2000   2000 )
K="${KIND[$T]}"; ST="${STEPS[$T]}"; L="${LR[$T]}"; S="${SEED[$T]}"
N="$K$ST-lr$L-pl${PL[$T]}"; OUT="$DATA_p330/runs/stepnorminitlong/$N"; mkdir -p "$OUT"
echo "node $(hostname) task $T $N seed $S"
COMMON=( --init-pl-alpha "${PL[$T]}" --batch-sequences 512 --micro-batch 256 --rho-micro 128 --precision bf16 --seed "$S" --device cuda
         --log-every 23 --eval-every "${EV[$T]}" --train-steps "$ST" --adamw-lr 8e-3 --data-path "$DATA/c4_train.bin"
         --val-path "$DATA/c4_val.bin" --out-dir "$OUT" )
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
