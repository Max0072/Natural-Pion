#!/bin/bash
#SBATCH -A p330
#SBATCH -J stepfin
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# The combined build with batched F^-1 and Cayley (STEP_NORM_FAST, _BATCH, A and D last 6, momentum 0.9, cosine), 3000 steps.
#   0      t_fac 10, angle 0.2, seed 0          clock, on rtx6006 like Pion's 5300
#   1, 2   Pion, published, seed 0, 3840 / 4500 steps: Pion's runs at the wall time of batch t_fac 25 (1789 s at 0.466 s)
#          and of t_fac 10 batched (estimated 0.70 s/step, 2100 s). Direct clock comparison, no interpolation. Same node.
#   3-8    seeds 1-3: t_fac 25 angle 0.2, t_fac 10 angle 0.2, in pairs of one seed
# Seed 0 so far: t_fac 25 batched 3.7242 at 0.596 s/step; t_fac 10 unbatched 3.7034 at 0.80.
# Clock rule per build: Pion(at our wall time) - ours >= +0.010 ahead, |.| < 0.010 level, <= -0.010 behind.
# Seeds: mean and spread against Pion seeds 0-3 at 3000 (3.8394) for the step count claim.
# PREDICTION: t_fac 25 batched ahead (Pion at 3840 about 3.77 by interpolation), t_fac 10 batched about 0.70 s/step, ahead by more.
#   sbatch -x rtx6003,rtx6004 -w rtx6006 --array=0-2 scripts/sbatch/step_norm_final.sbatch
#   sbatch -x rtx6003,rtx6004 --array=3-8%2 scripts/sbatch/step_norm_final.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
T=$SLURM_ARRAY_TASK_ID
NAME=( tf10 pion3840 pion4500 tf25 tf10 tf25 tf10 tf25 tf10 ); SEED=( 0 0 0 1 1 2 2 3 3 ); TF=( 10 0 0 25 10 25 10 25 10 )
N="${NAME[$T]}"; S="${SEED[$T]}"
OUT="$DATA_p330/runs/stepnormfinal/$N"; mkdir -p "$OUT"
echo "node $(hostname) task $T $N seed $S"
COMMON=( --batch-sequences 512 --micro-batch 256 --rho-micro 128 --precision bf16 --seed "$S" --device cuda --log-every 23 --eval-every 500 --no-resume
         --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" --out-dir "$OUT" )
if [ "$N" = pion3840 ] || [ "$N" = pion4500 ]; then
  apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
      python -m harness.run --optimizer pion --lr 1e-3 --adamw-lr 8e-3 --train-steps "${N#pion}" "${COMMON[@]}"
else
  apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --env STEP_NORM_A_LAST=6 --env STEP_NORM_D_LAST=6 --env STEP_NORM_FAST=1 --env STEP_NORM_BATCH=1 \
      --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
      python scripts/probes/step_norm.py --optimizer ngd-pion-m --ngd-momentum lie --ngd-beta1 0.9 --lr 0.2 --adamw-lr 8e-3 --ngd-trust none \
        --ngd-beta 0.5 --ngd-beta-backward 0.5 --train-steps 3000 --ngd-t-fac "${TF[$T]}" "${COMMON[@]}"
fi
