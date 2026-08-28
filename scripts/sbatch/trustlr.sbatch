#!/bin/bash
#SBATCH -A p330
#SBATCH -J trustlr
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# The trust region on the thing that scales the step, with the floor kept.
#
# `damped.py` adapted a damping that **replaced** the spectral floor, and job
# 297755 separated the two: the adaptation is worth 0.39 against fixed damping,
# and the substitution costs 0.72 against the floor-based variant. The rule was
# never what failed. So the rule moves to `eta`, which is this variant's
# trust-region radius, and the floor stays.
#
# Measured `rho` on the full-length run: **0.47 at the start, 1.32 by step
# 33000**, crossing 1 near 15000. The step is too long early and too short
# late, in that order, and one scalar cannot be both -- a sweep can only pick
# the compromise between them.
#
# THE DESIGN IS THE SHARP QUESTION, NOT THE COMFORTABLE ONE. Three `eta`: the
# swept optimum, and two deliberately wrong by a factor of ten either way.
#
#   * If the rule works, all three converge, because the radius finds itself
#     and the starting point stops mattering. **That is the claim worth having**
#     -- not "slightly better loss" but "no learning rate to tune".
#   * If only the arm already at the optimum does well, the rule is cosmetic.
#   * `lr_scale` is logged, so "did it move" is answerable without inference.
#     Two experiments today were launched blind to their own adapted quantity.
#
# Against, on disk: the same optimizer at fixed `eta = 2e-2`,
# 4.5947 4.2177 4.0866 4.0124 3.9619 3.9184 at 500..3000.
#
#   sbatch -p b200 scripts/sbatch/trustlr.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/trustlr"

read -r -a ETAS <<< "${ETAS:-2e-3 2e-2 2e-1}"
# The band the rule steers into. K-FAC's `[0.25, 0.75]` is measurably wrong
# here: over 625 unaliased samples `rho` at the swept optimum runs 0.47 rising
# to 1.35, so the *good* rate sits above the upper threshold, the rule reads
# "be bolder" exactly where it should hold, and job 297936 duly settled at 0.15
# from every starting rate. Target `rho ~ 1` instead.
LO="${LO:-0.8}"
HI="${HI:-1.2}"
ETA="${ETAS[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID:-?}  eta0=$ETA band=[$LO,$HI] rho_every=5"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$ETA" \
        --ngd-trust-lr true --rho-every 5 --rho-micro 128 \
        --ngd-trust-lr-lo "$LO" --ngd-trust-lr-hi "$HI" \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --max-steps 3000 --log-every 100 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
