#!/bin/bash
#SBATCH -A p330
#SBATCH -J lm
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# The trust region the way the literature actually does it.
#
# Today's attempt adapted a **per-layer, per-step** multiplier from a
# subsampled curvature estimate, and it failed on noise: the per-layer signal
# is 8.4x and the estimation noise within one layer is 109.5x.
#
# kfac-jax does the opposite on all three axes: **one global scalar**, adapted
# **every 5 steps**, from `rho = [L(t+1) - L(t)] / [q(v) - q(0)]` -- a real loss
# evaluation, not a ratio of two model quantities. Damping is what moves, not
# the step multiplier: `rho < 0.25` raises it, `rho > 0.75` lowers it.
#
# **All of that is already implemented here and was starved.** `damped.py`
# carries the rule with the same thresholds, `harness.train` already calls
# `adapt_damping(rho)` -- but `rho` was computed only on logged steps, so at
# `log_every = 100` the rule fired once per hundred steps, and the run that
# judged it a dead end was **150 steps long**. The mechanism adapted about five
# times in its entire evaluation. `rho_every` decouples the measurement from
# logging; this uses 5, as kfac-jax does.
#
# PRE-REGISTERED:
#   * Measured `rho` on the full-length run has median **1.22**, starting at
#     0.47 and rising through 1 around step 15000. Above 0.75 the rule lowers
#     `lam`, so it should decay steadily and the step should lengthen. **If
#     `lam` does not move, the rule is still starved and the fault is upstream
#     of the rule.**
#   * `eta` is meant to sit still while `lam` carries the trust region, which is
#     the whole claim. `eta = 2` is `damped.py`'s intended value in the
#     post-2026-08-28 convention; the fourth arm holds `eta` at the swept
#     optimum instead, so that "the damping is doing it" is separable from
#     "the step happened to be better".
#   * The comparison is `ngd-pion-s` at its own optimum: 4.5947 4.2177 4.0866
#     4.0124 3.9619 3.9184 at 500..3000.
#
#   sbatch -p b200 scripts/sbatch/lm.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/lm"

# `eta:lam:adapt`. **The adapt=false arms are the control and are not
# optional.** `ngd-pion-damped` is not `ngd-pion-s` plus adaptation: it drops
# the spectral floor entirely and lets `lam` replace it, so comparing it
# against `ngd-pion-s` confounds two changes. Holding `lam` fixed at the same
# value isolates the one being tested.
SPECS=(2:0.1:false 2:0.1:true 2:1:false 2:1:true)
IFS=: read -r ETA LAM ADAPT <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID:-?}  eta=$ETA lam=$LAM adapt=$ADAPT rho_every=5"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-damped --lr "$ETA" \
        --ngd-lam "$LAM" --ngd-lam-adapt "$ADAPT" --rho-every 5 \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --max-steps 3000 --log-every 100 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
