#!/bin/bash
#SBATCH -A p330
#SBATCH -J trust
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# A trust region that is one, against the one that is not.
#
# `alpha = quad/curv` shortened the step on **0 of 338 logged steps** of the
# full-length run. It cannot do otherwise: both halves are formed from
# `X = F^-1 G` with the same operator, so on a fresh basis it is 1 by algebra
# and it can only ever read out that the basis has gone stale. `T_fac = 25` is
# conservative enough that it has not.
#
# `trust="exact"` divides by `curv_exact = 4 E_b[s_b^2]`, the curvature measured
# on tokens rather than by the operator that built the step -- what K-FAC does,
# and the only version of the ratio that can notice the model is wrong.
#
# PRE-REGISTERED, before submission:
#   * Measured `alpha_exact/alpha` on this model is 4.7e-3 at step 1 and
#     0.04-0.12 in steady state, so the step shortens by 8 to 25x and `eta*`
#     should rise by about that factor: from 2e-2 to **0.16-0.5**. The grid
#     brackets that on both sides. If `eta*` does not move, the ratio is not
#     acting as a trust region and the flag is inert in a new way.
#   * The cross-layer angle spread should narrow from the 18-61x `ngd-pion-s`
#     shows. **Not to 2-3x**: `alpha_exact` varies only 2-5x between attention
#     and FFN on the real model, so this fixes the time axis firmly and the
#     layer axis partially. Predicting otherwise would be transplanting the
#     toy's 1000x split, which did not reproduce.
#   * `alpha` is a per-step estimate from `ngd_exact_tokens` tokens, so it is
#     noisy. A trust region that jitters is a variance source. If the arm is
#     worse everywhere, smoothing `curv_exact` is the next thing to try, not
#     abandoning the ratio.
#
# `eta` IS IN THE POST-2026-08-28 CONVENTION. The incumbent `1e-2` of the
# earlier sweeps is `2e-2` here; see ALGORITHM.md section 1.
#
# THROTTLED TO TWO AT A TIME (`%2`). Four full-length runs share `b201` and the
# `ngd-pion-s` pair projects to 22.7 h against a 24 h wall -- a 1.3 h margin
# that five concurrent arms would have eaten, since the crossover measured a
# 17% slowdown at that density. The remaining tasks start themselves as the
# Shampoo pair finishes.
#
#   sbatch -p b200 scripts/sbatch/trust.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/trust"

read -r -a ETAS <<< "${ETAS:-6e-2 2e-1 6e-1 2}"
# `BETA` smooths `curv_exact`. The first pass ran it at 0 and lost: measured,
# `alpha` varies 8.4x across layers and 109.5x within one layer over time, so
# the per-layer signal sits under thirteen times as much estimation noise.
# More tokens cannot close that -- the whole batch buys 5.7x against the 13x
# needed -- so the only route is averaging in time, which is what `A` and `D`
# already do and what this one was alone in not doing.
BETA="${BETA:-0}"
ETA="${ETAS[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID:-?}  eta=$ETA  trust=exact  beta=$BETA"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$ETA" \
        --ngd-trust exact --ngd-exact-tokens 4096 --ngd-exact-beta "$BETA" \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --max-steps 3000 --log-every 100 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
