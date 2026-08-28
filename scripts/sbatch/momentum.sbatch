#!/bin/bash
#SBATCH -A p330
#SBATCH -J mom
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Does first-moment averaging move `eta*`, and does it help?
#
# `scripts/probes/split_half_step.py` on the real model at step 24499 puts the
# signal fraction of one 512-sequence step at **0.0357** -- the step is 96.4%
# sampling noise, and two halves of one batch give nearly orthogonal
# directions. `A` and `D` are EMAs but they are the denominator; `powered.py`
# shows `F`'s eigenvalues are exactly the second moments of the generator
# components, so this method is RMSProp and has never had Adam's `m`.
#
# PRE-REGISTERED, and the first item matters whatever the loss does:
#   * An EMA with `beta1 = 0.9` averages `k = 19` draws. If `eta* ~ 2 SNR^2`
#     holds, `eta*` should rise by roughly 19x, from 1e-2 to the order of
#     0.1-0.2. **If `eta*` does not move, that account of `eta` is wrong**, and
#     that is worth knowing separately from whether the loss improves.
#   * The predicted signal fraction under momentum is 0.41, but that assumes the
#     true direction is stable across nineteen steps. This method rotates
#     0.2-0.7 rad per step, so it is an upper bound and the arm may not gain.
#
# 3000 STEPS, NOT 150. The 150-step protocol was measured to invert: for
# `shampoo-pion` at `eps = 1e-6`, `eta = 1e-1` is the worst arm at step 500 and
# the best from 1000 on, while its 150-step optimum was `3e-3`.
#
# NO CONTROL ARM HERE, deliberately. `tests/test_momentum.py` pins that
# `momentum="none"` is bit-identical to `FastNGDPionS`, and `ngd-pion-s` *is*
# `FastNGDPionS`, so the crossover curve for `ngd-pion-s` at `eta = 1e-2`
# (job 279222_4) is already the exact control. Spending a GPU to reproduce a
# deterministic curve would buy nothing.
#
#   sbatch -p rtx -w rtx6002,rtx6004 scripts/sbatch/momentum.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/momentum"

# Doubled on 2026-08-28 with the generator convention, so this still runs
# the sweep it ran before: 1e-2 3e-2 1e-1 3e-1 1 in the old convention.
read -r -a ETAS <<< "${ETAS:-2e-2 6e-2 2e-1 6e-1 2}"
ETA="${ETAS[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID:-?}  eta=$ETA"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-m --lr "$ETA" \
        --ngd-momentum lie --ngd-beta1 0.9 \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --max-steps 3000 --log-every 100 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
