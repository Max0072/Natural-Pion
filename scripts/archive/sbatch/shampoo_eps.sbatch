#!/bin/bash
#SBATCH -A p330
#SBATCH -J sheps
#SBATCH --array=0-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# `shampoo_eps` x `eta`, two-dimensional on purpose.
#
# WHY NOT ONE-DIMENSIONAL. `||X||_2` is set by the directions sitting *on* the
# floor -- they receive the largest boost, `(eps lam_max)^-1/2`, while the top
# eigenvalue is never floored -- so `||X|| ~ eps^-1/2` and changing `eps` at
# fixed `eta` mostly rescales the step, which `eta` then re-absorbs. That is
# exactly how the `ngd-pion-op` damping sweep came back null and was misread as
# "damping is not a lever". Each `eps` row therefore gets its own `eta` sweep.
#
# WHAT `eps` IS HERE, and it is not only damping. The floor is
# `max(lam, eps lam_max)`. At `eps = 1` the whole spectrum collapses to
# `lam_max`, `Q` becomes proportional to the identity and the step is the raw
# generator -- ablated Pion up to a scalar. At `eps -> 0` it is full Shampoo.
# So `eps` traces a continuous path from the control to the method.
#
# PRE-REGISTERED, with a null:
#   * NULL: `eps` only rescales. Then `eta*(eps)` moves as `sqrt(eps)` and the
#     loss AT each row's own optimum is unchanged across rows.
#   * If the optimum improves as `eps` falls, preconditioning more of the
#     spectrum genuinely helps and `eps` is a real lever.
#   * If it degrades as `eps` falls, the floored directions are noise and
#     damping them is the point.
#   * Built-in check: as `eps` rises toward 1 the row's optimum must approach
#     `pion_ablated`'s 6.1143. If it does not, the interpolation claim above is
#     wrong.
#
# Rows are centred on the predicted `eta* = 1e-2 * sqrt(eps / 1e-4)`, taken
# from the `eps = 1e-4` row already measured (jobs 276154, 278352, 278521),
# whose optimum is `eta = 1e-2` at 5.9795. That row is not repeated here.
#
#   sbatch -p rtx -w rtx6002,rtx6003 scripts/sbatch/shampoo_eps.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/shampoo_eps"

#        eps:eta      predicted optimum of each row is the middle entry
PAIRS=(
  1e-6:1e-4  1e-6:3e-4  1e-6:1e-3  1e-6:3e-3  1e-6:1e-2
  1e-2:1e-2  1e-2:3e-2  1e-2:1e-1  1e-2:3e-1  1e-2:1
  1e-1:3e-2  1e-1:1e-1  1e-1:3e-1  1e-1:1     1e-1:3
)
IFS=: read -r EPS ETA <<< "${PAIRS[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  job ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?}  eps=$EPS eta=$ETA"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer shampoo-pion --lr "$ETA" --shampoo-eps "$EPS" \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --shampoo-plane-every 0 \
        --max-steps 150 --log-every 30 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
