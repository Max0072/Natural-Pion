#!/bin/bash
#SBATCH -A p330
#SBATCH -J notrust
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# The isolation nobody ran: `A` and `S` with no trust region at all.
#
# Every NGD run on disk used `trust="quad_curv"`, and it was assumed inert
# because `alpha = quad/curv` is 1 on a fresh basis by algebra. Measured over
# the 24304 layer-steps of the running full-length job, it is **not**: `alpha`
# is below 1 on 37.1% of them, below 0.9 on 13.6%, minimum 0.0262, mean 0.9563.
# So `trust="none"` is a real change and not a no-op, and the preconditioner
# has never been measured without a trust region wrapped around it.
#
# PRE-REGISTERED: removing it lengthens the step -- mean `alpha` 0.9563 -- so
# `eta*` should fall by roughly the same few per cent, and the grid brackets
# the incumbent `2e-2` on both sides rather than assuming it moves.
#   * If the arm matches the control at its own optimum, the trust region is
#     doing nothing a learning rate cannot do, and the honest thing is to drop
#     it from the specification.
#   * If it is worse, `quad/curv` earns its place by shortening the step on
#     exactly the layer-steps where the basis has gone stale -- which is what
#     it structurally can detect and all that it can.
#
# Against, on disk: the same optimizer with `quad_curv` at `eta = 2e-2`,
# 4.5947 4.2177 4.0866 4.0124 3.9619 3.9184 at 500..3000.
#
#   sbatch -p rtx -w rtx6002 scripts/sbatch/notrust.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/notrust"

read -r -a ETAS <<< "${ETAS:-1e-2 2e-2 4e-2}"
ETA="${ETAS[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID:-?}  eta=$ETA trust=none"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$ETA" --ngd-trust none \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --max-steps 3000 --log-every 100 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
