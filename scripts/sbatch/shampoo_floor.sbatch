#!/bin/bash
#SBATCH -A p330
#SBATCH -J shfloor
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Two controls, and the left edge of the two rows job 278661 failed to bracket.
#
# WHY THE CONTROLS. Across the `eps x eta` grid a great many arms land at
# 6.10-6.17, and `pion_ablated` sits at 6.1090 to 6.1376 while its `eta` moves
# by a factor of 500 -- the signature of a rotation that is contributing
# nothing. If that band is simply what AdamW reaches on its own over the
# parameters Pion does not own, then every loss quoted at 150 steps has to be
# read as a contribution *above* it, and `pion_ablated` is not an isolating
# baseline but a zero.
#
#   * `adamw` -- AdamW owning every parameter. The ordinary reference.
#   * `shampoo-pion` at `eta = 1e-12` -- the rotation switched off numerically,
#     so the matrices stay at initialisation and AdamW trains only the rest.
#     This is the one that measures the band directly, because it differs from
#     the grid arms in exactly one thing: whether the rotation does anything.
#
# WHY THE EXTENSION. `eta*` was predicted to move as `sqrt(eps)` and does not:
# measured `eta*` spans at most 10x over five orders of `eps`, against the 300x
# predicted. `P` is built from the gradients themselves, so a direction with a
# small `P` eigenvalue is one where the gradient was small, and the floor's
# boost meets a small numerator. The `eps = 1e-2` and `1e-1` rows were centred
# on the wrong prediction and their minima sit on the low edge; these are the
# points below it.
#
#   sbatch -p rtx scripts/sbatch/shampoo_floor.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"

EXEC="apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH=$REPO \
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --nv --bind $DATA_p330:$DATA_p330 --pwd $REPO $SIF"
# `FORCE=1` adds `--force`, which takes a run directory's lock from a holder
# that is gone. A cancelled job leaves `.run.lock` behind, and the harness then
# refuses to start -- correctly, since two trainers in one directory interleave
# their logs and overwrite each other's checkpoints. Set it only after checking
# with `squeue` that the named job really is dead.
FORCE_FLAG=""
[ "${FORCE:-0}" = "1" ] && FORCE_FLAG="--force"

COMMON="--ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --shampoo-plane-every 0 --max-steps 150 --log-every 30 --no-resume \
        $FORCE_FLAG \
        --data-path $DATA/c4_train.bin --val-path $DATA/c4_val.bin"

#      kind:eps:eta
SPECS=(
  control:adamw:1e-3
  control:1e-4:1e-12
  grid:1e-2:3e-4  grid:1e-2:1e-3  grid:1e-2:3e-3
  grid:1e-1:1e-3  grid:1e-1:3e-3  grid:1e-1:1e-2
)
IFS=: read -r KIND EPS ETA <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID:-?}  kind=$KIND eps=$EPS eta=$ETA"
nvidia-smi --query-gpu=name --format=csv,noheader

if [ "$EPS" = "adamw" ]; then
    OUT="$DATA_p330/runs/control"
    $EXEC python -m harness.run $COMMON --optimizer adamw --lr "$ETA" --out-dir "$OUT"
else
    OUT="$DATA_p330/runs/control"
    [ "$KIND" = "grid" ] && OUT="$DATA_p330/runs/shampoo_eps"
    $EXEC python -m harness.run $COMMON --optimizer shampoo-pion \
        --lr "$ETA" --shampoo-eps "$EPS" --out-dir "$OUT"
fi
