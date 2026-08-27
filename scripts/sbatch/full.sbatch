#!/bin/bash
#SBATCH -A p330
#SBATCH -J full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Full length, 73242 steps. The runs the paper would report.
#
# `ngd-pion-s` leads `pion` at every horizon from 500 to 3000 (+0.133 falling to
# +0.070 against the best of eight `pion` runs, +0.095 against their median) and
# has never been taken to length. The only full-length NGD run used `S = I` at
# `eta = 1.0` -- the variant since measured worse by 0.23-0.32, at an `eta`
# chosen by a protocol since measured to invert.
#
# TWO `eta` PER OPTIMIZER, the incumbent and one step up. Both incumbents were
# chosen at a short horizon, and the short horizon is now known to select too
# small a step: for `shampoo-pion` at `eps = 1e-6`, `eta*` moved from `3e-3` at
# 150 steps to `1e-1` at 3000, a factor of 30, and `1e-1` was the *worst* arm at
# step 500. `3e-1` and `3e-2` are the hedge against the same bias.
#
# NO `--no-resume`. At 73242 steps a wall-clock limit is a real possibility and
# resubmitting the identical command continues from the checkpoint. The run
# directory is keyed by the configuration hash, so the continuation lands in the
# same place by construction.
#
# WHICH POOL, measured rather than argued. A full `pion` run takes 9.4-9.5 h on
# `rtx` and 4.8-5.5 h on `b200` -- 1.9x, from the eight anchor runs already on
# disk, not from a peak-FLOPS ratio. `AGENTS.md` still lists this measurement as
# the first open blocker; it was answered by reading the logs.
#
#   SPECS="ngd-pion-s:-:1e-2 ngd-pion-s:-:3e-2" \
#     sbatch -p b200 --array=0-1 scripts/sbatch/full.sbatch
#   SPECS="shampoo-pion:1e-6:1e-1 shampoo-pion:1e-6:3e-1" \
#     sbatch -p rtx -w rtx6002 --array=0-1 scripts/sbatch/full.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/full"

read -r -a SPECS <<< "${SPECS:?set SPECS to a space-separated list of optimizer:eps:eta}"
IFS=: read -r OPT EPS ETA <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID:-?}  opt=$OPT eps=$EPS eta=$ETA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Every arm shares `seed 0`. Concurrent runs on one node must, or they read
# different corpus windows and halve each other's IOPS.
COMMON="--ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --log-every 100 --eval-every 500 \
        --data-path $DATA/c4_train.bin --val-path $DATA/c4_val.bin --out-dir $OUT"

EXEC="apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH=$REPO \
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --nv --bind $DATA_p330:$DATA_p330 --pwd $REPO $SIF"

if [ "$OPT" = "ngd-pion-s" ]; then
    $EXEC python -m harness.run $COMMON --optimizer ngd-pion-s --lr "$ETA"
else
    $EXEC python -m harness.run $COMMON --optimizer shampoo-pion \
        --lr "$ETA" --shampoo-eps "$EPS" --shampoo-plane-every 0
fi
