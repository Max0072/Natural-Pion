#!/bin/bash
#SBATCH -A p330
#SBATCH -J cross
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Where does the rotation start paying for itself?
#
# At 150 steps, freezing the matrices at initialisation reaches 5.7071 and beats
# every rotational arm except `ngd-pion-s`: `shampoo-pion` at its swept optimum
# is 0.23 worse, `ngd-pion` 0.17 worse, `pion_ablated` 0.40 worse. Frozen
# matrices leave 32.9M of 60M trainable and must lose eventually -- the
# full-length `ngd-pion` run reaches 3.6728 -- so there is a crossover, and
# nobody knows where it is. Every sweep in this project was measured at 150
# steps, which is 0.2% of the schedule.
#
# 3000 steps is 4% of it, and `eval_every = 500` makes the deliverable a curve
# rather than an endpoint: val at 500, 1000, 1500, 2000, 2500, 3000.
#
# THREE `eta`, NOT ONE. `eta = 3e-3` is the optimum of the `eps = 1e-6` row --
# measured on the 150-step proxy this run exists to question. Carrying only it
# would trust that proxy twice. The neighbours bracket it, so if the optimum
# moves with horizon, the movement is visible rather than assumed.
#
# `ngd-pion-s` is here as the reference arm, not as scope creep: it is the only
# optimizer on the board above the do-nothing line, and whether its 0.20 lead
# grows or decays with horizon decides which method the paper carries.
#
# PRE-REGISTERED:
#   * If `shampoo-pion` crosses the inert control before ~1000 steps, the
#     150-step proxy is pessimistic but usable with care.
#   * If it crosses late or not at all by 3000, the 150-step protocol cannot
#     rank these optimizers and every `eta` in this project was tuned in a
#     regime the real run spends 0.2% of its time in.
#   * The `eta` that leads at 3000 need not be the one that led at 150. If it
#     is a neighbour, the sweeps have to be redone at horizon.
#
#   sbatch -p rtx -w rtx6002,rtx6003 scripts/sbatch/crossover.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/crossover"

EXEC="apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH=$REPO \
      --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --nv --bind $DATA_p330:$DATA_p330 --pwd $REPO $SIF"
# Every arm shares `seed 0`, which is a requirement and not a convention: with
# different seeds concurrent runs read different corpus windows and halve each
# other's IOPS.
COMMON="--ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --shampoo-plane-every 0 --max-steps 3000 --log-every 100 \
        --eval-every 500 --no-resume \
        --data-path $DATA/c4_train.bin --val-path $DATA/c4_val.bin --out-dir $OUT"

#      optimizer:eps:eta
SPECS=(
  shampoo-pion:1e-6:1e-12      # the do-nothing line
  shampoo-pion:1e-6:1e-3
  shampoo-pion:1e-6:3e-3       # the 150-step optimum
  shampoo-pion:1e-6:1e-2
  ngd-pion-s:-:1e-2            # the only arm above the line at 150 steps
)
IFS=: read -r OPT EPS ETA <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID:-?}  opt=$OPT eps=$EPS eta=$ETA"
nvidia-smi --query-gpu=name --format=csv,noheader

if [ "$OPT" = "ngd-pion-s" ]; then
    $EXEC python -m harness.run $COMMON --optimizer ngd-pion-s --lr "$ETA"
else
    $EXEC python -m harness.run $COMMON --optimizer shampoo-pion \
        --lr "$ETA" --shampoo-eps "$EPS"
fi
