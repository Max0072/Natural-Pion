#!/bin/bash
#SBATCH -A p330
#SBATCH -J fullm
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# `ngd-pion-m` at full length: the run most likely to turn the tie into a result.
#
# `ngd-pion-s` finished inside `pion`'s own eight-run spread -- 3.3868 at 96.4%
# against their 3.3719 to 3.4432. Momentum is worth **+0.045 at 3000 steps** and
# doubles the margin over `pion` there, from 0.070 to 0.115, and it has never
# been taken to length.
#
# TWO RATES, because the 3000-step optimum was a grid edge. That sweep offered
# `2e-2 6e-2 2e-1 6e-1 2` and `2e-2` won, so the optimum is at or below the
# bottom of it and a single arm would sit on an edge again. `1e-2` brackets from
# below.
#
# No `--no-resume`: 73242 steps against a 24 h wall is close, and resubmitting
# the identical command continues from the checkpoint.
#
#   sbatch -p rtx scripts/sbatch/full_m.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/full"

ETAS=(2e-2 1e-2)
ETA="${ETAS[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID:-?}  ngd-pion-m eta=$ETA"
nvidia-smi --query-gpu=name --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-m --lr "$ETA" \
        --ngd-momentum lie --ngd-beta1 0.9 \
        --ngd-t-fac 25 --adamw-lr 1e-3 --precision bf16 --seed 0 \
        --log-every 97 --eval-every 500 \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
