#!/bin/bash
#SBATCH -A p330
#SBATCH -J alphafrz
#SBATCH --array=0-3
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Does a frozen profile of alpha do what live alpha does? (journal 2026-10-30.)
#
# THE ONE VARIABLE. The with-alpha arm (rot 6e-3, t_fac 25, seed 0, val 3.7954), with
# live alpha replaced by a table from seeds 1-3, so the profile is out of sample:
#   0 time    alpha's mean by 100-step bin
#   1 layer   alpha's mean by layer, weighted by the cosine rate
#   2 both    by layer and bin, the closest a frozen table can come
#   3 ones    identity check, rot 2.5e-4, 50 steps: train_loss must equal
#             runs/tfac1/ngd-pion-s-lr0.00025-s0-549e192a1a
#
# THE RULE, fixed before the numbers. Pion 3.8412. R = (3.8412 - val) / 0.0458, the
# share of NGD-Pion's lead over Pion that the table reproduces:
#     R >= 0.8  this table carries the lead; live alpha is not needed for it
#     0.2-0.8   partly; report the share
#     R < 0.2   it does not
# `both` below 0.8 means the live, step-to-step part of alpha matters, whatever the
# others do. Seed noise about 0.005, so R is good to about 0.1-0.2; one seed.
#
# PREDICTION. both: R >= 0.8. time: near both, since time is alpha's largest factor.
# layer: fails, it has no early shrink and the stale basis cannot take full early
# rates (ablation B). A layer failure at this one rate says little about layers;
# both minus time is the layer effect's measure. No rate grid: this replays
# the rates the arm really used.
#
#   sbatch -x rtx6004 scripts/sbatch/alpha_frozen.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"

MODES=( time layer both ones )
ROT=( 6e-3 6e-3 6e-3 2.5e-4 )
EXTRA=( "" "" "" "--max-steps 50" )
M="${MODES[$SLURM_ARRAY_TASK_ID]}"
L="${ROT[$SLURM_ARRAY_TASK_ID]}"
OUT="$DATA_p330/runs/alphafrozen/$M"
mkdir -p "$OUT"

echo "node $(hostname)  task $SLURM_ARRAY_TASK_ID  mode=$M  rot=$L  t_fac=25  adamw=8e-3  batch=512  seed=0"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env ALPHA_FROZEN_MODE="$M" \
    --env ALPHA_FROZEN_TABLE="$REPO/runs-record/alphafrozen/profile.json" \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python scripts/probes/alpha_frozen.py \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --train-steps 3000 ${EXTRA[$SLURM_ARRAY_TASK_ID]} \
        --ngd-t-fac 25 --precision bf16 --seed 0 --device cuda \
        --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
