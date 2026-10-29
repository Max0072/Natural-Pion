#!/bin/bash
#SBATCH -A p330
#SBATCH -J tfac1
#SBATCH --array=0-6
#SBATCH -p rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
# Is alpha = quad/curv a correction for a stale basis, or something more?
# (journal 2026-10-29.)
#
# THE ONE VARIABLE. Every flag is that of notrust_head.sbatch (ablation B: stale
# basis, t_fac 25, alpha pinned at 1, best 4.1969 at rot 2.5e-4), except
# `--ngd-t-fac 1`: the basis is rebuilt every step, so the direction is the fresh
# K-FAC natural gradient F_t^-1 G, and alpha would be 1 by algebra anyway. Same
# rot grid. Task 6 re-runs ablation B's best cell (t_fac 25, rot 2.5e-4) on the
# current code as the determinism check: it must give 4.1969 or the comparison
# is between code states, not between cadences.
#
# THE RULE, fixed before the numbers. V = best val here, with-alpha optimum
# 3.7954, ablation B 4.1969 (difference 0.4015). S = (4.1969 - V) / 0.4015, the
# share of alpha's benefit that a fresh basis recovers without it:
#     S >= 0.8       alpha is a stale-basis correction; a fresh F^-1 needs no alpha
#     0.2 <= S < 0.8 both: report the share
#     S < 0.2        staleness is not the reason; alpha helps beyond it
# A best cell on an edge of the grid means extend before quoting S.
#
#   sbatch scripts/sbatch/tfac1.sbatch
set -uo pipefail
REPO="${REPO:-$DATA_p330/Natural-Pion}"
SIF="${SIF:-$DATA_p330/containers/ngd-pion.sif}"
DATA="${DATA:-$DATA_p330/c4}"
OUT="$DATA_p330/runs/tfac1"

ROT=( 1.25e-4 2.5e-4 5e-4 1e-3 2e-3 4e-3 2.5e-4 )
TFAC=( 1 1 1 1 1 1 25 )
L="${ROT[$SLURM_ARRAY_TASK_ID]}"
T="${TFAC[$SLURM_ARRAY_TASK_ID]}"

echo "node $(hostname)  task $SLURM_ARRAY_TASK_ID  ngd-pion-s trust=none  t_fac=$T  rot=$L  adamw=8e-3  batch=512  steps=3000  seed=0"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=uuid,name,memory.used --format=csv,noheader

apptainer exec --env PYTHONUNBUFFERED=1 --env PYTHONPATH="$REPO" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --nv --bind "$DATA_p330:$DATA_p330" --pwd "$REPO" "$SIF" \
    python -m harness.run \
        --optimizer ngd-pion-s --lr "$L" --adamw-lr 8e-3 --ngd-trust none \
        --batch-sequences 512 --micro-batch 256 --rho-micro 128 \
        --train-steps 3000 \
        --ngd-t-fac "$T" --precision bf16 --seed 0 --device cuda \
        --log-every 23 --eval-every 500 --no-resume \
        --data-path "$DATA/c4_train.bin" --val-path "$DATA/c4_val.bin" \
        --out-dir "$OUT"
